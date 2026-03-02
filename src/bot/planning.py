"""Planning tool support — ExitPlanMode & AskUserQuestion.

Intercepts Claude Code planning tool calls from the stream, presents
plans and questions to the user via Telegram (inline keyboards), collects
answers, and resumes the Claude session with the user's response.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

logger = structlog.get_logger()

# Tool names we care about
PLANNING_TOOLS: frozenset = frozenset(
    {"EnterPlanMode", "ExitPlanMode", "AskUserQuestion"}
)

# Subset that should interrupt the stream and require user interaction
INTERRUPT_PLANNING_TOOLS: frozenset = frozenset({"ExitPlanMode", "AskUserQuestion"})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def find_latest_plan_file(working_dir: str) -> Optional[Path]:
    """Find the newest ``.md`` plan file written by Claude Code.

    Searches ``{working_dir}/.claude/plans/`` first, then falls back to
    ``~/.claude/plans/``.
    """
    candidates: list[Path] = []
    for base in [Path(working_dir), Path.home()]:
        plans_dir = base / ".claude" / "plans"
        if plans_dir.is_dir():
            candidates.extend(plans_dir.glob("*.md"))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def detect_planning_tool(tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the first interrupt-worthy planning tool call, or ``None``."""
    for tc in tool_calls:
        if tc.get("name") in INTERRUPT_PLANNING_TOOLS:
            return tc
    return None


# ---------------------------------------------------------------------------
# PlanningState
# ---------------------------------------------------------------------------


@dataclass
class PlanningState:
    """Serialisable state for an in-progress planning interaction."""

    type: str  # "plan_approval" | "user_question"
    session_id: Optional[str] = None
    working_directory: str = ""
    chat_id: Optional[int] = None

    # -- plan_approval fields --
    plan_file_path: Optional[str] = None
    awaiting_changes_text: bool = False

    # -- user_question fields --
    questions: Optional[List[Dict[str, Any]]] = None
    answers: Optional[Dict[str, List[str]]] = None
    current_question_idx: int = 0
    awaiting_other_text: bool = False
    other_question_idx: Optional[int] = None
    question_message_id: Optional[int] = None

    # -- multi-select toggle state --
    multi_select_chosen: Optional[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def _save_state(context: ContextTypes.DEFAULT_TYPE, state: PlanningState) -> None:
    context.user_data["planning_state"] = state.to_dict()


def _clear_state(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("planning_state", None)


def _load_state(context: ContextTypes.DEFAULT_TYPE) -> Optional[PlanningState]:
    raw = context.user_data.get("planning_state")
    if raw is None:
        return None
    return PlanningState.from_dict(raw)


# ---------------------------------------------------------------------------
# PlanningHandler
# ---------------------------------------------------------------------------


class PlanningHandler:
    """Presents plans/questions and collects user responses."""

    # -- Plan approval --------------------------------------------------

    @staticmethod
    async def present_plan(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        state: PlanningState,
        plan_content: str = "",
    ) -> None:
        """Send the plan as an ``.md`` file and approval buttons."""
        message = update.effective_message
        user_id = update.effective_user.id

        sent = False

        # 1. Prefer the plan file on disk — Claude writes the full plan to
        #    the file *before* calling ExitPlanMode, so it is always complete.
        #    The streamed plan_content may be truncated because we interrupt
        #    the SDK as soon as we detect the tool call.
        plan_path = find_latest_plan_file(state.working_directory)
        if plan_path and plan_path.is_file():
            state.plan_file_path = str(plan_path)
            _save_state(context, state)
            try:
                await message.reply_document(
                    document=plan_path,
                    caption="Claude has prepared a plan.",
                )
                sent = True
            except Exception as exc:
                logger.warning("Failed to send plan file", error=str(exc))

        # 2. Fallback: use streamed plan_content (may be partial)
        if not sent and plan_content and plan_content.strip():
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".md",
                    prefix="plan_",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    tmp.write(plan_content)
                    tmp_path = Path(tmp.name)
                await message.reply_document(
                    document=tmp_path,
                    caption="Claude has prepared a plan.",
                )
                sent = True
                tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to send plan content as file", error=str(exc))

        # 3. Nothing available
        if not sent:
            await message.reply_text(
                "Claude has prepared a plan (no content available)."
            )

        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Approve",
                        callback_data=f"plan:{user_id}:approve",
                    ),
                    InlineKeyboardButton(
                        "Request Changes",
                        callback_data=f"plan:{user_id}:changes",
                    ),
                    InlineKeyboardButton(
                        "Reject",
                        callback_data=f"plan:{user_id}:reject",
                    ),
                ]
            ]
        )
        await message.reply_text("What would you like to do?", reply_markup=kb)

    # -- Question presentation ------------------------------------------

    @staticmethod
    async def present_question(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        state: PlanningState,
    ) -> None:
        """Show the current question with option buttons."""
        message = update.effective_message
        user_id = update.effective_user.id
        questions = state.questions or []

        if state.current_question_idx >= len(questions):
            # All answered — should not happen, but guard
            return

        q = questions[state.current_question_idx]
        q_idx = state.current_question_idx
        q_text = q.get("question", "Question?")
        header = q.get("header", "")
        options = q.get("options", [])
        is_multi = q.get("multiSelect", False)

        # Build display text
        lines = []
        if header:
            lines.append(f"<b>{_escape(header)}</b>")
        lines.append(f"\n{_escape(q_text)}")
        for i, opt in enumerate(options):
            label = opt.get("label", f"Option {i+1}")
            desc = opt.get("description", "")
            lines.append(f"\n<b>{i+1}.</b> {_escape(label)}")
            if desc:
                lines.append(f"   <i>{_escape(desc)}</i>")

        text = "\n".join(lines)

        # Build keyboard
        chosen = state.multi_select_chosen or []
        rows: list[list[InlineKeyboardButton]] = []
        for i, opt in enumerate(options):
            label = opt.get("label", f"Option {i+1}")
            if is_multi:
                prefix = "\u2611" if i in chosen else "\u2610"
                label = f"{prefix} {label}"
            rows.append(
                [
                    InlineKeyboardButton(
                        label,
                        callback_data=f"ask:{user_id}:{q_idx}:{i}",
                    )
                ]
            )

        # "Other..." button
        rows.append(
            [
                InlineKeyboardButton(
                    "Other...",
                    callback_data=f"ask:{user_id}:{q_idx}:other",
                )
            ]
        )

        # "Done" for multi-select
        if is_multi:
            rows.append(
                [
                    InlineKeyboardButton(
                        "Done \u2713",
                        callback_data=f"ask:{user_id}:{q_idx}:done",
                    )
                ]
            )

        kb = InlineKeyboardMarkup(rows)

        sent = await message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        state.question_message_id = sent.message_id
        _save_state(context, state)

    # -- Callback handlers ----------------------------------------------

    @staticmethod
    async def handle_plan_callback(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        resume_fn: Any,
    ) -> None:
        """Handle plan:{user_id}:{action} callbacks."""
        query = update.callback_query
        parts = query.data.split(":")
        if len(parts) < 3:
            await query.answer("Invalid callback")
            return

        target_user_id = int(parts[1])
        action = parts[2]

        if query.from_user.id != target_user_id:
            await query.answer("Not your plan.", show_alert=True)
            return

        state = _load_state(context)
        if not state or state.type != "plan_approval":
            await query.answer("No pending plan.", show_alert=True)
            return

        await query.answer()

        # Remove buttons
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        if action == "approve":
            _clear_state(context)
            await query.message.reply_text("Plan approved. Proceeding...")
            await resume_fn(
                update,
                context,
                state,
                "User approved the plan. Proceed with implementation.",
            )

        elif action == "changes":
            state.awaiting_changes_text = True
            _save_state(context, state)
            await query.message.reply_text(
                "Please type your feedback / requested changes:"
            )

        elif action == "reject":
            _clear_state(context)
            await query.message.reply_text("Plan rejected.")

    @staticmethod
    async def handle_ask_callback(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        resume_fn: Any,
    ) -> None:
        """Handle ask:{user_id}:{q_idx}:{opt_idx|other|done} callbacks."""
        query = update.callback_query
        parts = query.data.split(":")
        if len(parts) < 4:
            await query.answer("Invalid callback")
            return

        target_user_id = int(parts[1])
        q_idx = int(parts[2])
        choice = parts[3]

        if query.from_user.id != target_user_id:
            await query.answer("Not your question.", show_alert=True)
            return

        state = _load_state(context)
        if not state or state.type != "user_question":
            await query.answer("No pending question.", show_alert=True)
            return

        questions = state.questions or []
        if q_idx >= len(questions):
            await query.answer("Question expired.", show_alert=True)
            return

        q = questions[q_idx]
        options = q.get("options", [])
        is_multi = q.get("multiSelect", False)

        if choice == "other":
            await query.answer()
            state.awaiting_other_text = True
            state.other_question_idx = q_idx
            _save_state(context, state)
            await query.message.reply_text("Please type your answer:")
            return

        if choice == "done" and is_multi:
            # Finalize multi-select
            await query.answer()
            chosen = state.multi_select_chosen or []
            labels = [
                options[i].get("label", f"Option {i+1}")
                for i in chosen
                if i < len(options)
            ]
            if not labels:
                await query.answer("Select at least one option.", show_alert=True)
                return

            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass

            _record_answer(state, q, labels)
            _advance_question(state)

            if state.current_question_idx >= len(questions):
                # All done
                prompt = _format_answers_prompt(state)
                _clear_state(context)
                await query.message.reply_text("Thanks! Resuming...")
                await resume_fn(update, context, state, prompt)
            else:
                state.multi_select_chosen = []
                _save_state(context, state)
                await PlanningHandler.present_question(update, context, state)
            return

        # Numeric option index
        try:
            opt_idx = int(choice)
        except ValueError:
            await query.answer("Invalid option")
            return

        if opt_idx >= len(options):
            await query.answer("Invalid option")
            return

        if is_multi:
            # Toggle
            chosen = state.multi_select_chosen or []
            if opt_idx in chosen:
                chosen.remove(opt_idx)
            else:
                chosen.append(opt_idx)
            state.multi_select_chosen = chosen
            _save_state(context, state)

            # Re-render buttons
            await query.answer()
            rows: list[list[InlineKeyboardButton]] = []
            user_id = query.from_user.id
            for i, opt in enumerate(options):
                label = opt.get("label", f"Option {i+1}")
                prefix = "\u2611" if i in chosen else "\u2610"
                rows.append(
                    [
                        InlineKeyboardButton(
                            f"{prefix} {label}",
                            callback_data=f"ask:{user_id}:{q_idx}:{i}",
                        )
                    ]
                )
            rows.append(
                [
                    InlineKeyboardButton(
                        "Other...",
                        callback_data=f"ask:{user_id}:{q_idx}:other",
                    )
                ]
            )
            rows.append(
                [
                    InlineKeyboardButton(
                        "Done \u2713",
                        callback_data=f"ask:{user_id}:{q_idx}:done",
                    )
                ]
            )
            try:
                await query.edit_message_reply_markup(
                    reply_markup=InlineKeyboardMarkup(rows)
                )
            except Exception:
                pass
            return

        # Single-select
        await query.answer()
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        label = options[opt_idx].get("label", f"Option {opt_idx+1}")
        _record_answer(state, q, [label])
        _advance_question(state)

        if state.current_question_idx >= len(questions):
            prompt = _format_answers_prompt(state)
            _clear_state(context)
            await query.message.reply_text("Thanks! Resuming...")
            await resume_fn(update, context, state, prompt)
        else:
            state.multi_select_chosen = []
            _save_state(context, state)
            await PlanningHandler.present_question(update, context, state)

    # -- Text intercept (for "Other..." and "Request Changes") ----------

    @staticmethod
    async def handle_text_for_planning(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        state: PlanningState,
        resume_fn: Any,
    ) -> bool:
        """Handle free-text input when we're awaiting it.

        Returns True if the message was consumed.
        """
        text = update.message.text or ""

        if state.type == "plan_approval" and state.awaiting_changes_text:
            state.awaiting_changes_text = False
            _clear_state(context)
            await update.message.reply_text("Sending feedback to Claude...")
            await resume_fn(
                update,
                context,
                state,
                f"User requested changes to the plan:\n\n{text}",
            )
            return True

        if state.type == "user_question" and state.awaiting_other_text:
            questions = state.questions or []
            q_idx = (
                state.other_question_idx
                if state.other_question_idx is not None
                else state.current_question_idx
            )
            q = questions[q_idx] if q_idx < len(questions) else {}

            state.awaiting_other_text = False
            state.other_question_idx = None

            _record_answer(state, q, [text])
            _advance_question(state)

            if state.current_question_idx >= len(questions):
                prompt = _format_answers_prompt(state)
                _clear_state(context)
                await update.message.reply_text("Thanks! Resuming...")
                await resume_fn(update, context, state, prompt)
            else:
                state.multi_select_chosen = []
                _save_state(context, state)
                await PlanningHandler.present_question(update, context, state)
            return True

        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _record_answer(
    state: PlanningState,
    question: Dict[str, Any],
    labels: List[str],
) -> None:
    """Record answer labels for a question."""
    if state.answers is None:
        state.answers = {}
    q_text = question.get("question", f"Q{state.current_question_idx}")
    state.answers[q_text] = labels


def _advance_question(state: PlanningState) -> None:
    state.current_question_idx += 1


def _format_answers_prompt(state: PlanningState) -> str:
    """Build a prompt string from collected answers."""
    if not state.answers:
        return "User provided no answers."
    lines = ["User answered the questions as follows:"]
    for q_text, labels in state.answers.items():
        lines.append(f"\nQ: {q_text}")
        lines.append(f"A: {', '.join(labels)}")
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Minimal HTML escaping for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
