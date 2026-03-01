# Remote Devbox Orchestration

## Concept
One Telegram bot manages multiple remote devboxes via SSH.
Each devbox = separate forum topic. Claude Code runs on the devbox, not locally.

```
Telegram Topic "feature-auth"  ──→  Bot  ──SSH──→  devbox-1 (claude CLI)
Telegram Topic "feature-cart"  ──→  Bot  ──SSH──→  devbox-2 (claude CLI)
```

## Decisions
- **SSH auth**: SSH key on bot's server (path in config)
- **Scope**: Only Claude Code (no arbitrary commands)
- **Claude Code**: Already installed on devboxes
- **Multiplexing**: Forum topics (1 devbox = 1 topic)

---

## Phase 1: Data Model + Storage
Store devboxes in SQLite (dynamic, registered via Telegram commands).

```
DevboxModel:
  - name (slug, unique)
  - ssh_host
  - ssh_user (default: root)
  - ssh_key_path (override or use global default)
  - remote_work_dir (default: /app)
  - owner_user_id
  - chat_id + message_thread_id (forum topic)
  - status (active / unreachable / removed)
  - created_at, last_used_at
```

New: `src/devbox/models.py`, `src/devbox/repository.py`
Modify: `src/storage/database.py` (migration), `src/storage/facade.py`

---

## Phase 2: SSH Executor
Run `claude` CLI on remote devbox over SSH, stream results back.

```
Bot sends prompt
  → asyncssh connects to devbox
  → runs: claude -p "prompt" --output-format stream-json [--resume session_id]
  → streams JSON lines back
  → parse with parse_message() from claude-agent-sdk
  → returns ClaudeResponse (same interface as local)
```

Interrupts: send SIGINT over SSH channel.
Connection pooling: reuse SSH connections between messages.

New: `src/devbox/executor.py` (RemoteClaudeExecutor)
Modify: `src/claude/facade.py` (route to remote when devbox context present)

---

## Phase 3: Telegram Commands

```
/devbox add <name> <ssh_host> [user] [work_dir]
  → saves to DB
  → tests SSH connectivity + claude --version
  → creates forum topic named "devbox: <name>"

/devbox list
  → shows all devboxes with status

/devbox remove <name>
  → marks as removed, archives topic

/devbox status [name]
  → SSH ping + claude --version check
```

New: `src/devbox/commands.py`
Modify: `src/bot/orchestrator.py` (register commands)

---

## Phase 4: Forum Topic Routing
When a message arrives in a devbox topic, route it to that devbox.

```
Message in topic
  → orchestrator resolves thread_id → DevboxModel
  → creates RemoteClaudeExecutor for that devbox
  → runs prompt on remote claude
  → sends response back to same topic
```

Sessions: per-devbox (session_id stored alongside devbox in thread state).
Progress: same "Working..." / spinner as local, just runs remotely.

New: `src/devbox/manager.py` (DevboxManager — topic creation, resolution)
Modify: `src/bot/orchestrator.py` (extend `_apply_thread_routing_context`)

---

## Phase 5: Config + Wiring

```env
ENABLE_DEVBOX=true
DEVBOX_SSH_KEY_PATH=/root/.ssh/id_ed25519
DEVBOX_SSH_DEFAULT_USER=root
DEVBOX_DEFAULT_WORK_DIR=/app
```

Modify: `src/config/settings.py`, `src/config/features.py`, `src/main.py`, `pyproject.toml` (+asyncssh)

---

## Implementation Order
1. Phase 1 (model + storage) — foundation
2. Phase 2 (SSH executor) — core feature, can test standalone
3. Phase 3 (commands) — user-facing registration
4. Phase 4 (topic routing) — ties it all together
5. Phase 5 (config) — can be done incrementally alongside phases 1-4
