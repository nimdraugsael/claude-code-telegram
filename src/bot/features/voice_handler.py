"""
Handle voice message transcription via OpenAI Whisper API.

Flow: Voice message -> download OGG -> Whisper API -> TranscribedVoice
"""

import io
from dataclasses import dataclass, field
from typing import Any, Dict

import structlog
from openai import AsyncOpenAI
from telegram import Voice

from src.config import Settings

logger = structlog.get_logger(__name__)

# Whisper API file size limit (25 MB)
_MAX_FILE_SIZE = 25 * 1024 * 1024

# Whisper pricing: $0.006 per minute
WHISPER_COST_PER_MINUTE = 0.006


@dataclass
class TranscribedVoice:
    """Result of voice message transcription."""

    text: str
    duration_seconds: int
    file_size: int
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VoiceHandler:
    """Transcribe Telegram voice messages using OpenAI Whisper."""

    # Default vocabulary hints for Whisper transcription.
    # Technical terms that Whisper commonly misrecognizes.
    _DEFAULT_PROMPT = (
        "pull requests, merge, commit, push, branch, rebase, "
        "repository, GitHub, CI/CD, Docker, Kubernetes, "
        "API, SDK, CLI, REST, GraphQL, WebSocket, "
        "Python, TypeScript, JavaScript, Rust, Go, "
        "Claude, Anthropic, OpenAI, Whisper, Telegram"
    )

    def __init__(self, config: Settings) -> None:
        api_key = config.openai_api_key_str
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for VoiceHandler")
        self.client = AsyncOpenAI(api_key=api_key)
        self.whisper_prompt = (
            getattr(config, "whisper_prompt", None) or self._DEFAULT_PROMPT
        )

    async def process_voice(self, voice: Voice) -> TranscribedVoice:
        """Download and transcribe a Telegram voice message."""
        if voice.file_size and voice.file_size > _MAX_FILE_SIZE:
            raise ValueError(
                f"Voice message too large ({voice.file_size / 1024 / 1024:.1f}MB). "
                f"Whisper limit is 25MB."
            )

        file = await voice.get_file()
        audio_bytes = await file.download_as_bytearray()

        if len(audio_bytes) > _MAX_FILE_SIZE:
            raise ValueError(
                f"Voice message too large ({len(audio_bytes) / 1024 / 1024:.1f}MB). "
                f"Whisper limit is 25MB."
            )

        logger.info(
            "Transcribing voice message",
            file_size=len(audio_bytes),
            duration=voice.duration,
        )

        # OGG/Opus is natively supported by Whisper — no conversion needed
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "voice.ogg"

        transcription = await self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt=self.whisper_prompt,
        )

        text = transcription.text.strip() if transcription.text else ""

        logger.info(
            "Voice transcription complete",
            text_length=len(text),
            duration=voice.duration,
        )

        duration = voice.duration if isinstance(voice.duration, int) else 0
        whisper_cost = max(duration, 1) / 60.0 * WHISPER_COST_PER_MINUTE

        return TranscribedVoice(
            text=text,
            duration_seconds=duration,
            file_size=len(audio_bytes),
            cost=whisper_cost,
            metadata={
                "mime_type": voice.mime_type or "audio/ogg",
            },
        )
