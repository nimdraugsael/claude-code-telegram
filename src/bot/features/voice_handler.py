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


@dataclass
class TranscribedVoice:
    """Result of voice message transcription."""

    text: str
    duration_seconds: int
    file_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class VoiceHandler:
    """Transcribe Telegram voice messages using OpenAI Whisper."""

    def __init__(self, config: Settings) -> None:
        api_key = config.openai_api_key_str
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for VoiceHandler")
        self.client = AsyncOpenAI(api_key=api_key)

    async def process_voice(self, voice: Voice) -> TranscribedVoice:
        """Download and transcribe a Telegram voice message.

        Args:
            voice: Telegram Voice object from the incoming message.

        Returns:
            TranscribedVoice with the transcribed text and metadata.

        Raises:
            ValueError: If the file exceeds the 25 MB Whisper limit.
            Exception: On Whisper API or download errors.
        """
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
        )

        text = transcription.text.strip() if transcription.text else ""

        logger.info(
            "Voice transcription complete",
            text_length=len(text),
            duration=voice.duration,
        )

        duration = voice.duration if isinstance(voice.duration, int) else 0

        return TranscribedVoice(
            text=text,
            duration_seconds=duration,
            file_size=len(audio_bytes),
            metadata={
                "mime_type": voice.mime_type or "audio/ogg",
            },
        )
