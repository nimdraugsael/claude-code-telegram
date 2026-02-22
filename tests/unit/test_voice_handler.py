"""Tests for voice message transcription via OpenAI Whisper."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.features.voice_handler import (
    TranscribedVoice,
    VoiceHandler,
    _MAX_FILE_SIZE,
)


# --- TranscribedVoice dataclass ---


class TestTranscribedVoice:
    """Verify the TranscribedVoice data container."""

    def test_basic_construction(self):
        result = TranscribedVoice(
            text="Hello world",
            duration_seconds=5,
            file_size=1024,
        )
        assert result.text == "Hello world"
        assert result.duration_seconds == 5
        assert result.file_size == 1024
        assert result.metadata == {}

    def test_metadata_defaults_to_empty_dict(self):
        r1 = TranscribedVoice(text="a", duration_seconds=1, file_size=100)
        r2 = TranscribedVoice(text="b", duration_seconds=2, file_size=200)
        # Each instance should get its own dict (no shared mutable default)
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata

    def test_custom_metadata(self):
        result = TranscribedVoice(
            text="test",
            duration_seconds=3,
            file_size=512,
            metadata={"mime_type": "audio/ogg", "extra": True},
        )
        assert result.metadata["mime_type"] == "audio/ogg"
        assert result.metadata["extra"] is True


# --- VoiceHandler construction ---


class TestVoiceHandlerInit:
    """Verify VoiceHandler requires a valid OpenAI API key."""

    def test_raises_without_api_key(self):
        config = MagicMock()
        config.openai_api_key_str = None
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            VoiceHandler(config=config)

    def test_raises_with_empty_api_key(self):
        config = MagicMock()
        config.openai_api_key_str = ""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            VoiceHandler(config=config)

    @patch("src.bot.features.voice_handler.AsyncOpenAI")
    def test_creates_client_with_key(self, mock_openai_cls):
        config = MagicMock()
        config.openai_api_key_str = "sk-test-key-123"

        handler = VoiceHandler(config=config)

        mock_openai_cls.assert_called_once_with(api_key="sk-test-key-123")
        assert handler.client is mock_openai_cls.return_value


# --- process_voice ---


class TestProcessVoice:
    """Verify the voice download + Whisper transcription flow."""

    @pytest.fixture
    def handler(self):
        """VoiceHandler with a mocked OpenAI client."""
        with patch("src.bot.features.voice_handler.AsyncOpenAI") as mock_cls:
            config = MagicMock()
            config.openai_api_key_str = "sk-test"
            h = VoiceHandler(config=config)
            h.client = mock_cls.return_value
            return h

    @staticmethod
    def _make_voice(
        *,
        file_size: int = 1024,
        duration: int = 5,
        mime_type: str = "audio/ogg",
        audio_bytes: bytes = b"\x00" * 1024,
    ) -> MagicMock:
        """Create a mock Telegram Voice object."""
        voice = MagicMock()
        voice.file_size = file_size
        voice.duration = duration
        voice.mime_type = mime_type

        mock_file = AsyncMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(audio_bytes)
        )
        voice.get_file = AsyncMock(return_value=mock_file)
        return voice

    async def test_happy_path(self, handler):
        """Successful transcription returns text with metadata."""
        voice = self._make_voice(
            file_size=2048,
            duration=10,
            audio_bytes=b"\xff" * 2048,
        )

        # Mock Whisper response
        transcription = MagicMock()
        transcription.text = "Hello, this is a test message."
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)

        assert result.text == "Hello, this is a test message."
        assert result.duration_seconds == 10
        assert result.file_size == 2048
        assert result.metadata["mime_type"] == "audio/ogg"

        # Verify Whisper was called with correct params
        call_kwargs = handler.client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-1"
        # The file should be a BytesIO with name "voice.ogg"
        audio_file = call_kwargs["file"]
        assert isinstance(audio_file, io.BytesIO)
        assert audio_file.name == "voice.ogg"

    async def test_file_size_rejected_before_download(self, handler):
        """Files exceeding 25MB are rejected via file_size metadata (pre-download)."""
        voice = self._make_voice(file_size=30 * 1024 * 1024)

        with pytest.raises(ValueError, match="too large"):
            await handler.process_voice(voice)

        # Should not even attempt to download
        voice.get_file.assert_not_called()

    async def test_file_size_rejected_after_download(self, handler):
        """Files exceeding 25MB are rejected after download (actual size check)."""
        # file_size metadata says it's small, but actual bytes are too large
        big_bytes = b"\x00" * (26 * 1024 * 1024)
        voice = self._make_voice(file_size=1024, audio_bytes=big_bytes)

        with pytest.raises(ValueError, match="too large"):
            await handler.process_voice(voice)

        # Download happened but Whisper was NOT called
        voice.get_file.assert_called_once()
        handler.client.audio.transcriptions.create.assert_not_called()

    async def test_file_size_exactly_at_limit(self, handler):
        """File exactly at 25MB should be accepted."""
        exact_bytes = b"\x00" * _MAX_FILE_SIZE
        voice = self._make_voice(
            file_size=_MAX_FILE_SIZE,
            audio_bytes=exact_bytes,
        )
        transcription = MagicMock()
        transcription.text = "At the limit"
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)
        assert result.text == "At the limit"

    async def test_file_size_one_byte_over_limit(self, handler):
        """File one byte over 25MB should be rejected."""
        over_bytes = b"\x00" * (_MAX_FILE_SIZE + 1)
        voice = self._make_voice(
            file_size=_MAX_FILE_SIZE + 1,
            audio_bytes=over_bytes,
        )

        with pytest.raises(ValueError, match="too large"):
            await handler.process_voice(voice)

    async def test_none_file_size_skips_pre_check(self, handler):
        """When voice.file_size is None, skip pre-download check and rely on post-download."""
        voice = self._make_voice(audio_bytes=b"\x00" * 512)
        voice.file_size = None

        transcription = MagicMock()
        transcription.text = "No pre-check"
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)
        assert result.text == "No pre-check"

    async def test_empty_transcription_returns_empty_string(self, handler):
        """Whisper returning empty text yields empty string."""
        voice = self._make_voice()

        transcription = MagicMock()
        transcription.text = ""
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)
        assert result.text == ""

    async def test_none_transcription_returns_empty_string(self, handler):
        """Whisper returning None text yields empty string."""
        voice = self._make_voice()

        transcription = MagicMock()
        transcription.text = None
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)
        assert result.text == ""

    async def test_whitespace_transcription_stripped(self, handler):
        """Leading/trailing whitespace in transcription is stripped."""
        voice = self._make_voice()

        transcription = MagicMock()
        transcription.text = "  hello world  \n"
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)
        assert result.text == "hello world"

    async def test_missing_mime_type_defaults_to_ogg(self, handler):
        """When voice.mime_type is None, metadata defaults to audio/ogg."""
        voice = self._make_voice(mime_type=None)
        # Telegram Voice mock needs explicit None
        voice.mime_type = None

        transcription = MagicMock()
        transcription.text = "test"
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)
        assert result.metadata["mime_type"] == "audio/ogg"

    async def test_non_integer_duration_defaults_to_zero(self, handler):
        """Non-integer duration is replaced with 0."""
        voice = self._make_voice()
        voice.duration = "not-an-int"

        transcription = MagicMock()
        transcription.text = "test"
        handler.client.audio.transcriptions.create = AsyncMock(
            return_value=transcription,
        )

        result = await handler.process_voice(voice)
        assert result.duration_seconds == 0

    async def test_whisper_api_error_propagates(self, handler):
        """Errors from the Whisper API propagate to the caller."""
        voice = self._make_voice()

        handler.client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("Whisper API unavailable"),
        )

        with pytest.raises(Exception, match="Whisper API unavailable"):
            await handler.process_voice(voice)

    async def test_download_error_propagates(self, handler):
        """Errors downloading the voice file propagate to the caller."""
        voice = self._make_voice()
        voice.get_file = AsyncMock(side_effect=Exception("Telegram download failed"))

        with pytest.raises(Exception, match="Telegram download failed"):
            await handler.process_voice(voice)


# --- Constants ---


class TestConstants:
    """Verify module-level constants."""

    def test_max_file_size_is_25mb(self):
        assert _MAX_FILE_SIZE == 25 * 1024 * 1024
