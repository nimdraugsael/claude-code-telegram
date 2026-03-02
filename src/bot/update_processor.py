"""Selective-concurrency update processor for PTB.

Regular updates (messages, commands) process sequentially -- one at a time.
Priority callbacks (stop:*, plan:*, ask:*) bypass the queue and run
immediately so they can interrupt the currently-running handler via
asyncio.Event.
"""

import asyncio
from typing import Any, Awaitable

from telegram import Update
from telegram.ext._baseupdateprocessor import BaseUpdateProcessor

_PRIORITY_PREFIXES = ("stop:", "plan:", "ask:")


class StopAwareUpdateProcessor(BaseUpdateProcessor):
    """Update processor that lets priority callbacks bypass sequential processing.

    PTB calls ``process_update(update, coroutine)`` for every incoming update.
    The base class holds a semaphore (max 256) then calls our
    ``do_process_update()``.

    For priority callbacks (stop:*, plan:*, ask:*): we just ``await coroutine``
    -- runs immediately.
    For everything else: we acquire ``_sequential_lock`` first -- only one
    runs at a time.

    A stop callback arrives while a text handler holds the lock -> stop
    callback runs concurrently -> fires the ``asyncio.Event`` -> the watcher
    task inside ``execute_command()`` calls ``client.interrupt()`` -> Claude
    stops -> ``run_command()`` returns -> handler finishes -> lock released.
    """

    def __init__(self) -> None:
        # High limit so priority callbacks are never blocked by semaphore
        super().__init__(max_concurrent_updates=256)
        self._sequential_lock = asyncio.Lock()

    @staticmethod
    def _is_priority_callback(update: object) -> bool:
        """Return True if the update is a priority callback query."""
        if not isinstance(update, Update):
            return False
        cb = update.callback_query
        if cb is None or cb.data is None:
            return False
        return any(cb.data.startswith(p) for p in _PRIORITY_PREFIXES)

    async def do_process_update(
        self,
        update: object,
        coroutine: Awaitable[Any],
    ) -> None:
        """Process an update, applying sequential lock for non-priority updates."""
        if self._is_priority_callback(update):
            # Run immediately -- no sequential lock
            await coroutine
        else:
            # One at a time for everything else
            async with self._sequential_lock:
                await coroutine

    async def initialize(self) -> None:
        """Initialize the processor (no-op)."""

    async def shutdown(self) -> None:
        """Shutdown the processor (no-op)."""
