from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, AsyncIterator, Callable, Any
import asyncio


class SSEWebSocketAdapter:
    def __init__(self, formatter: Callable[[str], str]):
        self.queue = asyncio.Queue()
        self.completed = False  # Mimic WebSocket connection state
        self.formatter = formatter

    async def send_text(self, message: str):
        if self.completed:
            raise RuntimeError("connection is closed.")
        await self.queue.put(self.formatter(message))

    async def complete(self):
        await self.queue.put(self.formatter(""))
        self.completed = True

    def __aiter__(self) -> AsyncIterator[str]:
        """Make this object directly async iterable."""
        return self

    async def __anext__(self) -> str:
        """Yield messages from the queue."""
        try:
            if self.completed and self.queue.empty():
                raise StopAsyncIteration
            message = await self.queue.get()
            return f"data: {message}\n\n"
        except asyncio.CancelledError:
            await self.complete()
            raise
