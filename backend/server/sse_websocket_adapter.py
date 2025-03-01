from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, AsyncIterator, Callable, Any, Literal
import asyncio

FormatType = Literal["first", "last"] | None


# define an interface for SocketLike objects
class SocketLike:
    async def send_text(self, message: str):
        pass

    async def send_json(self, message: str):
        pass

    async def complete(self):
        pass

class SSEWebSocketAdapter:
    def __init__(self, formatter: Callable[[str, FormatType], str]):
        self.queue = asyncio.Queue()
        self.completed = False  # Mimic WebSocket connection state
        self.formatter = formatter
        self.is_first = False

    async def send_text(self, message: str):
        if self.completed:
            raise RuntimeError("connection is closed.")
        format_type: FormatType = None
        if self.is_first:
            format_type = "first"
            self.is_first = False
        await self.queue.put(self.formatter(message, format_type))

    async def complete(self):
        await self.queue.put(self.formatter("", "last"))
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
