import json
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, AsyncIterator, Callable, Any, Literal, Protocol
import asyncio

class WebSocketLike(Protocol):
    async def send_text(self, message: str): ...

    async def send_json(self, message: str): ...

    async def close(self): ...

EventPosition = Literal["first", "last", "middle"]
MessageType = Literal["logs", "report", "chat", "text"]

class SSEWebSocketAdapter(WebSocketLike):
    def __init__(self, formatter: Callable[[str, EventPosition, MessageType], str]):
        self.queue = asyncio.Queue()
        self.completed = False  # Mimic WebSocket connection state
        self.formatter = formatter
        self.is_first = False

    async def send_text(self, message: str):
        return await self._send_text(message, "text")

    async def _send_text(self, message: str, message_type: MessageType):
        if self.completed:
            raise RuntimeError("connection is closed.")
        format_type: EventPosition = "middle"
        if self.is_first:
            format_type = "first"
            self.is_first = False
        await self.queue.put(self.formatter(message, format_type, message_type))

    async def send_json(self, message: str):
        data = json.loads(message)

    async def close(self):
        await self.queue.put(self.formatter("", "last", "text"))
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
            await self.close()
            raise
