import json
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generator,
    Literal,
    Protocol,
)
import asyncio

class WebSocketLike(Protocol):
    async def send_text(self, message: str): ...

    async def send_json(self, data: dict): ...

    async def close(self): ...

EventPosition = Literal["first", "last", "middle"]
MessageType = Literal["logs", "report", "chat", "text"]

class SSEWebSocketAdapter(WebSocketLike):
    def __init__(self, formatter: Callable[[str, EventPosition, MessageType], str]):
        self.queue: asyncio.Queue[tuple[str, EventPosition, MessageType]] = (
            asyncio.Queue()
        )
        self.completed = False  # Mimic WebSocket connection state
        self.formatter = formatter
        self.is_first = True

    async def send_text(self, message: str):
        if len(message) == 0:
            return
        return await self._send_text(message, "text")

    async def _send_text(self, message: str, message_type: MessageType):
        if self.completed:
            raise RuntimeError("connection is closed.")
        format_type: EventPosition = "middle"
        if self.is_first:
            format_type = "first"
            self.is_first = False
        await self.queue.put((message, format_type, message_type))

    async def send_json(self, data: dict):
        message_type = data.get("type", "text")
        if message_type == "path":
            message_type = "logs"

        if message_type == "logs":
            message = data.get("output", "")
        else:
            message = data.get("content", "")

        await self._send_text(
            message,
            message_type,
        )

    async def complete(self, message: str):
        if self.completed:
            return
        await self.queue.put((message, "last", "text"))
        self.completed = True

    async def close(self):
        if self.is_first:
            await self.send_text("Error: Connection closed before any data was sent.")
        await self.complete("\n")

    async def get_full_message(self) -> str:
        sb = []
        while not self.completed and self.queue.empty():
            message, _, _ = await self.queue.get()
            sb.append(message)
        return self.formatter("".join(sb), "last", "text")

    def __await__(self) -> Generator[Any, None, str]:
        return self.get_full_message().__await__()

    def __aiter__(self) -> AsyncIterator[str]:
        """Make this object directly async iterable."""
        return self

    async def __anext__(self) -> str:
        """Yield messages from the queue."""
        try:
            if self.completed and self.queue.empty():
                raise StopAsyncIteration
            message, position, message_type = await self.queue.get()
            formatted = self.formatter(message, position, message_type)
            return f"data: {formatted}\n\n"
        except asyncio.CancelledError:
            await self.close()
            raise