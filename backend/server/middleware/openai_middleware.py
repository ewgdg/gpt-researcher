import asyncio
import json
import time
from typing import List, Optional, cast
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gpt_researcher.config.config import Config
from gpt_researcher.utils.llm import get_llm

from ..sse_websocket_adapter import FormatType, SSEWebSocketAdapter

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]


class ModelsResponse(BaseModel):
    object: str
    data: List[dict]


class ModelResponse(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    # Implement your logic to list available engines
    response = {
        "object": "list",
        "data": [
            {
                "id": "gpt-researcher",
                "object": "model",
                "created": 0,
                "owned_by": "gpt-researcher",
            },
        ],
    }
    return response


@router.get("/models/{model}", response_model=ModelResponse)
async def retrieve_model(model: str):
    # Implement your logic to retrieve a specific engine
    response = {
        "id": model,
        "object": "model",
        "created": 0,
        "owned_by": "gpt-researcher",
    }
    return response

async def test_stream(response: SSEWebSocketAdapter):
    await response.send_text("test1\n")
    await asyncio.sleep(1)
    await response.send_text("test2\n")
    await asyncio.sleep(1)
    await response.send_text("test3\n")
    await asyncio.sleep(1)
    await response.complete()


async def test_gen():
    for i in range(3):
        yield f"data: test{i}\n\n"
        await asyncio.sleep(1)


from langchain.schema import HumanMessage, AIMessage, SystemMessage


def convert_to_langchain_messages(messages):
    langchain_messages = []
    for message in messages:
        if message.role == "user":
            langchain_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            langchain_messages.append(AIMessage(content=message.content))
        elif message.role == "system":
            langchain_messages.append(SystemMessage(content=message.content))
        else:
            raise ValueError(f"Unsupported role: {message.role}")
    return langchain_messages


cfg = Config()
@router.post("/chat/completions", response_model=ChatCompletionsResponse)
async def create_completion(request: ChatCompletionsRequest):
    print("testing")
    print(request)
    response_id = str(uuid.uuid4())
    created_time = int(time.time())

    if not request.stream:
        smart_agent = get_llm(
            llm_provider=cfg.smart_llm_provider,
            model=cfg.smart_llm_model,
            temperature=0.35,
            max_tokens=cfg.smart_token_limit,  # type: ignore
            **cfg.llm_kwargs,
        )
        response = await smart_agent.get_chat_response(
            messages=convert_to_langchain_messages(request.messages), stream=False
        )
        # return the response not stream
        json_data = {
            "id": response_id,
            "object": "chat.completion",
            "created": created_time,
            "model": "gpt-researcher",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        }
        return json_data

    # manager.start_sender()

    def format_message(message: str, format_type: FormatType) -> str:
        delta = {"content": message} if message else {}
        if format_type == "first":
            delta["role"] = "assistant"
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "gpt-researcher",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "stop" if format_type == "last" else None,
                }
            ],
        }
        return json.dumps(chunk)

    streaming_response = SSEWebSocketAdapter(format_message)
    asyncio.create_task(test_stream(streaming_response))

    return StreamingResponse(streaming_response, media_type="text/event-stream")
