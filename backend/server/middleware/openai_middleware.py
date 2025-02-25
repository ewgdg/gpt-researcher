from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    messages: List[Message]


class ChatCompletionsResponse(BaseModel):
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


@router.post("/chat/completions", response_model=ChatCompletionsResponse)
async def create_completion(request: ChatCompletionsRequest):
    # Implement your logic to handle the request and generate a response
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "service_tier": "default",
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
    }
    return response
