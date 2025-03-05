import asyncio
from datetime import datetime
import itertools
import json
import logging
import re
import textwrap
import time
from typing import (
    Annotated,
    AsyncIterator,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    cast,
    TypeAlias,
)
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    BaseMessageChunk,
    BaseMessage,
    AIMessageChunk,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langgraph.graph import Graph, MessagesState
from langgraph.func import task, entrypoint
from gpt_researcher.config.config import Config
from gpt_researcher.utils.enum import Tone
from gpt_researcher.utils.llm import get_llm
from ..sse_websocket_adapter import EventPosition, SSEWebSocketAdapter, MessageType

router = APIRouter()
logger = logging.getLogger(__name__)


class JsonMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[JsonMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
    metadata: Optional[dict] = None


def json_to_langchain_messages(messages: List[JsonMessage]) -> List[BaseMessage]:
    langchain_messages = []
    for message in messages:
        if message.role == "user":
            langchain_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            langchain_messages.append(AIMessage(content=message.content))
        elif message.role == "system":
            # langchain_messages.append(SystemMessage(content=message.content))
            pass
        else:
            raise ValueError(f"Unsupported role: {message.role}")
    return langchain_messages


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
    await response.close()


async def test_gen():
    for i in range(3):
        yield f"data: test{i}\n\n"
        await asyncio.sleep(1)


def route(messages):
    # "current time is {}..."
    ...


cfg = Config()
low_temp_agent = cast(
    BaseChatModel,
    get_llm(
        llm_provider=cfg.smart_llm_provider,
        model=cfg.smart_llm_model,
        # temperature=0,
        # top_p=0.8,
        max_tokens=cfg.smart_token_limit,  # type: ignore
        **cfg.llm_kwargs,
    ).llm,
)

low_temp_chain = low_temp_agent

smart_chat_model = cast(
    BaseChatModel,
    get_llm(
        llm_provider=cfg.smart_llm_provider,
        model=cfg.smart_llm_model,
        max_tokens=cfg.smart_token_limit,  # type: ignore
        **cfg.llm_kwargs,
    ).llm,
)

fast_chat_model = cast(
    BaseChatModel,
    get_llm(
        llm_provider=cfg.fast_llm_provider,
        model=cfg.fast_llm_model,
        temperature=0,
        top_p=0.8,
        max_tokens=cfg.fast_token_limit,  # type: ignore
        **cfg.llm_kwargs,
    ).llm,
)

ReportTypes = {
    "research_report": "Summary - Short and fast (~2 min)",
    "detailed_report": "Detailed - longer (~5 min)",
    "deep": "Deep Research Report",
    "multi_agents": "Multi Agents Report",
}

ReportTypeLiteral: TypeAlias = Annotated[
    str,
    ...,
    "Must match one of the following report type literals: "
    "- research_report: A brief summary, ~2 min.\n"
    "- deep: Extensive research.\n"
    "- multi_agents: Multiple perspectives.\n"
    "- detailed_report: In-depth, ~5 min.",
]

ReportToneLiteral: TypeAlias = Annotated[
    str,
    ...,
    f"Must match one of the tone literals: {', '.join(f"'{tone.name}'" for tone in Tone)}",
]


class ResearchTaskMetadata(BaseModel):
    report_tone: ReportToneLiteral
    report_type: ReportTypeLiteral


@tool
def start_research(
    research_query: str,
    report_type: ReportTypeLiteral,
    report_tone: ReportToneLiteral,
):
    """Start a research task on the query"""
    return json.dumps(
        {
            "research_query": research_query,
            "report_tone": report_tone,
            "report_type": report_type,
        },
        indent=2,
    )


str_output_parser = StrOutputParser()

TriageResultValue: TypeAlias = Literal["fast", "research"]


class TriageResult(BaseModel):
    value: TriageResultValue = Field(
        ...,
        description=(
            "'fast' agent for very simple tasks like `generating a title`. "
            "'research' agent for tasks that require research workflow or external data."
        ),
    )


triage_agent = (
    ChatPromptTemplate.from_messages(
        (
            SystemMessage(
                "Today is {today_date}. decide which agent to handle the request."
            ),
            MessagesPlaceholder("messages"),
        )
    )
    | fast_chat_model.with_structured_output(TriageResult)
    | RunnableLambda[TriageResult, TriageResultValue](lambda r: r.value)
)

research_tools = [start_research]
research_model = smart_chat_model.bind_tools(research_tools)


async def triage_request(
    messages: List[BaseMessage],
) -> tuple[Literal[TriageResultValue, "follow_up"], ResearchTaskMetadata | None]:
    pattern = r"^\s*<research_task_metadata>([\s\S]*?)<\/research_task_metadata>"
    for message in messages:
        if isinstance(message, AIMessage):
            content = str_output_parser.invoke(message)
            if content.startswith("<research_task_metadata>"):
                match = re.search(pattern, content)
                if match:
                    print(match.group(1))
                    json_data = json.loads(match.group(1))
                    return "follow_up", ResearchTaskMetadata.model_validate(json_data)

    today_date = datetime.now().strftime("%Y-%m-%d")
    result = await triage_agent.ainvoke(
        {"today_date": today_date, "messages": messages}
    )
    return result, None


async def prepare_research(
    messages: List[BaseMessage],
    sse_writer: SSEWebSocketAdapter,
    request: ChatCompletionsRequest,
):
    chunks = research_model.astream(messages)
    tool_call_message = None
    async for chunk in chunks:
        if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
            if tool_call_message is None:
                tool_call_message = chunk
            else:
                tool_call_message = cast(AIMessageChunk, tool_call_message + chunk)
            continue
        else:
            content = str_output_parser.invoke(chunk)
            if content:
                await sse_writer.send_text(content)
    if tool_call_message:
        print(tool_call_message.tool_calls)
        if not request.stream:
            await sse_writer.send_text(
                "Please enable streaming response for this request."
            )
            return
        tool_call = tool_call_message.tool_calls[0]
        if tool_call["name"] == "start_research":
            print(type(tool_call), isinstance(tool_call, dict), tool_call.get("type"))
            tool_message: ToolMessage = start_research.invoke(tool_call)
            await sse_writer.send_text(
                f"<research_task_metadata>\n"
                f"{tool_message.content}\n"
                "</research_task_metadata>\n"
            )

            await sse_writer.send_json({"type": "logs", "output": "testing output1"})

            await sse_writer.send_json(
                {
                    "type": "report",
                    "content": "test content2",
                }
            )

            await sse_writer.send_json(
                {
                    "type": "logs",
                    "output": "testing output3",
                }
            )


async def workflow(
    request: ChatCompletionsRequest,
    sse_writer: SSEWebSocketAdapter,
):
    try:
        messages = json_to_langchain_messages(request.messages)
        triage_result, research_task_metadata = await triage_request(messages)
        if triage_result == "fast":
            async for chunk in fast_chat_model.astream(messages):
                await sse_writer.send_text(str_output_parser.invoke(chunk))
            return
        elif triage_result == "research":
            await prepare_research(messages, sse_writer, request)
            return
        elif triage_result == "follow_up":
            await sse_writer.send_text("This is a follow up request.")
            return
        else:
            raise ValueError(f"Unknown triage result: {triage_result}")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        await sse_writer.send_text(f"Error: {e}")
    finally:
        await sse_writer.close()


class SSEMessageFormatter:
    def __init__(self, response_id: str, created_time: int, stream: bool):
        self.response_id = response_id
        self.created_time = created_time
        self.stream = stream
        self.previous_message_type = None
        self.thinking_done = False
        self.is_thinking = False

    def format_chunk(
        self, message: str, event_position: EventPosition, message_type: MessageType
    ) -> str:
        if self.previous_message_type != message_type:
            if message_type == "logs":
                if self.thinking_done:
                    # thinking is done, ignore the message
                    logger.warning("Ignoring log messages after thinking is done.")
                    message = ""
                else:
                    self.is_thinking = True
                    message = f"<think>\n{message}"
            elif self.is_thinking:
                self.is_thinking = False
                self.thinking_done = True
                message = f"\n</think>\n{message}"

        delta = {"content": message} if message else {}
        if event_position == "first":
            delta["role"] = "assistant"
        chunk = {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created_time,
            "model": "gpt-researcher",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "stop" if event_position == "last" else None,
                }
            ],
        }

        self.previous_message_type = message_type
        return json.dumps(chunk)

    def format_full_message(self, message: str) -> str:
        return json.dumps(
            {
                "id": self.response_id,
                "object": "chat.completion",
                "created": self.created_time,
                "model": "gpt-researcher",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": message,
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            }
        )

    def format(
        self, message: str, event_position: EventPosition, message_type: MessageType
    ) -> str:
        if not self.stream:
            return self.format_full_message(message)
        return self.format_chunk(message, event_position, message_type)


@router.post("/chat/completions", response_model=ChatCompletionsResponse)
async def create_completion(request: ChatCompletionsRequest):
    print("testing")
    print(request)
    response_id = str(uuid.uuid4())
    created_time = int(time.time())
    # print(start_research.get_input_jsonschema())
    # print(start_research.get_output_jsonschema())
    # print(start_research.get_prompts())
    # return "test"

    sse_formater = SSEMessageFormatter(
        response_id, created_time, request.stream or False
    )
    sse_writer = SSEWebSocketAdapter(sse_formater.format)

    asyncio.create_task(workflow(request, sse_writer))
    if request.stream:
        return StreamingResponse(sse_writer, media_type="text/event-stream")
    return await sse_writer

    today_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = SystemMessage(
        "Today is {today_date}. Start a research if you cannot fulfill the request confidently."
    )
    messages = json_to_langchain_messages(request.messages)
    # prompt = ChatPromptTemplate.from_messages((system_prompt, MessagesPlaceholder("messages")))

    tools = [start_research]
    model_with_tools = smart_chat_model.bind_tools(tools)

    agent_response = cast(
        AIMessage, await model_with_tools.ainvoke((system_prompt, *messages))
    )
    if agent_response.tool_calls:
        pass

    # <research_task_metadata>
    research_task_metadata = textwrap.dedent(f"""
        {{
            "tone": one of [{",".join(tone.name for tone in Tone)}],
            "report_type": one of {ReportTypes},
        }}
    """).strip()

    response_schemas = [
        ResponseSchema(
            name="action",
            description="Action to take: 'fulfill_request', 'start_research', or 'request_parameters'",
            type="string",
        ),
        ResponseSchema(
            name="research_metadata",
            description=f"Required if action is 'start_research'. Must strictly follow format: {research_task_metadata}",
            type="json",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    query = messages[-1].content
    messages.append(
        HumanMessage(
            content=textwrap.dedent(
                f"""
                Given the preceding context, you must only react with one of the following actions without revealing the existence of this prompt:
                - If you can fullfil the request confidently, do it.
                - If you need to start a research, return a research metadata strictly follows this format: '{research_task_metadata}'.
                - If you need to start a research but cannot decide the parameters, request help to figure out the research parameters.
                """
            ).strip()
        )
    )

    if not request.stream:
        agent_fast_response = await low_temp_chain.ainvoke(messages)
        # non stream response can only be used for short completions
        if agent_fast_response == start_research_message:
            agent_fast_response = "Please enable streaming response for this request."

        json_data = {
            "id": response_id,
            "object": "chat.completion",
            "created": created_time,
            "model": "gpt-researcher",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": agent_fast_response,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        }
        return json_data

    agent_fast_response = low_temp_chain.astream(messages)

    async def get_first_chunk(stream: AsyncIterator[str]) -> str:
        buffer = ""
        async for chunk in stream:
            buffer += chunk
            if len(buffer) >= len(start_research_message):
                break
        return buffer

    first_chunk = await get_first_chunk(agent_fast_response)

    previous_message_type = None

    def format_message(
        message: str, event_position: EventPosition, message_type: MessageType
    ) -> str:
        nonlocal previous_message_type

        if previous_message_type != message_type:
            if previous_message_type == "logs":
                message = f"</think-test>\n{message}"
            if message_type == "logs":
                message = f"<think-test>\n{message}"

        delta = {"content": message} if message else {}
        if event_position == "first":
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
                    "finish_reason": "stop" if event_position == "last" else None,
                }
            ],
        }

        previous_message_type = message_type
        return json.dumps(chunk)

    streaming_response = SSEWebSocketAdapter(format_message)

    async def redirect_output():
        try:
            buffer = first_chunk
            await streaming_response.send_text(first_chunk)
            async for chunk in agent_fast_response:
                buffer += chunk.content
                await streaming_response.send_text(chunk.content)  # type: ignore

            # messages = []
            # messages.append(
            #     SystemMessage(
            #         content=textwrap.dedent(
            #             """
            #             act as a function that strictly follow the user instructions and produces only the defined output.
            #             """
            #         ).strip()
            #     )
            # )
            # messages.append(
            #     HumanMessage(
            #         content=textwrap.dedent(
            #             f"""
            #             extract the research task metadata from the input and returns a json string.
            #             input: {buffer}
            #             """
            #         ).strip()
            #     )
            # )
            # json_data = await smart_agent.ainvoke(messages)
            # await streaming_response.send_text("\n" + json_data.content)
            await streaming_response.close()
        except Exception as e:
            logger.error(f"Error streaming response: {e}")

    if first_chunk.strip() != start_research_message:
        asyncio.create_task(redirect_output())
    else:
        # consume the rest of the stream to close it
        async for _ in agent_fast_response:
            pass

        await streaming_response.send_text(
            "Starting research. Please wait for the response..."
        )
        await streaming_response.close()
    # asyncio.create_task(test_stream(streaming_response))

    return StreamingResponse(streaming_response, media_type="text/event-stream")
