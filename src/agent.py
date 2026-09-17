"""LangGraph State Graph w/guardrails & strict step limits."""

from __future__ import annotations

from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import LLM_MODEL, MAX_AGENT_ITERATIONS, get_gemini_api_key
from src.prompts import SYSTEM_PROMPT
from src.tools import TOOLS


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration_count: int


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=get_gemini_api_key(),
        temperature=0,
        timeout=45,
        max_retries=1,
    )


def agent_node(state: AgentState) -> dict:
    """Call Gemini w/ tools, enforcing max retrieval/tool step budget."""
    iteration_count = int(state.get("iteration_count") or 0)

    if iteration_count >= MAX_AGENT_ITERATIONS:
        return {
            "messages": [
                AIMessage(content="Analysis halted: Exceeded maximum retrieval steps.")
            ],
            "iteration_count": iteration_count,
        }

    llm = _build_llm().bind_tools(TOOLS)
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    response = llm.invoke(messages)

    # Gemini flash-lite sometimes returns empty content + no tool_calls after
    # tools already produced useful results (math OR document search).
    tool_calls = getattr(response, "tool_calls", None) or []
    content = response.content
    empty_content = (
        content is None
        or (isinstance(content, str) and not content.strip())
        or content == []
    )
    if empty_content and not tool_calls:
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "financial_calculator":
                result = msg.content if isinstance(msg.content, str) else str(msg.content)
                if result and not str(result).startswith("Error"):
                    response = AIMessage(content=str(result))
                    empty_content = False
                break

        if empty_content:
            has_tool_context = any(isinstance(m, ToolMessage) for m in messages)
            if has_tool_context:
                # Force a plain text answer from the same tool-augmented history.
                plain = _build_llm().invoke(
                    [
                        *messages,
                        AIMessage(
                            content=(
                                "Using only the tool results above, write the final "
                                "grounded answer with citations. Do not call tools."
                            )
                        ),
                    ]
                )
                plain_content = plain.content
                plain_empty = (
                    plain_content is None
                    or (isinstance(plain_content, str) and not plain_content.strip())
                    or plain_content == []
                )
                if not plain_empty:
                    response = AIMessage(content=plain_content)

    return {
        "messages": [response],
        "iteration_count": iteration_count + 1,
    }


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """Halt on step limit, otherwise use tools_condition."""
    messages = state["messages"]
    last = messages[-1] if messages else None
    if (
        isinstance(last, AIMessage)
        and isinstance(last.content, str)
        and "Exceeded maximum retrieval steps" in last.content
    ):
        return END

    decision = tools_condition(state)
    return decision


def build_graph():
    """Compile the LangGraph agent w/agent & tools nodes."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(question: str, recursion_limit: int | None = None) -> dict:
    """
    **Invoke agent on user question.
    Returns final state including full message trace.
    """
    app = build_graph()
    limit = recursion_limit if recursion_limit is not None else MAX_AGENT_ITERATIONS * 2 + 2
    return app.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "iteration_count": 0,
        },
        config={"recursion_limit": limit},
    )
