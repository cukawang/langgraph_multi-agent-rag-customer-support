from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from customer_support_chat.app.core.humanloop_manager import humanloop_adapter
from customer_support_chat.app.core.state import State
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.services.utils import (
    create_tool_node_with_fallback,
    flight_info_to_string,
    create_entry_node,
)
from customer_support_chat.app.services.tools.flights import fetch_user_flight_information
from customer_support_chat.app.services.guardrails.guardrail_agents import (
    jailbreak_guardrail_agent,
    jailbreak_guardrail_agent_instructions,
    relevance_guardrail_agent,
    relevance_guardrail_agent_instructions,
)
from customer_support_chat.app.services.assistants.assistant_base import (
    Assistant,
    CompleteOrEscalate,
    llm,
)
from customer_support_chat.app.services.assistants.primary_assistant import (
    primary_assistant,
    primary_assistant_tools,
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
)
from customer_support_chat.app.services.assistants.flight_booking_assistant import (
    flight_booking_assistant,
    update_flight_safe_tools,
    update_flight_sensitive_tools,
)
from customer_support_chat.app.services.assistants.car_rental_assistant import (
    car_rental_assistant,
    book_car_rental_safe_tools,
    book_car_rental_sensitive_tools,
)
from customer_support_chat.app.services.assistants.hotel_booking_assistant import (
    hotel_booking_assistant,
    book_hotel_safe_tools,
    book_hotel_sensitive_tools,
)

# ---------------------------------------------------------------------------
# Module-level graph reference.
# Call initialize(amap_tools) at app startup to rebuild with live map tools.
# ---------------------------------------------------------------------------
multi_agentic_graph = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared node functions
# ---------------------------------------------------------------------------

def _user_info(state: State, config: RunnableConfig):
    flight_info = fetch_user_flight_information.invoke(input={}, config=config)
    return {"user_info": flight_info_to_string(flight_info)}


def _guardrail_check(state: State, config: RunnableConfig):
    """Check user input for safety and relevance."""
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        logger.warning("No user message found for guardrail check. Allowing.")
        return {"messages": [HumanMessage(content="No user input to check. Please provide a query.")]}

    user_input = user_messages[-1].content
    logger.info(f"🛡️ Checking safety and relevance for: '{user_input}'")

    jailbreak_result = jailbreak_guardrail_agent.invoke(
        f"{jailbreak_guardrail_agent_instructions}\n\nUser Input: {user_input}"
    )
    if not jailbreak_result.is_safe:
        logger.warning(f"🚨 Jailbreak attempt detected: {jailbreak_result.reasoning}")
        return {"messages": [HumanMessage(content=f"I cannot assist with that request. Reason: {jailbreak_result.reasoning}")]}

    relevance_result = relevance_guardrail_agent.invoke(
        f"{relevance_guardrail_agent_instructions}\n\nUser Input: {user_input}"
    )
    if not relevance_result.is_relevant:
        logger.warning(f"⚠️ Irrelevant input detected: {relevance_result.reasoning}")

    logger.info("✅ Input passed safety and relevance checks.")
    return {"messages": []}


def _should_route_to_primary(state: State) -> bool:
    if state["messages"]:
        last = state["messages"][-1]
        if hasattr(last, "content") and isinstance(last.content, str):
            return "Task completed/escalated to main assistant" in last.content
    return False


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def _compile_graph(exc_assistant, exc_safe_tools: List, exc_sensitive_tools: List):
    """Build and compile the StateGraph."""
    builder = StateGraph(State)

    # ── shared ──────────────────────────────────────────────────────────────
    builder.add_node("fetch_user_info", _user_info)
    builder.add_node("guardrail_check", _guardrail_check)
    builder.add_edge(START, "fetch_user_info")
    builder.add_edge("fetch_user_info", "guardrail_check")
    builder.add_edge("guardrail_check", "primary_assistant")

    # ── Flight Booking ───────────────────────────────────────────────────────
    builder.add_node("enter_update_flight", create_entry_node("Flight Updates & Booking Assistant", "update_flight"))
    builder.add_node("update_flight", flight_booking_assistant)
    builder.add_edge("enter_update_flight", "update_flight")
    builder.add_node("update_flight_safe_tools", create_tool_node_with_fallback(update_flight_safe_tools))
    builder.add_node("update_flight_sensitive_tools", create_tool_node_with_fallback(update_flight_sensitive_tools))

    def route_update_flight(state: State) -> Literal[
        "update_flight_safe_tools", "update_flight_sensitive_tools", "primary_assistant", "__end__"
    ]:
        route = tools_condition(state)
        if route == END:
            return END
        safe_names = [t.name for t in update_flight_safe_tools]
        if all(tc["name"] in safe_names for tc in state["messages"][-1].tool_calls):
            return "update_flight_safe_tools"
        return "update_flight_sensitive_tools"

    def route_update_flight_tools(state: State) -> Literal["update_flight", "primary_assistant"]:
        return "primary_assistant" if _should_route_to_primary(state) else "update_flight"

    builder.add_conditional_edges("update_flight_safe_tools", route_update_flight_tools)
    builder.add_conditional_edges("update_flight_sensitive_tools", route_update_flight_tools)
    builder.add_conditional_edges("update_flight", route_update_flight)

    # ── Car Rental ───────────────────────────────────────────────────────────
    builder.add_node("enter_book_car_rental", create_entry_node("Car Rental Assistant", "book_car_rental"))
    builder.add_node("book_car_rental", car_rental_assistant)
    builder.add_edge("enter_book_car_rental", "book_car_rental")
    builder.add_node("book_car_rental_safe_tools", create_tool_node_with_fallback(book_car_rental_safe_tools))
    builder.add_node("book_car_rental_sensitive_tools", create_tool_node_with_fallback(book_car_rental_sensitive_tools))

    def route_book_car_rental(state: State) -> Literal[
        "book_car_rental_safe_tools", "book_car_rental_sensitive_tools", "primary_assistant", "__end__"
    ]:
        route = tools_condition(state)
        if route == END:
            return END
        safe_names = [t.name for t in book_car_rental_safe_tools]
        if all(tc["name"] in safe_names for tc in state["messages"][-1].tool_calls):
            return "book_car_rental_safe_tools"
        return "book_car_rental_sensitive_tools"

    def route_car_rental_tools(state: State) -> Literal["book_car_rental", "primary_assistant"]:
        return "primary_assistant" if _should_route_to_primary(state) else "book_car_rental"

    builder.add_conditional_edges("book_car_rental_safe_tools", route_car_rental_tools)
    builder.add_conditional_edges("book_car_rental_sensitive_tools", route_car_rental_tools)
    builder.add_conditional_edges("book_car_rental", route_book_car_rental)

    # ── Hotel Booking ────────────────────────────────────────────────────────
    builder.add_node("enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel"))
    builder.add_node("book_hotel", hotel_booking_assistant)
    builder.add_edge("enter_book_hotel", "book_hotel")
    builder.add_node("book_hotel_safe_tools", create_tool_node_with_fallback(book_hotel_safe_tools))
    builder.add_node("book_hotel_sensitive_tools", create_tool_node_with_fallback(book_hotel_sensitive_tools))

    def route_book_hotel(state: State) -> Literal[
        "book_hotel_safe_tools", "book_hotel_sensitive_tools", "primary_assistant", "__end__"
    ]:
        route = tools_condition(state)
        if route == END:
            return END
        safe_names = [t.name for t in book_hotel_safe_tools]
        if all(tc["name"] in safe_names for tc in state["messages"][-1].tool_calls):
            return "book_hotel_safe_tools"
        return "book_hotel_sensitive_tools"

    def route_hotel_tools(state: State) -> Literal["book_hotel", "primary_assistant"]:
        return "primary_assistant" if _should_route_to_primary(state) else "book_hotel"

    builder.add_conditional_edges("book_hotel_safe_tools", route_hotel_tools)
    builder.add_conditional_edges("book_hotel_sensitive_tools", route_hotel_tools)
    builder.add_conditional_edges("book_hotel", route_book_hotel)

    # ── Excursion / Trip Recommendation (+ Amap MCP tools) ──────────────────
    builder.add_node("enter_book_excursion", create_entry_node("Trip Recommendation Assistant", "book_excursion"))
    builder.add_node("book_excursion", exc_assistant)
    builder.add_edge("enter_book_excursion", "book_excursion")
    builder.add_node("book_excursion_safe_tools", create_tool_node_with_fallback(exc_safe_tools))
    builder.add_node("book_excursion_sensitive_tools", create_tool_node_with_fallback(exc_sensitive_tools))

    def route_book_excursion(state: State) -> Literal[
        "book_excursion_safe_tools", "book_excursion_sensitive_tools", "primary_assistant", "__end__"
    ]:
        route = tools_condition(state)
        if route == END:
            return END
        safe_names = [t.name for t in exc_safe_tools]
        if all(tc["name"] in safe_names for tc in state["messages"][-1].tool_calls):
            return "book_excursion_safe_tools"
        return "book_excursion_sensitive_tools"

    def route_excursion_tools(state: State) -> Literal["book_excursion", "primary_assistant"]:
        return "primary_assistant" if _should_route_to_primary(state) else "book_excursion"

    builder.add_conditional_edges("book_excursion_safe_tools", route_excursion_tools)
    builder.add_conditional_edges("book_excursion_sensitive_tools", route_excursion_tools)
    builder.add_conditional_edges("book_excursion", route_book_excursion)

    # ── Primary Assistant ────────────────────────────────────────────────────
    builder.add_node("primary_assistant", primary_assistant)
    builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))

    def route_primary_assistant(state: State) -> Literal[
        "primary_assistant_tools",
        "enter_update_flight",
        "enter_book_car_rental",
        "enter_book_hotel",
        "enter_book_excursion",
        "__end__",
    ]:
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            name = tool_calls[0]["name"]
            if name == ToFlightBookingAssistant.__name__:
                return "enter_update_flight"
            elif name == ToBookCarRental.__name__:
                return "enter_book_car_rental"
            elif name == ToHotelBookingAssistant.__name__:
                return "enter_book_hotel"
            elif name == ToBookExcursion.__name__:
                return "enter_book_excursion"
            else:
                return "primary_assistant_tools"
        return "primary_assistant"

    builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant,
        {
            "enter_update_flight": "enter_update_flight",
            "enter_book_car_rental": "enter_book_car_rental",
            "enter_book_hotel": "enter_book_hotel",
            "enter_book_excursion": "enter_book_excursion",
            "primary_assistant_tools": "primary_assistant_tools",
            END: END,
        },
    )
    builder.add_edge("primary_assistant_tools", "primary_assistant")

    interrupt_nodes = [
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ]
    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=interrupt_nodes)


def initialize(amap_tools: List = None):
    """
    编译（或重新编译）多智能体图。

    Args:
        amap_tools: 来自 amap_mcp.startup() 的高德地图工具列表（可选）。
    """
    global multi_agentic_graph

    if amap_tools is None:
        amap_tools = []

    from customer_support_chat.app.services.assistants.excursion_assistant import (
        build_excursion_assistant,
    )
    exc_assistant, exc_safe_tools, exc_sensitive_tools = build_excursion_assistant(amap_tools)
    multi_agentic_graph = _compile_graph(exc_assistant, exc_safe_tools, exc_sensitive_tools)

    map_status = f"✅ {len(amap_tools)} 个高德地图工具" if amap_tools else "⚠️  无高德地图工具（基础模式）"
    logger.info(f"✅ 多智能体图已编译。{map_status}")
    return multi_agentic_graph


# Default compile at import time (no Amap tools).
initialize()
