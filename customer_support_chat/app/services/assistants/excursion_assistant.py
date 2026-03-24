from datetime import datetime
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from customer_support_chat.app.services.tools import (
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
)
from customer_support_chat.app.services.assistants.assistant_base import (
    Assistant,
    CompleteOrEscalate,
    llm,
)

# Base prompt — used when no map tools are available
_BASE_SYSTEM = (
    "You are a specialized assistant for handling trip recommendations. "
    "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
    "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
    "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
    "When searching, be persistent. Expand your query bounds if the first search returns no results. "
    "Remember that a booking isn't completed until after the relevant tool has successfully been used."
    "\nCurrent time: {time}."
    '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
    "\n\nSome examples for which you should CompleteOrEscalate:\n"
    " - 'nevermind I think I'll book separately'\n"
    " - 'I need to figure out transportation while I'm there'\n"
    " - 'Oh wait I haven't booked my flight yet I'll do that first'\n"
    " - 'Excursion booking confirmed!'"
)

# Enhanced prompt — used when Amap MCP map tools are available
_MAP_ENHANCED_SYSTEM = (
    "You are a specialized assistant for handling trip recommendations, "
    "equipped with real-time map data from Amap (高德地图). "
    "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
    "\n\n"
    "You have access to live map tools that let you:\n"
    "  • Search for real POIs (attractions, restaurants, hotels) by keyword or location\n"
    "  • Look up weather at any destination before recommending it\n"
    "  • Calculate driving/transit/walking routes and distances between places\n"
    "  • Verify that a place name or address actually exists\n"
    "\n"
    "IMPORTANT RULES to avoid hallucinations:\n"
    "  1. Always use map search tools to verify attraction/destination names before presenting them.\n"
    "  2. When suggesting nearby activities, use maps_around_search with a real POI as the center.\n"
    "  3. When the user asks about weather or travel time, call the relevant map tool — do NOT guess.\n"
    "  4. Only present attractions that appear in the map search results.\n"
    "\n"
    "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
    "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
    "When searching, be persistent. Expand your query bounds if the first search returns no results. "
    "Remember that a booking isn't completed until after the relevant tool has successfully been used."
    "\nCurrent time: {time}."
    '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.'
    "\n\nSome examples for which you should CompleteOrEscalate:\n"
    " - 'nevermind I think I'll book separately'\n"
    " - 'I need to figure out transportation while I'm there'\n"
    " - 'Oh wait I haven't booked my flight yet I'll do that first'\n"
    " - 'Excursion booking confirmed!'"
)


def build_excursion_assistant(
    amap_tools: List = None,
) -> Tuple[Assistant, List, List]:
    """
    构建行程推荐助手，可选注入高德地图 MCP 工具。

    Args:
        amap_tools: 高德地图 MCP 工具列表（由 amap_mcp.startup() 加载）。
                    传入空列表或 None 时使用无地图的基础模式。

    Returns:
        (assistant, safe_tools, sensitive_tools) 三元组，供 graph.py 构建节点使用。
    """
    if amap_tools is None:
        amap_tools = []

    system_msg = _MAP_ENHANCED_SYSTEM if amap_tools else _BASE_SYSTEM

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now())

    safe_tools = [search_trip_recommendations, CompleteOrEscalate] + list(amap_tools)
    sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
    all_tools = safe_tools + sensitive_tools

    runnable = prompt | llm.bind_tools(all_tools)
    assistant = Assistant(runnable)

    return assistant, safe_tools, sensitive_tools


# ---------------------------------------------------------------------------
# Default module-level instances (no Amap tools) — kept for backward compat.
# graph.py calls build_excursion_assistant(amap_tools) inside initialize().
# ---------------------------------------------------------------------------
excursion_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _BASE_SYSTEM),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

book_excursion_safe_tools = [search_trip_recommendations, CompleteOrEscalate]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools

book_excursion_runnable = excursion_prompt | llm.bind_tools(book_excursion_tools)
excursion_assistant = Assistant(book_excursion_runnable)
