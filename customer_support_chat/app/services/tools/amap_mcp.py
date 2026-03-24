"""
高德地图 MCP 服务集成 (Amap Maps MCP Service Integration)

通过 DashScope 接入高德地图 MCP 服务，为行程推荐助手提供真实地图数据，
减少地名/地点/路线推荐中的幻觉问题。

Available tools (由 MCP 服务器动态提供):
  maps_text_search          - 关键词搜索 POI（景点/餐厅/酒店等）
  maps_around_search        - 周边 POI 搜索
  maps_search_detail        - POI 详情查询
  maps_geo                  - 地理编码（地址 → 坐标）
  maps_regeo                - 逆地理编码（坐标 → 地址）
  maps_direction_walking    - 步行路线规划
  maps_direction_driving    - 驾车路线规划
  maps_direction_transit_integrated - 公交/地铁路线规划
  maps_distance             - 两点距离计算
  maps_weather              - 实时天气查询
  maps_district             - 行政区域查询
  maps_bicycling            - 骑行路线规划
  maps_navi_driving         - 导航路线规划
"""

import asyncio
from contextlib import AsyncExitStack
from typing import List

from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.core.settings import get_settings

settings = get_settings()

_AMAP_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"

# Module-level state
_exit_stack: AsyncExitStack | None = None
_amap_tools: List = []


def _build_mcp_config() -> dict:
    return {
        "amap-maps": {
            "transport": "streamable_http",
            "url": _AMAP_MCP_URL,
            "headers": {
                "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}"
            },
        }
    }


async def startup() -> List:
    """
    初始化高德地图 MCP 客户端并加载工具列表。
    在应用启动时调用一次，工具列表将被缓存供整个应用使用。

    Returns:
        加载到的 LangChain 工具列表；若初始化失败则返回空列表。
    """
    global _exit_stack, _amap_tools

    if not settings.DASHSCOPE_API_KEY:
        logger.warning(
            "⚠️  DASHSCOPE_API_KEY 未配置，高德地图 MCP 工具不可用。"
            " 请在 .env 中设置 DASHSCOPE_API_KEY。"
        )
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        _exit_stack = AsyncExitStack()
        client = MultiServerMCPClient(_build_mcp_config())
        await _exit_stack.enter_async_context(client)
        _amap_tools = client.get_tools()
        logger.info(
            f"✅ 高德地图 MCP 工具加载成功，共 {len(_amap_tools)} 个工具: "
            f"{[t.name for t in _amap_tools]}"
        )
        return _amap_tools

    except ImportError:
        logger.error(
            "❌ 未找到 langchain-mcp-adapters 包。"
            " 请运行: poetry add langchain-mcp-adapters"
        )
        return []
    except Exception as exc:
        logger.error(f"❌ 高德地图 MCP 客户端初始化失败: {exc}")
        return []


async def shutdown() -> None:
    """
    关闭高德地图 MCP 客户端连接。
    在应用关闭时调用。
    """
    global _exit_stack, _amap_tools

    if _exit_stack is not None:
        try:
            await _exit_stack.aclose()
            logger.info("✅ 高德地图 MCP 客户端已关闭。")
        except Exception as exc:
            logger.warning(f"关闭高德地图 MCP 客户端时出错: {exc}")
        finally:
            _exit_stack = None
            _amap_tools = []


def get_amap_tools() -> List:
    """返回已加载的高德地图 MCP 工具列表（缓存副本）。"""
    return list(_amap_tools)


def is_initialized() -> bool:
    """返回 MCP 客户端是否已成功初始化。"""
    return len(_amap_tools) > 0
