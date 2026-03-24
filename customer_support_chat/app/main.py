# main.py

import asyncio
import uuid
import os
from langchain_core.messages import ToolMessage
from customer_support_chat.app.services.utils import download_and_prepare_db
from customer_support_chat.app.core.logger import logger


async def async_main():
    # Ensure the database is downloaded and prepared
    download_and_prepare_db()

    # ------------------------------------------------------------------ #
    # 1. 初始化高德地图 MCP 客户端，加载实时地图工具
    # ------------------------------------------------------------------ #
    from customer_support_chat.app.services.tools.amap_mcp import startup, shutdown, get_amap_tools
    amap_tools = await startup()

    # ------------------------------------------------------------------ #
    # 2. 用 Amap 工具重新编译多智能体图
    # ------------------------------------------------------------------ #
    from customer_support_chat.app.graph import initialize
    graph = initialize(amap_tools)

    # ------------------------------------------------------------------ #
    # 3. 生成并保存图可视化（可选）
    # ------------------------------------------------------------------ #
    try:
        graph_obj = graph.get_graph(xray=True)
        graph_image = graph_obj.draw_mermaid_png()
        graphs_dir = "./graphs"
        os.makedirs(graphs_dir, exist_ok=True)
        image_path = os.path.join(graphs_dir, "multi-agent-rag-system-graph.png")
        with open(image_path, "wb") as f:
            f.write(graph_image)
        print(f"Graph saved at {image_path}")
    except Exception as e:
        logger.error(f"An error occurred while generating the graph visualization: {e}")
        print("Graph visualization could not be generated. Continuing without it.")

    # ------------------------------------------------------------------ #
    # 4. 主对话循环
    # ------------------------------------------------------------------ #
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "5102 899977",
            "thread_id": thread_id,
        }
    }
    printed_message_ids = set()

    try:
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            events = graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )

            for event in events:
                messages = event.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id)

            # Handle human-in-the-loop interrupts
            snapshot = graph.get_state(config)
            while snapshot.next:
                user_input = input(
                    "\nDo you approve of the above actions? Type 'y' to continue; "
                    "otherwise, explain your requested changes.\n\n"
                )
                if user_input.strip().lower() == "y":
                    result = graph.invoke(None, config)
                else:
                    tool_call_id = snapshot.value["messages"][-1].tool_calls[0]["id"]
                    result = graph.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=tool_call_id,
                                    content=(
                                        f"API call denied by user. Reasoning: '{user_input}'. "
                                        "Continue assisting, accounting for the user's input."
                                    ),
                                )
                            ]
                        },
                        config,
                    )
                messages = result.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id)
                snapshot = graph.get_state(config)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print("An unexpected error occurred. Please check the logs for more details.")
    finally:
        # ------------------------------------------------------------------ #
        # 5. 关闭 MCP 客户端
        # ------------------------------------------------------------------ #
        await shutdown()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
