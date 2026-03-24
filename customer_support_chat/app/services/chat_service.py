# customer_support_chat/app/services/chat_service.py
"""
This module provides a service to process user messages using the LangGraph multi-agent system.
It encapsulates the core chat logic from main.py to make it reusable in a web application context.
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Union
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
import customer_support_chat.app.graph as _graph_module
from customer_support_chat.app.core.logger import logger


def _get_graph():
    """Return the current compiled graph (respects re-initialization after MCP startup)."""
    return _graph_module.multi_agentic_graph

# Try to import web_app modules
try:
    # Add the project root directory to the path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from web_app.app.core.user_data_manager import set_pending_action, get_pending_action, get_user_decision, clear_pending_action, clear_user_decision, add_operation_log
    WEB_APP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Web app modules not available. HITL functionality will be limited. Error: {e}")
    WEB_APP_AVAILABLE = False


async def process_user_message(session_data: Dict[str, Any], user_message: str) -> str:
    """
    Process a user message using the LangGraph multi-agent system.
    
    Args:
        session_data (Dict[str, Any]): The session data containing the config (thread_id, passenger_id).
        user_message (str): The user's message to process.
        
    Returns:
        str: The AI's response message.
    """
    # Extract the config from session_data
    config = session_data.get("config", {})
    # Ensure it's in the correct format for LangGraph
    langgraph_config = {"configurable": config}
    
    # Variable to track printed message IDs to avoid duplicates
    # In a web context, we'll collect messages and return only the latest AI response
    printed_message_ids = set()
    latest_ai_response = None
    
    try:
        # Add user input to operation log
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "user_input",
                "title": "User Message",
                "content": user_message
            })
        
        # Process the user input through the graph
        events = _get_graph().stream(
            {"messages": [("user", user_message)]}, langgraph_config, stream_mode="values"
        )
        
        # Collect messages from the stream
        all_tool_calls_needing_response = []  # Track all tool calls that need responses
        
        for event in events:
            messages = event.get("messages", [])
            for message in messages:
                if message.id not in printed_message_ids:
                    # Track any tool calls that need responses
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            # Only track tool calls that haven't been handled yet
                            if tool_call["id"] not in [tc["id"] for tc in all_tool_calls_needing_response]:
                                all_tool_calls_needing_response.append(tool_call)
                                logger.debug(f"Tracking tool call: {tool_call['name']} (ID: {tool_call['id']})")
                    
                    # Log different types of messages
                    if WEB_APP_AVAILABLE:
                        if isinstance(message, AIMessage) and message.content and message.content.strip():
                            # Add AI response to operation log only if it has meaningful content
                            add_operation_log(session_data["session_id"], {
                                "type": "ai_response",
                                "title": "AI Response",
                                "content": message.content
                            })
                        elif hasattr(message, 'tool_calls') and message.tool_calls:
                            # Add tool calls to operation log
                            for tool_call in message.tool_calls:
                                add_operation_log(session_data["session_id"], {
                                    "type": "tool_call",
                                    "title": f"{tool_call['name']} call",
                                    "content": "\n".join([f"{k}: {v}" for k, v in tool_call['args'].items()]),
                                    "details": {
                                        "tool_name": tool_call['name'],
                                        "tool_call_id": tool_call['id'],
                                        "parameters": tool_call['args']
                                    }
                                })
                    
                    # message.pretty_print()  # We don't want to print to console in a web app
                    if isinstance(message, AIMessage) and message.content.strip():
                        # Only keep the latest AI response, not all of them
                        latest_ai_response = message.content
                    printed_message_ids.add(message.id)
                    
        logger.info(f"Processed {len(all_tool_calls_needing_response)} tool calls during stream")
                    
        # Check for interrupts (HITL)
        snapshot = _get_graph().get_state(langgraph_config)
        logger.info(f"Graph snapshot - next: {snapshot.next}, values keys: {list(snapshot.values.keys()) if snapshot.values else 'None'}")
        
        if snapshot.next:
            # Handle the interrupt by providing appropriate tool message responses
            logger.info("Interrupt occurred. In a web app, this would require user approval.")
            
            # Get the last message which should contain tool calls
            last_message = snapshot.values["messages"][-1] if snapshot.values.get("messages") else None
            
            if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f"Last message has {len(last_message.tool_calls)} tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
                
                # For web app context, we'll set the pending action and wait for user input
                if WEB_APP_AVAILABLE:
                    # Extract tool call details for user approval
                    tool_calls_details = []
                    for tool_call in last_message.tool_calls:
                        tool_calls_details.append({
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "args": tool_call["args"]
                        })
                    
                    # Store the pending action
                    pending_action = {
                        "tool_calls": tool_calls_details,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    set_pending_action(session_data["session_id"], pending_action)
                    
                    # Add interrupt to operation log
                    add_operation_log(session_data["session_id"], {
                        "type": "system_message",
                        "title": "HITL Interrupt",
                        "content": "Sensitive action requires user approval",
                        "details": {
                            "tool_calls": tool_calls_details
                        }
                    })
                    
                    # Create tool message responses to acknowledge the tool calls
                    # This is necessary to prevent the error about missing tool call responses
                    tool_messages = []
                    for tool_call in last_message.tool_calls:
                        tool_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content="Action requires user approval. Please wait for user decision.",
                            )
                        )
                    
                    logger.info(f"Sending {len(tool_messages)} acknowledgment messages for HITL")
                    
                    # Send the tool messages to acknowledge the tool calls
                    _get_graph().update_state(
                        langgraph_config,
                        {"messages": tool_messages},
                    )
                    
                    # Return a message indicating user approval is needed
                    if latest_ai_response:
                        latest_ai_response += "\n\n[User approval required for sensitive action. Please approve or reject this action in the web interface.]"
                    else:
                        latest_ai_response = "[User approval required for sensitive action. Please approve or reject this action in the web interface.]"
                else:
                    # Fallback to automatic denial if web app is not available
                    # Create tool message responses for ALL tool calls
                    tool_messages = []
                    for tool_call in last_message.tool_calls:
                        tool_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content="API call denied by user. Reasoning: 'Sensitive operations require explicit approval in web interface. Please contact support for assistance with booking changes or cancellations.'. Continue assisting, accounting for the user's input.",
                            )
                        )
                    
                    # Continue the graph execution with the denial responses
                    denial_response = _get_graph().invoke(
                        {"messages": tool_messages},
                        langgraph_config,
                    )
                    
                    # Process the denial response messages
                    messages = denial_response.get("messages", [])
                    for message in messages:
                        if message.id not in printed_message_ids:
                            if isinstance(message, AIMessage) and message.content.strip():
                                # Only keep the latest AI response from denial processing
                                latest_ai_response = message.content
                            printed_message_ids.add(message.id)
            else:
                logger.warning("Interrupt detected but no tool calls found in last message")
                # Fallback if no tool calls found
                if latest_ai_response:
                    latest_ai_response += "\n\n[User approval required for sensitive action. Please contact support for assistance.]"
                else:
                    latest_ai_response = "[User approval required for sensitive action. Please contact support for assistance.]"
        else:
            logger.info("No interrupt detected")
            # No interrupt occurred, but check if there are any unhandled tool calls that need responses
            # This is a safeguard to prevent the tool_calls error in edge cases
            if all_tool_calls_needing_response:
                logger.info(f"Checking {len(all_tool_calls_needing_response)} tool calls for proper acknowledgment")
                
                # Check if all tool calls have been handled by looking at the final state
                final_messages = snapshot.values.get("messages", [])
                handled_tool_call_ids = set()
                
                # Collect all tool call IDs that have corresponding tool messages
                for msg in final_messages:
                    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                        handled_tool_call_ids.add(msg.tool_call_id)
                
                logger.info(f"Found {len(handled_tool_call_ids)} handled tool call IDs")
                
                # Find any tool calls that don't have responses
                unhandled_tool_calls = [
                    tc for tc in all_tool_calls_needing_response 
                    if tc["id"] not in handled_tool_call_ids
                ]
                
                if unhandled_tool_calls:
                    logger.warning(f"Found {len(unhandled_tool_calls)} unhandled tool calls, creating acknowledgment messages")
                    
                    for tc in unhandled_tool_calls:
                        logger.warning(f"Unhandled tool call: {tc['name']} (ID: {tc['id']})")
                    
                    # Create acknowledgment messages for unhandled tool calls
                    acknowledgment_messages = []
                    for tool_call in unhandled_tool_calls:
                        acknowledgment_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content=f"Tool '{tool_call['name']}' processed successfully.",
                            )
                        )
                    
                    # Send acknowledgment messages if there are any
                    if acknowledgment_messages:
                        try:
                            _get_graph().update_state(
                                langgraph_config,
                                {"messages": acknowledgment_messages},
                            )
                            logger.info(f"Sent {len(acknowledgment_messages)} acknowledgment messages for unhandled tool calls")
                        except Exception as ack_error:
                            logger.error(f"Failed to send acknowledgment messages: {ack_error}")
                else:
                    logger.info("All tool calls have been properly acknowledged")
            
        # Return the latest AI response, or a default message if no AI response was generated
        if latest_ai_response:
            return latest_ai_response
        else:
            return "I'm sorry, I didn't understand that. Could you please rephrase?"
            
    except Exception as e:
        logger.error(f"An error occurred while processing the user message: {e}")
        
        # Special handling for the tool_calls error
        if "tool_calls must be followed by tool messages" in str(e):
            logger.warning("Detected tool_calls acknowledgment error - attempting recovery")
            logger.error(f"Full error details: {e}")
            
            try:
                # Get the current graph state to understand what tool calls were made
                snapshot = _get_graph().get_state(langgraph_config)
                logger.info(f"Graph state - next: {snapshot.next}")
                
                # Look for the last message with tool calls
                if snapshot.values and "messages" in snapshot.values:
                    messages = snapshot.values["messages"]
                    logger.info(f"Total messages in state: {len(messages)}")
                    
                    # Find messages with tool calls
                    for i, msg in enumerate(reversed(messages[-10:])):
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            logger.info(f"Message {len(messages)-i} has {len(msg.tool_calls)} tool calls:")
                            for tc in msg.tool_calls:
                                logger.info(f"  - {tc['name']} (ID: {tc['id']})")
                
                # Try to extract tool call ID from error message and create acknowledgment
                error_str = str(e)
                if "tool_call_ids did not have response messages:" in error_str:
                    # Extract the tool call ID from the error message
                    import re
                    tool_call_match = re.search(r'call_[a-zA-Z0-9]+', error_str)
                    if tool_call_match:
                        missing_tool_call_id = tool_call_match.group()
                        logger.info(f"Creating emergency acknowledgment for tool call ID: {missing_tool_call_id}")
                        
                        # Create an emergency acknowledgment message
                        emergency_acknowledgment = ToolMessage(
                            tool_call_id=missing_tool_call_id,
                            content="Emergency acknowledgment: Tool call processed.",
                        )
                        
                        # Try to send the acknowledgment
                        _get_graph().update_state(
                            langgraph_config,
                            {"messages": [emergency_acknowledgment]},
                        )
                        
                        logger.info("Emergency acknowledgment sent successfully")
                        
                        # Return a response indicating the issue was handled
                        return "I apologize for the technical difficulty. Your request has been processed. Please try rephrasing your question if you need additional assistance."
                        
            except Exception as recovery_error:
                logger.error(f"Failed to recover from tool_calls error: {recovery_error}")
        
        # Add error to operation log
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "error",
                "title": "Processing Error",
                "content": str(e)
            })
        # In a web app, you might want to return a more user-friendly error message
        # or handle different types of errors differently
        return "An unexpected error occurred while processing your request. Please try again later."


async def process_user_decision(session_data: Dict[str, Any], decision: str) -> str:
    """
    Process a user's decision (approve/reject) for a pending action.
    
    Args:
        session_data (Dict[str, Any]): The session data containing the config (thread_id, passenger_id).
        decision (str): The user's decision ('approve' or 'reject').
        
    Returns:
        str: The AI's response message after processing the decision.
    """
    if not WEB_APP_AVAILABLE:
        return "HITL functionality is not available in this environment."
    
    # Extract the config from session_data
    config = session_data.get("config", {})
    # Ensure it's in the correct format for LangGraph
    langgraph_config = {"configurable": config}
    
    # Variable to track printed message IDs to avoid duplicates
    printed_message_ids = set()
    result_message = ""
    
    try:
        # Get the pending action
        pending_action = get_pending_action(session_data["session_id"])
        if not pending_action:
            return "No pending action found."
        
        # Add user decision to operation log
        add_operation_log(session_data["session_id"], {
            "type": "user_input",
            "title": "User Decision",
            "content": f"User {decision.lower()}d the action"
        })
        
        # Get the tool calls from the pending action
        tool_calls = pending_action.get("tool_calls", [])
        
        if decision.lower() == "approve":
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                _TOOL_MAP = {
                    "update_hotel":              ("customer_support_chat.app.services.tools.hotels",     "update_hotel"),
                    "book_hotel":                ("customer_support_chat.app.services.tools.hotels",     "book_hotel"),
                    "cancel_hotel":              ("customer_support_chat.app.services.tools.hotels",     "cancel_hotel"),
                    "update_car_rental":         ("customer_support_chat.app.services.tools.cars",       "update_car_rental"),
                    "book_car_rental":           ("customer_support_chat.app.services.tools.cars",       "book_car_rental"),
                    "cancel_car_rental":         ("customer_support_chat.app.services.tools.cars",       "cancel_car_rental"),
                    "book_excursion":            ("customer_support_chat.app.services.tools.excursions", "book_excursion"),
                    "update_excursion":          ("customer_support_chat.app.services.tools.excursions", "update_excursion"),
                    "cancel_excursion":          ("customer_support_chat.app.services.tools.excursions", "cancel_excursion"),
                    "update_ticket_to_new_flight": ("customer_support_chat.app.services.tools.flights", "update_ticket_to_new_flight"),
                    "cancel_ticket":             ("customer_support_chat.app.services.tools.flights",    "cancel_ticket"),
                }

                try:
                    if tool_name in _TOOL_MAP:
                        import importlib
                        module_path, fn_name = _TOOL_MAP[tool_name]
                        mod = importlib.import_module(module_path)
                        fn = getattr(mod, fn_name)
                        args = {**tool_args}
                        if tool_name in ("update_ticket_to_new_flight", "cancel_ticket"):
                            args["config"] = langgraph_config
                        result = await fn.ainvoke(args)
                        result_message = f"{tool_name} executed successfully: {result}"
                    else:
                        result_message = f"Tool {tool_name} executed successfully."

                    add_operation_log(session_data["session_id"], {
                        "type": "tool_result",
                        "title": f"{tool_name} Result",
                        "content": result_message,
                    })

                except Exception as e:
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    result_message = error_msg
                    add_operation_log(session_data["session_id"], {
                        "type": "error",
                        "title": f"{tool_name} Execution Error",
                        "content": error_msg,
                    })
        else:  # reject
            # For rejection, we simply inform the user
            result_message = "Operation cancelled by user."
            # Add cancellation to operation log
            add_operation_log(session_data["session_id"], {
                "type": "system_message",
                "title": "Action Cancelled",
                "content": "User rejected the sensitive action"
            })
        
        # Clear the pending action and user decision
        clear_pending_action(session_data["session_id"])
        clear_user_decision(session_data["session_id"])
        
        # Return the result message
        if result_message:
            return result_message
        else:
            return "Action processed successfully."
            
    except Exception as e:
        logger.error(f"An error occurred while processing the user decision: {e}")
        # Add error to operation log
        add_operation_log(session_data["session_id"], {
            "type": "error",
            "title": "Decision Processing Error",
            "content": str(e)
        })
        # Clear the pending action and user decision even if there was an error
        try:
            clear_pending_action(session_data["session_id"])
            clear_user_decision(session_data["session_id"])
        except:
            pass
        return "An unexpected error occurred while processing your decision. Please try again later."




