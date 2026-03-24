from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    search_flights,
    lookup_policy,
)
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm
from pydantic import BaseModel, Field


class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""
    request: str = Field(description="Any necessary follow-up questions the update flight assistant should clarify before proceeding.")


class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""
    location: str = Field(description="The location where the user wants to rent a car.")
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(description="Any additional information or requests from the user regarding the car rental.")


class ToHotelBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle hotel bookings, modifications, and cancellations."""
    location: str = Field(description="The location where the user wants to book a hotel. Use 'Unknown' if not specified for cancellation requests.", default="Unknown")
    checkin_date: str = Field(description="The check-in date for the hotel. Use 'Unknown' if not specified for cancellation requests.", default="Unknown")
    checkout_date: str = Field(description="The check-out date for the hotel. Use 'Unknown' if not specified for cancellation requests.", default="Unknown")
    request: str = Field(description="Any additional information or requests from the user regarding the hotel operation (booking, cancellation, modification).")


class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""
    location: str = Field(description="The location where the user wants to book a recommended trip.")
    request: str = Field(description="Any additional information or requests from the user regarding the trip recommendation.")


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "When customers need help with specialized services, you must delegate to the appropriate assistant: "
            "\n\nDELEGATION RULES (ALWAYS delegate, never try to handle these yourself):"
            "\n- Flight updates/cancellations → ToFlightBookingAssistant"
            "\n- Car rental booking/modification/cancellation → ToBookCarRental"
            "\n- Hotel booking/modification/cancellation/status → ToHotelBookingAssistant"
            "\n- Trip recommendations/excursions → ToBookExcursion"
            "\n\nFor hotel operations, delegate even for:"
            "\n- 'cancel my hotel', 'cancel it' (when referring to hotel)"
            "\n- 'check hotel status', 'hotel booking status'"
            "\n- 'modify my hotel booking', 'change hotel dates'"
            "\n\nIMPORTANT: If the user asks about MULTIPLE services in one query (e.g., 'car and hotel status'), "
            "do NOT delegate to multiple assistants. Instead, handle it yourself by:"
            "\n1. Using search_flights to check current bookings"
            "\n2. Providing a summary of what you can see"
            "\n3. Asking the user to specify which service they want detailed help with"
            "\n\nOnly delegate to ONE assistant at a time. Never make multiple delegation calls in a single response."
            "\n\nOnly the specialized assistants have permission to make these changes. "
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

primary_assistant_tools = [
    DuckDuckGoSearchResults(max_results=10),
    search_flights,
    lookup_policy,
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
]

primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools)
primary_assistant = Assistant(primary_assistant_runnable)
