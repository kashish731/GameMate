"""
app.py - GameMate LangGraph Workflow Backend
Demonstrates:
- LangGraph StateGraph with conditional edge routing
- Dual reducers (add_messages, operator.add)
- Message trimming before every LLM call (trim_messages)
- Local Ollama LLM integration (llama3.2:3b)
- Short-term conversational memory using InMemorySaver checkpointer (thread_id)
- Long-term profile and backlog memory using InMemoryStore (user_id)
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import TypedDict, Annotated
import operator
import re

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.messages import (
    trim_messages,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_ollama import ChatOllama

# -----------------------------------------------------------------------------
# 1. FastAPI Application Setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="GameMate API",
    description="Personal Gaming Companion powered by LangGraph, Checkpointing, and Store",
    version="0.4.0",
)

# -----------------------------------------------------------------------------
# 2. Local Ollama LLM Initialization
# -----------------------------------------------------------------------------
# Initialized once globally to avoid reloading the model on every request
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,
)

# -----------------------------------------------------------------------------
# 3. Memory Instances (Checkpointer + Store)
# -----------------------------------------------------------------------------
# Short-term memory checkpointer: tracks conversation turns per thread_id
checkpointer = InMemorySaver()

# Long-term memory store: persists profiles and backlogs per user_id across threads
store = InMemoryStore()

# -----------------------------------------------------------------------------
# 4. State Definition & Reducers
# -----------------------------------------------------------------------------
class GamingState(TypedDict):
    # REDUCER 1: add_messages automatically appends messages and manages IDs
    messages: Annotated[list, add_messages]

    # REDUCER 2: operator.add automatically concatenates activity log strings across nodes
    activity_log: Annotated[list[str], operator.add]

    # User & Session IDs
    user_id: str
    thread_id: str

    # Routing and Diagnostics
    route: str
    trim_stats: dict

    # Long-term memory fields
    profile: dict
    backlog: list


# -----------------------------------------------------------------------------
# 5. Extraction & Trimming Helpers
# -----------------------------------------------------------------------------
def trim_conversation(state: GamingState) -> tuple[list, dict]:
    """
    Trims the conversation history to a manageable window before invoking the LLM.
    - Preserves state['messages'] intact (does not mutate state).
    - Uses LangChain's trim_messages to retain the most recent messages.
    - Uses token_counter=len with max_tokens=4 to keep at most the last 4 messages,
      starting on a user message (HumanMessage).
    - Returns (trimmed_messages, trim_stats).
    """
    raw_messages = state.get("messages", [])
    before_count = len(raw_messages)

    trimmed = trim_messages(
        raw_messages,
        max_tokens=4,
        strategy="last",
        token_counter=len,
        start_on="human",
    )
    after_count = len(trimmed)

    trim_stats = {
        "before": before_count,
        "after": after_count,
    }
    return trimmed, trim_stats


def extract_profile_updates(text: str, current_profile: dict) -> tuple[dict, list[str]]:
    """
    Extracts gaming preferences (genres, platforms) from text using keyword matching.
    Returns (updated_profile, list_of_log_messages).
    """
    text_lower = text.lower()
    updated = {
        "favorite_genres": list(current_profile.get("favorite_genres", [])),
        "platforms": list(current_profile.get("platforms", [])),
        "preferences": list(current_profile.get("preferences", [])),
    }
    logs = []

    # Gaming genre keywords
    genre_map = {
        "rpg": "RPG",
        "action rpg": "Action RPG",
        "soulslike": "Soulslike",
        "fps": "FPS",
        "shooter": "Shooter",
        "strategy": "Strategy",
        "roguelike": "Roguelike",
        "adventure": "Adventure",
        "open world": "Open World",
        "indie": "Indie",
        "horror": "Horror",
        "simulation": "Simulation",
    }

    # Gaming platform keywords
    platform_map = {
        "pc": "PC",
        "ps5": "PS5",
        "playstation": "PlayStation",
        "xbox": "Xbox",
        "switch": "Nintendo Switch",
        "nintendo": "Nintendo Switch",
        "mobile": "Mobile",
    }

    for kw, genre_name in genre_map.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            if genre_name not in updated["favorite_genres"]:
                updated["favorite_genres"].append(genre_name)
                logs.append(f"Profile updated: favorite_genres += {genre_name}")

    for kw, plat_name in platform_map.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            if plat_name not in updated["platforms"]:
                updated["platforms"].append(plat_name)
                logs.append(f"Profile updated: platforms += {plat_name}")

    return updated, logs


def extract_backlog_game(text: str) -> str | None:
    """
    Extracts a game title from phrases like 'Add Elden Ring to my backlog'.
    """
    text = text.strip()
    match = re.search(r"add\s+(.*?)\s+(?:to|in|into)\s+(?:my\s+)?backlog", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(" .,!?\"'")
    
    match2 = re.search(r"(?:add\s+to\s+backlog|backlog\s+add)[:\s]+(.*?)$", text, re.IGNORECASE)
    if match2:
        return match2.group(1).strip(" .,!?\"'")
    
    return None


# -----------------------------------------------------------------------------
# 6. Node Functions
# -----------------------------------------------------------------------------
def load_profile(state: GamingState, *, store: BaseStore) -> dict:
    """
    Loads both long-term profile and gaming backlog from InMemoryStore using user_id.
    Also inspects the incoming user message for any new preference declarations.
    """
    user_id = state.get("user_id", "player-1")
    thread_id = state.get("thread_id", "thread-1")

    activity_entries = [
        f"Short-term memory loaded for thread: {thread_id}",
        f"Long-term profile loaded for user: {user_id}",
    ]

    # 1. Fetch long-term profile from InMemoryStore namespace ("gaming_profile", user_id)
    profile_item = store.get(("gaming_profile", user_id), "profile")
    if profile_item and profile_item.value:
        profile = dict(profile_item.value)
    else:
        profile = {"favorite_genres": [], "platforms": [], "preferences": []}

    # 2. Fetch long-term backlog from InMemoryStore namespace ("gaming_backlog", user_id)
    backlog_item = store.get(("gaming_backlog", user_id), "backlog")
    if backlog_item and backlog_item.value:
        backlog = list(backlog_item.value.get("games", []))
    else:
        backlog = []

    # 3. Check current message for new preference declarations
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        updated_profile, update_logs = extract_profile_updates(user_text, profile)
        if update_logs:
            profile = updated_profile
            # Save updated profile back to long-term store
            store.put(("gaming_profile", user_id), "profile", profile)
            activity_entries.extend(update_logs)

    return {
        "profile": profile,
        "backlog": backlog,
        "activity_log": activity_entries,
    }


def route_intent(state: GamingState) -> dict:
    """
    Examines the latest message to determine player intent using keyword matching.
    Sets the route in state and appends to the activity log.
    """
    messages = state.get("messages", [])
    user_text = ""
    if messages:
        last_message = messages[-1]
        user_text = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    text_lower = user_text.lower()

    # Rule-based intent detection
    if any(k in text_lower for k in ["recommend", "what game should i play", "suggestion"]):
        selected_route = "game_recommendation"
    elif any(k in text_lower for k in ["backlog", "completed", "add to my backlog", "mark hades"]):
        selected_route = "backlog_management"
    else:
        selected_route = "general_gaming_chat"

    return {
        "route": selected_route,
        "activity_log": [f"Route selected: {selected_route}"],
    }


def choose_route(state: GamingState) -> str:
    """
    Conditional routing function for LangGraph.
    Reads the 'route' value decided in route_intent and directs to the appropriate node.
    """
    return state.get("route", "general_gaming_chat")


def game_recommendation(state: GamingState) -> dict:
    """
    Handler for game recommendation requests using local Ollama model (llama3.2:3b).
    Trims history, injects long-term profile preferences, and generates game suggestions.
    """
    trimmed_messages, trim_stats = trim_conversation(state)
    profile = state.get("profile", {})

    genres = ", ".join(profile.get("favorite_genres", []))
    platforms = ", ".join(profile.get("platforms", []))

    profile_context = ""
    if genres or platforms:
        profile_context = (
            f"User's Long-Term Gaming Preferences:\n"
            f"- Favorite Genres: {genres if genres else 'Any'}\n"
            f"- Preferred Platforms: {platforms if platforms else 'Any'}\n"
            "Use these preferences to give personalized recommendations."
        )
    else:
        profile_context = "User's Long-Term Gaming Preferences: None recorded yet."

    system_prompt = SystemMessage(
        content=(
            "You are GameMate, a friendly and knowledgeable personal gaming companion. "
            f"{profile_context}\n"
            "Recommend great video games based on their preferences and dialogue. "
            "Do NOT claim to remember anything not present in the profile or dialogue. "
            "Keep recommendations engaging, tailored, and concise (2-4 sentences)."
        )
    )

    prompt = [system_prompt] + trimmed_messages
    ai_response = llm.invoke(prompt)

    return {
        "messages": [ai_response],
        "trim_stats": trim_stats,
        "activity_log": [
            f"Messages trimmed: {trim_stats['before']} -> {trim_stats['after']}",
            "Recommendation generated by Ollama",
        ],
    }


def backlog_management(state: GamingState, *, store: BaseStore) -> dict:
    """
    Handler for backlog queries and updates using local Ollama model (llama3.2:3b).
    Persists updates to InMemoryStore and generates confirmation or backlog reviews.
    """
    user_id = state.get("user_id", "player-1")
    current_backlog = list(state.get("backlog", []))
    activity_entries = []

    messages = state.get("messages", [])
    user_text = ""
    if messages:
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # Check if a game should be added to backlog
    game_to_add = extract_backlog_game(user_text)
    if game_to_add:
        if game_to_add not in current_backlog:
            current_backlog.append(game_to_add)
        store.put(("gaming_backlog", user_id), "backlog", {"games": current_backlog})
        activity_entries.append(f"Backlog updated: {game_to_add} added")
    else:
        activity_entries.append("Backlog reviewed")

    trimmed_messages, trim_stats = trim_conversation(state)
    activity_entries.append(f"Messages trimmed: {trim_stats['before']} -> {trim_stats['after']}")

    backlog_str = ", ".join(current_backlog) if current_backlog else "empty"
    system_prompt = SystemMessage(
        content=(
            "You are GameMate, a friendly personal gaming companion. "
            f"The user's current backlog consists of: {backlog_str}. "
            "Help the player track, prioritize, or confirm changes to their backlog. "
            "If the user asks what is in their backlog, list the games. "
            "If they added a game, confirm it was added. "
            "Keep your response natural, helpful, and concise (2-4 sentences)."
        )
    )

    prompt = [system_prompt] + trimmed_messages
    ai_response = llm.invoke(prompt)
    activity_entries.append("Backlog response generated by Ollama")

    return {
        "messages": [ai_response],
        "backlog": current_backlog,
        "trim_stats": trim_stats,
        "activity_log": activity_entries,
    }


def general_gaming_chat(state: GamingState) -> dict:
    """
    Handler for general gaming questions, lore, and casual chat using local Ollama model (llama3.2:3b).
    Uses short-term conversation context and known profile attributes.
    """
    trimmed_messages, trim_stats = trim_conversation(state)
    profile = state.get("profile", {})

    genres = ", ".join(profile.get("favorite_genres", []))
    platforms = ", ".join(profile.get("platforms", []))

    known_info = ""
    if genres or platforms:
        known_info = (
            f"\nKnown player profile:\n"
            f"- Favorite Genres: {genres if genres else 'None'}\n"
            f"- Platforms: {platforms if platforms else 'None'}\n"
        )

    system_prompt = SystemMessage(
        content=(
            "You are GameMate, a friendly and enthusiastic personal gaming companion. "
            f"{known_info}"
            "Answer gaming questions, explain concepts/terms, or discuss gaming lore conversationally. "
            "If the user asks what games or genres they like, reference what they told you in the conversation or profile. "
            "Keep answers concise and friendly (2-4 sentences)."
        )
    )

    prompt = [system_prompt] + trimmed_messages
    ai_response = llm.invoke(prompt)

    return {
        "messages": [ai_response],
        "trim_stats": trim_stats,
        "activity_log": [
            f"Messages trimmed: {trim_stats['before']} -> {trim_stats['after']}",
            "General chat reply generated by Ollama",
        ],
    }


# -----------------------------------------------------------------------------
# 7. LangGraph Workflow Construction
# -----------------------------------------------------------------------------
workflow = StateGraph(GamingState)

# Add all 5 nodes
workflow.add_node("load_profile", load_profile)
workflow.add_node("route_intent", route_intent)
workflow.add_node("game_recommendation", game_recommendation)
workflow.add_node("backlog_management", backlog_management)
workflow.add_node("general_gaming_chat", general_gaming_chat)

# Define standard linear edges
workflow.add_edge(START, "load_profile")
workflow.add_edge("load_profile", "route_intent")

# CONDITIONAL EDGE: route_intent dynamically branches to one of 3 nodes
workflow.add_conditional_edges(
    "route_intent",
    choose_route,
    {
        "game_recommendation": "game_recommendation",
        "backlog_management": "backlog_management",
        "general_gaming_chat": "general_gaming_chat",
    },
)

# Terminal edges: each destination node connects to END
workflow.add_edge("game_recommendation", END)
workflow.add_edge("backlog_management", END)
workflow.add_edge("general_gaming_chat", END)

# Compile the workflow with InMemorySaver checkpointer and InMemoryStore
graph = workflow.compile(
    checkpointer=checkpointer,
    store=store,
)


# -----------------------------------------------------------------------------
# 8. API Endpoints
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "thread-1"
    user_id: str = "player-1"


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.get("/profile")
async def get_profile(user_id: str = "player-1"):
    """
    Returns the current long-term profile and backlog from InMemoryStore for user_id.
    Useful for UI updates when switching threads or viewing persistent state.
    """
    profile_item = store.get(("gaming_profile", user_id), "profile")
    profile = dict(profile_item.value) if profile_item and profile_item.value else {"favorite_genres": [], "platforms": [], "preferences": []}
    backlog_item = store.get(("gaming_backlog", user_id), "backlog")
    backlog = list(backlog_item.value.get("games", [])) if backlog_item and backlog_item.value else []
    return {
        "user_id": user_id,
        "profile": profile,
        "backlog": backlog
    }


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Invokes the compiled LangGraph StateGraph with the user's input,
    utilizing InMemorySaver checkpointer (thread_id) and InMemoryStore (user_id).
    """
    config = {
        "configurable": {
            "thread_id": payload.thread_id
        }
    }

    # Pass only the new turn's message.
    # The checkpointer automatically restores previous conversation turns for this thread_id!
    initial_state = {
        "messages": [HumanMessage(content=payload.message)],
        "activity_log": [],
        "user_id": payload.user_id,
        "thread_id": payload.thread_id,
        "route": "",
        "trim_stats": {"before": 0, "after": 0},
    }

    # Execute the LangGraph StateGraph with checkpoint configuration
    final_state = graph.invoke(initial_state, config=config)

    # Extract the assistant's reply from the last message
    last_message = final_state["messages"][-1]
    response_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Format compact representation of recent messages for UI inspector
    short_term_memory = []
    for m in final_state.get("messages", []):
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        content = m.content if hasattr(m, "content") else str(m)
        short_term_memory.append({"role": role, "content": content})

    return {
        "response": response_text,
        "route": final_state.get("route", ""),
        "activity_log": final_state.get("activity_log", []),
        "trim_stats": final_state.get("trim_stats", {"before": 0, "after": 0}),
        "thread_id": final_state.get("thread_id", payload.thread_id),
        "user_id": final_state.get("user_id", payload.user_id),
        "profile": final_state.get("profile", {}),
        "backlog": final_state.get("backlog", []),
        "short_term_memory": short_term_memory,
    }

