"""
持久化管理
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph = StateGraph(State)


@tool
def query_weather(query: str) -> str:
    """Query weather info for a place and return a short summary."""
    if "上海" in query or "天气" in query:
        print("进入tool调用，询问上海天气")
        return "明天上午有雨，下午阴天"
    return "明天天晴"