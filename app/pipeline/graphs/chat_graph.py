from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.pipeline.state import ChatState
from app.pipeline.chat.agent import run_chat_agent

builder = StateGraph(ChatState)

builder.add_node("chat_agent", run_chat_agent)

builder.set_entry_point("chat_agent")
builder.add_edge("chat_agent", END)

checkpointer = MemorySaver()
chat_graph = builder.compile(checkpointer=checkpointer)
