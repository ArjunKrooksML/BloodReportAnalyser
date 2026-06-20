import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.pipeline.state import ChatState
from app.pipeline.chat.tools import make_tools

log = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)

SYSTEM_PROMPT = """You are a medical report assistant. You have access to the patient's extracted lab results.
Use the available tools to look up values, check ranges, and retrieve anomalies before answering.
Always ground your answer in the actual data from the report.
Never make up values. If something is not in the report, say so.
Add a brief disclaimer that you are not a substitute for professional medical advice."""


def run_chat_agent(state: ChatState) -> dict:
    log.info("[Agent] Starting ReAct loop | session=%s | question=%.80s", state["session_id"], state["question"])

    tools = make_tools(state)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in state.get("chat_history", []):
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["response"]))
    messages.append(HumanMessage(content=state["question"]))

    iteration = 0
    while True:
        iteration += 1
        log.debug("[Agent] LLM call | session=%s | iteration=%d", state["session_id"], iteration)
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            log.info("[Agent] No more tool calls — generating final response | session=%s | iterations=%d",
                     state["session_id"], iteration)
            break

        for call in response.tool_calls:
            log.debug("[Agent] Tool call | session=%s | tool=%s | args=%s", state["session_id"], call["name"], call["args"])
            result = tool_map[call["name"]].invoke(call["args"])
            log.debug("[Agent] Tool result | tool=%s | result=%.120s", call["name"], str(result))
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final_response = response.content
    log.info("[Agent] Done | session=%s | response_chars=%d", state["session_id"], len(final_response))

    return {
        "response": final_response,
        "chat_history": [{"question": state["question"], "response": final_response}]
    }
