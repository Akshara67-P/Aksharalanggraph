import os
import uvicorn
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# =========================
# 1. STATE
# =========================

class CrewState(TypedDict):
    messages: List
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    manager_choice: Optional[str]


# =========================
# 2. GEMINI MODEL
# =========================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# =========================
# 3. TASK INPUT NODE
# =========================

def task_input_node(state: CrewState):
    return {
        "next_step": "developer"
    }


# =========================
# 4. DEVELOPER NODE
# =========================

def real_time_developer(state: CrewState):

    messages = state.get("messages", [])

    if messages:
        task = messages[-1].content
    else:
        task = "No task provided."

    prompt = f"""
You are a real-time developer.

Write a Python program for the following coding task:

{task}

Return only the Python code.
"""

    response = llm_flash.invoke(prompt)

    if hasattr(response, "content"):
        generated_code = response.content
    else:
        generated_code = str(response)

    return {
        "code": generated_code,
        "next_step": "tester"
    }


# =========================
# 5. TESTER NODE
# =========================

def real_time_tester(state: CrewState):

    code = state.get("code", "")

    prompt = f"""
You are a Senior QA Engineer.

Analyze the following Python code:

{code}

Generate a simple testing report containing:

1. Test scenarios
2. Expected result
3. Possible issues
4. Overall testing status
"""

    response = llm_flash.invoke(prompt)

    if hasattr(response, "content"):
        report = response.content
    else:
        report = str(response)

    return {
        "report": report,
        "next_step": "manager_decision"
    }


# =========================
# 6. MANAGER DECISION
# =========================

def manager_decision_node(state: CrewState):

    choice = state.get("manager_choice", "store")

    if choice.lower() == "store":
        return {
            "next_step": "archiver"
        }

    return {
        "next_step": "task_input"
    }


# =========================
# 7. ARCHIVER
# =========================

def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# =========================
# 8. ROUTING
# =========================

def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return END

    return "developer"


def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


# =========================
# 9. LANGGRAPH WORKFLOW
# =========================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)

rt_workflow.add_edge(
    START,
    "task_input"
)

rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)

rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)

rt_workflow.add_edge(
    "archiver",
    END
)

rt_app = rt_workflow.compile()


# =========================
# 10. PLAYGROUND INPUT
# =========================

class AgentInput(BaseModel):
    input: str
    manager_choice: str = "store"


# =========================
# 11. RUN AGENT
# =========================

def run_agent(data):

    user_input = data.get(
        "input",
        "Create a Python program."
    )

    manager_choice = data.get(
        "manager_choice",
        "store"
    )

    initial_state = {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "next_step": None,
        "code": None,
        "report": None,
        "manager_choice": manager_choice
    }

    result = rt_app.invoke(initial_state)

    return {
        "code": result.get("code"),
        "report": result.get("report"),
        "next_step": result.get("next_step"),
        "manager_choice": result.get("manager_choice")
    }


# =========================
# 12. LANGSERVE CHAIN
# =========================

agent_chain = RunnableLambda(
    run_agent
).with_types(
    input_type=AgentInput
)


# =========================
# 13. FASTAPI
# =========================

app = FastAPI(
    title="LangGraph Verilog Workflow",
    version="1.0"
)


# =========================
# 14. /agent ROUTE
# =========================

add_routes(
    app,
    agent_chain,
    path="/agent"
)


# =========================
# 15. START SERVER
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
