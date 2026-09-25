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
# STATE
# =========================

class CrewState(TypedDict):
    messages: List
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    manager_choice: Optional[str]


# =========================
# GEMINI
# =========================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# =========================
# TASK INPUT
# =========================

def task_input_node(state):
    return {
        "next_step": "developer"
    }


# =========================
# DEVELOPER
# =========================

def developer_node(state):

    task = state["messages"][-1].content

    prompt = f"""
You are a Python developer.

Task:
{task}

Write a correct Python program.

Return only the Python code.
"""

    response = llm.invoke(prompt)

    return {
        "code": response.content,
        "next_step": "tester"
    }


# =========================
# TESTER
# =========================

def tester_node(state):

    code = state.get("code", "")

    prompt = f"""
You are a software tester.

Test/analyze this Python code:

{code}

Give a simple testing report with:

1. Test scenarios
2. Expected results
3. Possible issues
4. Overall status
"""

    response = llm.invoke(prompt)

    return {
        "report": response.content,
        "next_step": "manager_decision"
    }


# =========================
# MANAGER
# =========================

def manager_node(state):

    choice = state.get("manager_choice", "store")

    if choice.lower() == "store":
        return {
            "next_step": "archiver"
        }

    return {
        "next_step": "task_input"
    }


# =========================
# ARCHIVER
# =========================

def archiver_node(state):

    return {
        "next_step": "exit"
    }


# =========================
# ROUTING
# =========================

def route_input(state):

    return "developer"


def route_manager(state):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


# =========================
# LANGGRAPH
# =========================

workflow = StateGraph(CrewState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager_decision", manager_node)
workflow.add_node("archiver", archiver_node)

workflow.add_edge(START, "task_input")

workflow.add_conditional_edges(
    "task_input",
    route_input
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    "manager_decision"
)

workflow.add_conditional_edges(
    "manager_decision",
    route_manager
)

workflow.add_edge(
    "archiver",
    END
)

graph = workflow.compile()


# =========================
# PLAYGROUND INPUT
# =========================

class AgentInput(BaseModel):
    input: str
    manager_choice: str = "store"


# =========================
# RUN AGENT
# =========================

def run_agent(data):

    try:

        user_input = data["input"]

        manager_choice = data.get(
            "manager_choice",
            "store"
        )

        state = {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "next_step": None,
            "code": None,
            "report": None,
            "manager_choice": manager_choice
        }

        result = graph.invoke(
            state,
            config={
                "recursion_limit": 20
            }
        )

        return {
            "status": "success",
            "generated_code": result.get("code"),
            "testing_report": result.get("report"),
            "next_step": result.get("next_step")
        }

    except Exception as e:

        return {
            "status": "error",
            "error": str(e)
        }


# =========================
# LANGSERVE
# =========================

agent_chain = RunnableLambda(
    run_agent
).with_types(
    input_type=AgentInput
)


# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="LangGraph Real-Time Developer Workflow",
    version="1.0"
)


add_routes(
    app,
    agent_chain,
    path="/agent"
)


# =========================
# SERVER
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
