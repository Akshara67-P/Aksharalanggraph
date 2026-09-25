import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional


class CrewState(TypedDict):
    messages: List
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    manager_choice: Optional[str]


GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)


def task_input_node(state: CrewState):
    return {"next_step": "developer"}


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


def manager_decision_node(state: CrewState):

    choice = state.get("manager_choice", "store")

    if choice.lower() == "store":
        return {"next_step": "archiver"}

    return {"next_step": "task_input"}


def archiver_node(state: CrewState):
    return {"next_step": "exit"}


def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return END

    return "developer"


def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


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

    return rt_app.invoke(initial_state)


agent_chain = RunnableLambda(run_agent)


app = FastAPI(
    title="LangGraph Verilog Workflow",
    version="1.0"
)


add_routes(
    app,
    agent_chain,
    path="/agent"
)


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
