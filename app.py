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


# ============================================================
# 1. WORKFLOW STATE
# ============================================================

class CrewState(TypedDict):
    messages: List
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    manager_choice: Optional[str]


# ============================================================
# 2. GEMINI MODEL
# ============================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):

    return {
        "next_step": "developer"
    }


# ============================================================
# 4. REAL-TIME DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    messages = state.get("messages", [])

    if messages:
        task = messages[-1].content
    else:
        task = "No task provided."

    prompt = f"""
You are a real-time Python developer.

Write a correct Python program for the following task:

{task}

Requirements:
- Write simple and correct Python code.
- Return only the Python code.
- Do not include explanations.
- Do not use Markdown code fences.
"""

    response = llm.invoke(prompt)

    generated_code = response.content

    return {
        "code": generated_code,
        "next_step": "tester"
    }


# ============================================================
# 5. REAL-TIME TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    code = state.get("code", "")

    prompt = f"""
You are a Senior QA Engineer.

Analyze and test the following Python program:

{code}

Create a simple testing report containing:

1. Test scenarios
2. Expected results
3. Possible issues
4. Overall testing status

Keep the report clear and simple.
"""

    response = llm.invoke(prompt)

    report = response.content

    return {
        "report": report,
        "next_step": "manager_decision"
    }


# ============================================================
# 6. MANAGER DECISION NODE
# ============================================================

def manager_decision_node(state: CrewState):

    choice = state.get(
        "manager_choice",
        "store"
    )

    if choice.lower() == "store":

        return {
            "next_step": "archiver"
        }

    else:

        return {
            "next_step": "task_input"
        }


# ============================================================
# 7. ARCHIVER NODE
# ============================================================

def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# ============================================================
# 8. ROUTING FUNCTIONS
# ============================================================

def route_from_input(state: CrewState):

    return "developer"


def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":

        return "archiver"

    return "task_input"


# ============================================================
# 9. CREATE LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(CrewState)


workflow.add_node(
    "task_input",
    task_input_node
)

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "manager_decision",
    manager_decision_node
)

workflow.add_node(
    "archiver",
    archiver_node
)


# ============================================================
# 10. WORKFLOW EDGES
# ============================================================

workflow.add_edge(
    START,
    "task_input"
)

workflow.add_conditional_edges(
    "task_input",
    route_from_input
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
    route_from_decision
)

workflow.add_edge(
    "archiver",
    END
)


# Compile LangGraph
graph = workflow.compile()


# ============================================================
# 11. PLAYGROUND INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    input: str

    manager_choice: str = "store"


# ============================================================
# 12. RUN AGENT
# ============================================================

def run_agent(data):

    try:

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
                HumanMessage(
                    content=user_input
                )
            ],

            "next_step": None,

            "code": None,

            "report": None,

            "manager_choice": manager_choice
        }


        result = graph.invoke(
            initial_state,
            config={
                "recursion_limit": 20
            }
        )


        return {

            "status": "success",

            "generated_code": result.get(
                "code"
            ),

            "testing_report": result.get(
                "report"
            ),

            "next_step": result.get(
                "next_step"
            ),

            "manager_choice": result.get(
                "manager_choice"
            )
        }


    except Exception as e:

        return {

            "status": "error",

            "error": str(e)
        }


# ============================================================
# 13. LANGSERVE CHAIN
# ============================================================

agent_chain = RunnableLambda(
    run_agent
).with_types(
    input_type=AgentInput
)


# ============================================================
# 14. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LangGraph Real-Time Developer Workflow",
    version="1.0"
)


# ============================================================
# 15. LANGSERVE /agent ROUTE
# ============================================================

add_routes(
    app,
    agent_chain,
    path="/agent"
)


# ============================================================
# 16. START SERVER
# ============================================================

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
