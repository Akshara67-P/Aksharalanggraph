import os
import sys
import subprocess
import tempfile
from typing import TypedDict, List, Optional, Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from langserve import add_routes

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. WORKFLOW STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    execution_output: Optional[str]
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
# 3. CONVERT GEMINI RESPONSE TO NORMAL TEXT
# ============================================================

def response_to_text(response) -> str:

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                if "text" in item:
                    parts.append(str(item["text"]))

        return "\n".join(parts)

    return str(content)


# ============================================================
# 4. CLEAN PYTHON CODE
# ============================================================

def clean_code(code: str) -> str:

    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):].strip()

    elif code.startswith("```"):
        code = code[3:].strip()

    if code.endswith("```"):
        code = code[:-3].strip()

    return code


# ============================================================
# 5. EXECUTE GENERATED PYTHON CODE
# ============================================================

def run_python_code(code: str) -> str:

    code = clean_code(code)

    file_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as file:

            file.write(code)
            file_path = file.name

        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:

            if output:
                return output

            return "Program executed successfully with no output."

        return "PROGRAM ERROR:\n" + error

    except subprocess.TimeoutExpired:

        return (
            "PROGRAM ERROR:\n"
            "Program exceeded the 10-second execution limit."
        )

    except Exception as e:

        return "PROGRAM ERROR:\n" + str(e)

    finally:

        if file_path:

            try:
                os.remove(file_path)
            except OSError:
                pass


# ============================================================
# 6. TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):

    return {
        "next_step": "developer"
    }


# ============================================================
# 7. DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    messages = state.get("messages", [])

    if messages:
        task = messages[-1].content
    else:
        task = "Create a simple Python program."

    prompt = f"""
You are a real-time Python developer.

Programming task:

{task}

Write a complete and executable Python program.

Rules:
1. Return ONLY Python code.
2. Do not use Markdown.
3. Do not use ``` symbols.
4. Make the program directly executable.
5. Include a sample value when the task needs an input.
"""

    response = llm.invoke(prompt)

    generated_code = clean_code(
        response_to_text(response)
    )

    return {
        "code": generated_code,
        "next_step": "tester"
    }


# ============================================================
# 8. TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    code = state.get("code", "")

    # Actually execute generated code
    execution_output = run_python_code(code)

    prompt = f"""
You are a Senior QA Engineer.

Generated Python code:

---------------- CODE ----------------

{code}

-------------- END CODE --------------

Actual execution result:

------------- RESULT -----------------

{execution_output}

----------- END RESULT ---------------

Prepare a testing report containing:

1. Test scenarios
2. Expected results
3. Actual execution result
4. Possible issues
5. Overall testing status

Use the actual execution result above.
Do not invent results.
"""

    response = llm.invoke(prompt)

    report = response_to_text(response)

    return {
        "execution_output": execution_output,
        "report": report,
        "next_step": "manager_decision"
    }


# ============================================================
# 9. MANAGER DECISION NODE
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

    return {
        "next_step": "task_input"
    }


# ============================================================
# 10. ARCHIVER NODE
# ============================================================

def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# ============================================================
# 11. ROUTING
# ============================================================

def route_from_input(state: CrewState):

    return "developer"


def route_from_manager(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


# ============================================================
# 12. BUILD LANGGRAPH
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
    route_from_manager
)

workflow.add_edge(
    "archiver",
    END
)


graph = workflow.compile()


# ============================================================
# 13. PLAYGROUND INPUT
# ============================================================

class AgentInput(BaseModel):

    input: str

    manager_choice: Literal[
        "store",
        "another"
    ] = "store"


# ============================================================
# 14. PLAYGROUND OUTPUT
# ============================================================

class AgentOutput(BaseModel):

    status: str

    generated_code: Optional[str] = None

    execution_output: Optional[str] = None

    testing_report: Optional[str] = None

    next_step: Optional[str] = None

    manager_choice: Optional[str] = None

    error: Optional[str] = None


# ============================================================
# 15. RUN AGENT
# ============================================================

def run_agent(data):

    try:

        user_input = data.get(
            "input",
            ""
        )

        manager_choice = data.get(
            "manager_choice",
            "store"
        )

        if not user_input.strip():

            return {
                "status": "error",
                "error": "Please enter a programming task."
            }

        initial_state: CrewState = {

            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],

            "next_step": None,

            "code": None,

            "execution_output": None,

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
                "code",
                ""
            ),

            "execution_output": result.get(
                "execution_output",
                ""
            ),

            "testing_report": result.get(
                "report",
                ""
            ),

            "next_step": result.get(
                "next_step",
                ""
            ),

            "manager_choice": result.get(
                "manager_choice",
                manager_choice
            )
        }

    except Exception as e:

        return {

            "status": "error",

            "error": str(e)
        }


# ============================================================
# 16. LANGSERVE
# ============================================================

agent_chain = RunnableLambda(
    run_agent
).with_types(
    input_type=AgentInput,
    output_type=AgentOutput
)


# ============================================================
# 17. FASTAPI
# ============================================================

app = FastAPI(
    title="LangGraph Real-Time Developer Workflow",
    version="1.0"
)


# ============================================================
# 18. /agent ROUTE
# ============================================================

add_routes(
    app,
    agent_chain,
    path="/agent"
)


# ============================================================
# 19. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
