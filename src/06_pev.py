"""
PEV - Agentic Architecture
Extracted from: 06_PEV.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv langchain-tavily

import os
import re
from typing import List, Annotated, TypedDict, Optional
from dotenv import load_dotenv
import json

# LangChain components
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel, Field

# LangGraph components
from langgraph.graph import StateGraph, END

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - PEV (Nebius)"

for key in ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} not found. Please create a .env file and set it.")

print("Environment variables loaded and tracing is set up.")

console = Console()
llm = ChatNebius(model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0)

# Define a 'flaky' tool that will fail for a specific query
def flaky_web_search(query: str) -> str:
    """Performs a web search, but is designed to fail for a specific query."""
    console.print(f"--- TOOL: Searching for '{query}'... ---")
    if "employee count" in query.lower():
        console.print("--- TOOL: [bold red]Simulating API failure![/bold red] ---")
        return "Error: Could not retrieve data. The API endpoint is currently unavailable."
    else:
        result = TavilySearch(max_results=2).invoke(query)
        # 🔑 Ensure result is always a string
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)

# Define the state for the basic P-E agent
class BasicPEState(TypedDict):
    user_request: str
    plan: Optional[List[str]]
    intermediate_steps: List[str]
    final_answer: Optional[str]

class Plan(BaseModel):
    steps: List[str] = Field(description="A list of tool calls to execute.")

def basic_planner_node(state: BasicPEState):
    console.print("--- (Basic) PLANNER: Creating plan... ---")
    planner_llm = llm.with_structured_output(Plan)

    prompt = f"""
    You are a planning agent. 
    Your job is to decompose the user's request into a list of clear tool queries.

    - Only return JSON that matches this schema: {{ "steps": [ "query1", "query2", ... ] }}
    - Do NOT return any prose or explanation.
    - Always use the 'flaky_web_search' tool for queries.

    User's request: "{state['user_request']}"
    """
    plan = planner_llm.invoke(prompt)
    return {"plan": plan.steps}

def basic_executor_node(state: BasicPEState):
    console.print("--- (Basic) EXECUTOR: Running next step... ---")
    next_step = state["plan"][0]
    result = flaky_web_search(next_step)
    return {"plan": state["plan"][1:], "intermediate_steps": state["intermediate_steps"] + [result]}

def basic_synthesizer_node(state: BasicPEState):
    console.print("--- (Basic) SYNTHESIZER: Generating final answer... ---")
    context = "\n".join(state["intermediate_steps"])
    prompt = f"Synthesize an answer for '{state['user_request']}' using this data:\n{context}"
    answer = llm.invoke(prompt).content
    return {"final_answer": answer}

# Build the graph
pe_graph_builder = StateGraph(BasicPEState)
pe_graph_builder.add_node("plan", basic_planner_node)
pe_graph_builder.add_node("execute", basic_executor_node)
pe_graph_builder.add_node("synthesize", basic_synthesizer_node)

pe_graph_builder.set_entry_point("plan")
pe_graph_builder.add_conditional_edges("plan", lambda s: "execute" if s["plan"] else "synthesize")
pe_graph_builder.add_conditional_edges("execute", lambda s: "execute" if s["plan"] else "synthesize")
pe_graph_builder.add_edge("synthesize", END)

basic_pe_app = pe_graph_builder.compile()
print("Basic Planner-Executor agent compiled successfully.")

flaky_query = "What was Apple's R&D spend in their last fiscal year, and what was their total employee count? Calculate the R&D spend per employee."

console.print(f"[bold yellow]Testing BASIC P-E agent on a flaky query:[/bold yellow]\n'{flaky_query}'\n")

initial_pe_input = {"user_request": flaky_query, "intermediate_steps": []}
final_pe_output = basic_pe_app.invoke(initial_pe_input)

console.print("\n--- [bold red]Final Output from Basic P-E Agent[/bold red] ---")
console.print(Markdown(final_pe_output['final_answer']))

class VerificationResult(BaseModel):
    """Schema for the Verifier's output."""
    is_successful: bool = Field(description="True if the tool execution was successful and the data is valid.")
    reasoning: str = Field(description="Reasoning for the verification decision.")

class PEVState(TypedDict):
    user_request: str
    plan: Optional[List[str]]
    last_tool_result: Optional[str]
    intermediate_steps: List[str]
    final_answer: Optional[str]
    retries: int  # count how many times we’ve replanned

from langchain_core.exceptions import OutputParserException

class Plan(BaseModel):
    steps: List[str] = Field(
        description="List of queries (max 5).",
        max_items=5
    )

def pev_planner_node(state: PEVState):
    retries = state.get("retries", 0)
    if retries > 3:  # stop after 3 replans
        console.print("--- (PEV) PLANNER: Retry limit reached. Stopping. ---")
        return {
            "plan": [],
            "final_answer": "Error: Unable to complete task after multiple retries."
        }

    console.print(f"--- (PEV) PLANNER: Creating/revising plan (retry {retries})... ---")

    planner_llm = llm.with_structured_output(Plan, strict=True)  # ✅ strict schema

    past_context = "\n".join(state["intermediate_steps"])
    base_prompt = f"""
    You are a planning agent. 
    Create a plan to answer: '{state['user_request']}'. 
    Use the 'flaky_web_search' tool.

    Rules:
    - Return ONLY valid JSON in this exact format: {{ "steps": ["query1", "query2"] }}
    - Maximum 5 steps.
    - Do NOT repeat failed queries or endless variations.
    - Do NOT output explanations, only JSON.

    Previous attempts and results:
    {past_context}
    """

    # ✅ retry wrapper for bad JSON
    for attempt in range(2):
        try:
            plan = planner_llm.invoke(base_prompt)
            return {"plan": plan.steps, "retries": retries + 1}
        except OutputParserException as e:
            console.print(f"[red]Planner parsing failed (attempt {attempt+1}): {e}[/red]")
            base_prompt = f"Return ONLY valid JSON with {{'steps': ['...']}}. {base_prompt}"

    # ultimate fallback to avoid crashing
    return {"plan": ["Apple R&D spend last fiscal year"], "retries": retries + 1}



def pev_executor_node(state: PEVState):
    if not state.get("plan"):  # ✅ guard against empty plan
        console.print("--- (PEV) EXECUTOR: No steps left, skipping execution. ---")
        return {}
    
    console.print("--- (PEV) EXECUTOR: Running next step... ---")
    next_step = state["plan"][0]
    result = flaky_web_search(next_step)
    return {"plan": state["plan"][1:], "last_tool_result": result}

def verifier_node(state: PEVState):
    console.print("--- VERIFIER: Checking last tool result... ---")
    verifier_llm = llm.with_structured_output(VerificationResult)
    prompt = f"Verify if the following tool output is a successful result or an error message. The task was '{state['user_request']}'.\n\nTool Output: '{state['last_tool_result']}'"
    verification = verifier_llm.invoke(prompt)
    console.print(f"--- VERIFIER: Judgment is '{'Success' if verification.is_successful else 'Failure'}' ---")
    if verification.is_successful:
        # If successful, add the valid result to our list of good steps
        return {"intermediate_steps": state["intermediate_steps"] + [state['last_tool_result']]}
    else:
        # If failed, add the failure reason and trigger re-planning by clearing the plan
        return {"plan": [], "intermediate_steps": state["intermediate_steps"] + [f"Verification Failed: {state['last_tool_result']}"]}

pev_synthesizer_node = basic_synthesizer_node # We can reuse the same synthesizer

def pev_router(state: PEVState):
    # ✅ If we already have a final answer (e.g. retry limit reached), stop
    if state.get("final_answer"):
        console.print("--- ROUTER: Final answer available. Moving to synthesizer. ---")
        return "synthesize"

    if not state["plan"]:
        # Check if plan is empty because of verification failure
        if state["intermediate_steps"] and "Verification Failed" in state["intermediate_steps"][-1]:
            console.print("--- ROUTER: Verification failed. Re-planning... ---")
            return "plan"
        else:
            console.print("--- ROUTER: Plan complete. Moving to synthesizer. ---")
            return "synthesize"
    else:
        console.print("--- ROUTER: Plan has more steps. Continuing execution. ---")
        return "execute"


# Build the PEV graph
pev_graph_builder = StateGraph(PEVState)
pev_graph_builder.add_node("plan", pev_planner_node)
pev_graph_builder.add_node("execute", pev_executor_node)
pev_graph_builder.add_node("verify", verifier_node)
pev_graph_builder.add_node("synthesize", pev_synthesizer_node)

pev_graph_builder.set_entry_point("plan")
pev_graph_builder.add_edge("plan", "execute")
pev_graph_builder.add_edge("execute", "verify")
pev_graph_builder.add_conditional_edges("verify", pev_router)
pev_graph_builder.add_edge("synthesize", END)

pev_agent_app = pev_graph_builder.compile()
print("Planner-Executor-Verifier (PEV) agent compiled successfully.")

console.print(f"[bold green]Testing PEV agent on the same flaky query:[/bold green]\n'{flaky_query}'\n")

initial_pev_input = {"user_request": flaky_query, "intermediate_steps": [], "retries": 0}

final_pev_output = pev_agent_app.invoke(initial_pev_input)

console.print("\n--- [bold green]Final Output from PEV Agent[/bold green] ---")
console.print(Markdown(final_pev_output['final_answer']))

class RobustnessEvaluation(BaseModel):
    """Schema for evaluating an agent's robustness and error handling."""
    task_completion_score: int = Field(description="Score 1-10 on whether the agent successfully completed the task, ignoring data errors.")
    error_handling_score: int = Field(description="Score 1-10 on the agent's ability to detect and recover from errors.")
    justification: str = Field(description="A brief justification for the scores.")

judge_llm = llm.with_structured_output(RobustnessEvaluation)

def evaluate_agent_robustness(query: str, final_state: dict):
    context = "\n".join(final_state.get("intermediate_steps", []))
    final_answer = final_state.get("final_answer", "")
    trace = f"Context:\n{context}\n\nFinal Answer:\n{final_answer}"
        
    prompt = f"""You are an expert judge of AI agents. A tool used by the agent was designed to fail on a specific query. Evaluate the agent's ability to handle this failure.
    
    **User's Task:** {query}
    **Full Agent Trace:**\n```\n{trace}\n```
    """
    return judge_llm.invoke(prompt)

console.print("--- Evaluating Basic P-E Agent's Robustness ---")
pe_agent_evaluation = evaluate_agent_robustness(flaky_query, final_pe_output)
console.print(pe_agent_evaluation.model_dump())

console.print("\n--- Evaluating PEV Agent's Robustness ---")
pev_agent_evaluation = evaluate_agent_robustness(flaky_query, final_pev_output)
console.print(pev_agent_evaluation.model_dump())
