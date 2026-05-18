"""
Meta-Controller - Agentic Architecture
Extracted from: 11_meta_controller.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv langchain-tavily

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Pydantic for data modeling
from pydantic import BaseModel, Field

# LangChain components
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

# LangGraph components
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Meta-Controller (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()
llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0)
search_tool = TavilySearch(max_results=3)

# Define the state for the overall graph
class MetaAgentState(TypedDict):
    user_request: str
    next_agent_to_call: Optional[str]
    generation: str

# A helper factory function to create specialist agent nodes
def create_specialist_node(persona: str, tools: list = None):
    """Factory to create a specialist agent node."""
    system_prompt = f"You are a specialist agent with the following persona: {persona}. Respond directly and concisely to the user's request based on your role."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_request}")
    ])
    
    if tools:
        chain = prompt | llm.bind_tools(tools)
    else:
        chain = prompt | llm
        
    def specialist_node(state: MetaAgentState) -> Dict[str, Any]:
        result = chain.invoke({"user_request": state['user_request']})
        return {"generation": result.content}
    
    return specialist_node

# 1. Generalist Agent Node
generalist_node = create_specialist_node(
    "You are a friendly and helpful generalist AI assistant. You handle casual conversation and simple questions."
)

# 2. Research Agent Node
research_agent_node = create_specialist_node(
    "You are an expert researcher. You must use your search tool to find information to answer the user's question.",
    tools=[search_tool]
)

# 3. Coding Agent Node
coding_agent_node = create_specialist_node(
    "You are an expert Python programmer. Your task is to write clean, efficient Python code based on the user's request. Provide only the code, wrapped in markdown code blocks, with minimal explanation."
)

print("Specialist agents defined successfully.")

# Pydantic model for the controller's routing decision
class ControllerDecision(BaseModel):
    next_agent: str = Field(description="The name of the specialist agent to call next. Must be one of ['Generalist', 'Researcher', 'Coder'].")
    reasoning: str = Field(description="A brief reason for choosing the next agent.")

def meta_controller_node(state: MetaAgentState) -> Dict[str, Any]:
    """The central controller that decides which specialist to call."""
    console.print("--- 🧠 Meta-Controller Analyzing Request ---")
    
    # Define the specialists and their descriptions for the controller
    specialists = {
        "Generalist": "Handles casual conversation, greetings, and simple questions.",
        "Researcher": "Answers questions about recent events, complex topics, or anything requiring up-to-date information from the web.",
        "Coder": "Writes Python code based on a user's specification."
    }
    
    specialist_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in specialists.items()])
    
    prompt = ChatPromptTemplate.from_template(
        f"""You are the meta-controller for a multi-agent AI system. Your job is to analyze the user's request and route it to the most appropriate specialist agent.

Here are the available specialists:
{specialist_descriptions}

Analyze the following user request and choose the best specialist to handle it. Provide your decision in the required format.

User Request: "{{user_request}}""""
    )
    
    controller_llm = llm.with_structured_output(ControllerDecision)
    chain = prompt | controller_llm
    
    decision = chain.invoke({"user_request": state['user_request']})
    console.print(f"[yellow]Routing decision:[/yellow] Send to [bold]{decision.next_agent}[/bold]. [italic]Reason: {decision.reasoning}[/italic]")
    
    return {"next_agent_to_call": decision.next_agent}

print("Meta-Controller node defined successfully.")

# Build the graph
workflow = StateGraph(MetaAgentState)

# Add nodes for the controller and each specialist
workflow.add_node("meta_controller", meta_controller_node)
workflow.add_node("Generalist", generalist_node)
workflow.add_node("Researcher", research_agent_node)
workflow.add_node("Coder", coding_agent_node)

# Set the entry point
workflow.set_entry_point("meta_controller")

# Define the conditional routing logic
def route_to_specialist(state: MetaAgentState) -> str:
    """Reads the controller's decision and returns the name of the node to route to."""
    return state["next_agent_to_call"]

workflow.add_conditional_edges(
    "meta_controller",
    route_to_specialist,
    {
        "Generalist": "Generalist",
        "Researcher": "Researcher",
        "Coder": "Coder"
    }
)

# After any specialist runs, the process ends
workflow.add_edge("Generalist", END)
workflow.add_edge("Researcher", END)
workflow.add_edge("Coder", END)

meta_agent = workflow.compile()
print("Meta-Controller agent graph compiled successfully.")

def run_agent(query: str):
    result = meta_agent.invoke({"user_request": query})
    console.print("\n[bold]Final Response:[/bold]")
    console.print(Markdown(result['generation']))

# Test 1: Should be routed to the Generalist
console.print("--- 💬 Test 1: General Conversation ---")
run_agent("Hello, how are you today?")

# Test 2: Should be routed to the Researcher
console.print("\n--- 🔬 Test 2: Research Question ---")
run_agent("What were NVIDIA's latest financial results?")

# Test 3: Should be routed to the Coder
console.print("\n--- 💻 Test 3: Coding Request ---")
run_agent("Can you write me a python function to calculate the nth fibonacci number?")
