"""
Reflexive Metacognitive - Agentic Architecture
Extracted from: 17_reflexive_metacognitive.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Pydantic for data modeling
from pydantic import BaseModel, Field

# LangChain components
from langchain_nebius import ChatNebius
from langchain_core.prompts import ChatPromptTemplate

# LangGraph components
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Metacognitive Agent (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()

# --- The Agent's Self-Model ---
class AgentSelfModel(BaseModel):
    """A structured representation of the agent's capabilities and limitations."""
    name: str
    role: str
    # The agent's explicit knowledge boundaries
    knowledge_domain: List[str] = Field(description="List of topics the agent is knowledgeable about.")
    # The agent's available tools
    available_tools: List[str] = Field(description="List of tools the agent can use.")
    confidence_threshold: float = Field(description="The confidence level (0-1) below which the agent must escalate.", default=0.6)

# Instantiate the self-model for our Medical Triage Agent
medical_agent_model = AgentSelfModel(
    name="TriageBot-3000",
    role="A helpful AI assistant for providing preliminary medical information.",
    knowledge_domain=["common_cold", "influenza", "allergies", "headaches", "basic_first_aid"],
    available_tools=["drug_interaction_checker"]
)

# --- Specialist Tools ---
class DrugInteractionChecker:
    """A mock tool to check for drug interactions."""
    def check(self, drug_a: str, drug_b: str) -> str:
        """Checks for interactions between two drugs."""
        # In a real system, this would query a medical database.
        known_interactions = {
            frozenset(["ibuprofen", "lisinopril"]): "Moderate risk: Ibuprofen may reduce the blood pressure-lowering effects of lisinopril. Monitor blood pressure.",
            frozenset(["aspirin", "warfarin"]): "High risk: Increased risk of bleeding. This combination should be avoided unless directed by a doctor."
        }
        interaction = known_interactions.get(frozenset([drug_a.lower(), drug_b.lower()]))
        if interaction:
            return f"Interaction Found: {interaction}"
        return "No known significant interactions found. However, always consult a pharmacist or doctor."

drug_tool = DrugInteractionChecker()
print("Agent Self-Model and Tools defined successfully.")

llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0)

# Pydantic Models for structured outputs
class MetacognitiveAnalysis(BaseModel):
    """The agent's self-analysis of a query."""
    confidence: float = Field(description="A score from 0.0 to 1.0 representing the agent's confidence in its ability to answer safely and accurately.")
    strategy: str = Field(description="The chosen strategy. Must be one of: 'reason_directly', 'use_tool', 'escalate'.")
    reasoning: str = Field(description="A brief justification for the chosen confidence and strategy.")
    tool_to_use: Optional[str] = Field(description="If strategy is 'use_tool', the name of the tool to use.", default=None)
    tool_args: Optional[Dict[str, Any]] = Field(description="If strategy is 'use_tool', the arguments for the tool.", default=None)

# LangGraph State
class AgentState(TypedDict):
    user_query: str
    self_model: AgentSelfModel
    metacognitive_analysis: Optional[MetacognitiveAnalysis]
    tool_output: Optional[str]
    final_response: str

# Graph Nodes
def metacognitive_analysis_node(state: AgentState) -> Dict[str, Any]:
    console.print(Panel("🤔 Agent is performing metacognitive analysis...", title="[yellow]Step: Self-Reflection[/yellow]"))
    prompt = ChatPromptTemplate.from_template(
        """You are a metacognitive reasoning engine for an AI assistant. Your task is to analyze a user's query in the context of the agent's own capabilities and limitations (its 'self-model').
        Your primary directive is **SAFETY**. You must determine the safest and most appropriate strategy for handling the query.

        **Agent's Self-Model:**
        - Name: {agent_name}
        - Role: {agent_role}
        - Knowledge Domain: {knowledge_domain}
        - Available Tools: {available_tools}

        **Strategy Rules:**
        1.  **escalate:** Choose this strategy if the query involves a potential medical emergency (e.g., chest pain, difficulty breathing, severe injury, broken bones), is outside the agent's knowledge domain, or if you have any doubt about providing a safe answer. **WHEN IN DOUBT, ESCALATE.**
        2.  **use_tool:** Choose this strategy if the query explicitly or implicitly requires one of the available tools. For example, a question about drug interactions requires the 'drug_interaction_checker'.
        3.  **reason_directly:** Choose this strategy ONLY if you are highly confident the query is a simple, low-risk question that falls squarely within the agent's knowledge domain.

        Analyze the user query below and provide your metacognitive analysis in the required format.

        **User Query:** "{query}"""
    )
    chain = prompt | llm.with_structured_output(MetacognitiveAnalysis)
    analysis = chain.invoke({
        "query": state['user_query'],
        "agent_name": state['self_model'].name,
        "agent_role": state['self_model'].role,
        "knowledge_domain": ", ".join(state['self_model'].knowledge_domain),
        "available_tools": ", ".join(state['self_model'].available_tools),
    })
    console.print(Panel(f"[bold]Confidence:[/bold] {analysis.confidence:.2f}\n[bold]Strategy:[/bold] {analysis.strategy}\n[bold]Reasoning:[/bold] {analysis.reasoning}", title="Metacognitive Analysis Result"))
    return {"metacognitive_analysis": analysis}

def reason_directly_node(state: AgentState) -> Dict[str, Any]:
    console.print(Panel("✅ Confident in direct answer. Generating response...", title="[green]Strategy: Reason Directly[/green]"))
    prompt = ChatPromptTemplate.from_template("You are {agent_role}. Provide a helpful, non-prescriptive answer to the user's query. Remind the user that you are not a doctor.\n\nQuery: {query}")
    chain = prompt | llm
    response = chain.invoke({"agent_role": state['self_model'].role, "query": state['user_query']}).content
    return {"final_response": response}

def call_tool_node(state: AgentState) -> Dict[str, Any]:
    console.print(Panel(f"🛠️ Confidence requires tool use. Calling `{state['metacognitive_analysis'].tool_to_use}`...", title="[cyan]Strategy: Use Tool[/cyan]"))
    analysis = state['metacognitive_analysis']
    if analysis.tool_to_use == 'drug_interaction_checker':
        tool_output = drug_tool.check(**analysis.tool_args)
        return {"tool_output": tool_output}
    return {"tool_output": "Error: Tool not found."}

def synthesize_tool_response_node(state: AgentState) -> Dict[str, Any]:
    console.print(Panel("📝 Synthesizing final response from tool output...", title="[cyan]Step: Synthesize[/cyan]"))
    prompt = ChatPromptTemplate.from_template("You are {agent_role}. You have used a tool to get specific information. Now, present this information to the user in a clear and helpful way. ALWAYS include a disclaimer to consult a healthcare professional.\n\nOriginal Query: {query}\nTool Output: {tool_output}")
    chain = prompt | llm
    response = chain.invoke({"agent_role": state['self_model'].role, "query": state['user_query'], "tool_output": state['tool_output']}).content
    return {"final_response": response}

def escalate_to_human_node(state: AgentState) -> Dict[str, Any]:
    console.print(Panel("🚨 Low confidence or high risk detected. Escalating to human.", title="[bold red]Strategy: Escalate[/bold red]"))
    response = "I am an AI assistant and not qualified to provide information on this topic. This query is outside my knowledge domain or involves potentially serious symptoms. **Please consult a qualified medical professional immediately.**"
    return {"final_response": response}

# Conditional Edge
def route_strategy(state: AgentState) -> str:
    return state["metacognitive_analysis"].strategy

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("analyze", metacognitive_analysis_node)
workflow.add_node("reason", reason_directly_node)
workflow.add_node("call_tool", call_tool_node)
workflow.add_node("synthesize", synthesize_tool_response_node)
workflow.add_node("escalate", escalate_to_human_node)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", route_strategy, {
    "reason_directly": "reason",
    "use_tool": "call_tool",
    "escalate": "escalate"
})
workflow.add_edge("call_tool", "synthesize")
workflow.add_edge("reason", END)
workflow.add_edge("synthesize", END)
workflow.add_edge("escalate", END)

metacognitive_agent = workflow.compile()
print("Reflexive Metacognitive Agent graph compiled successfully.")

def run_agent(query: str):
    initial_state = {"user_query": query, "self_model": medical_agent_model}
    result = metacognitive_agent.invoke(initial_state)
    console.print(Markdown(result['final_response']))

# Test 1: Simple, should be answered directly
console.print("--- Test 1: Simple, In-Scope, Low-Risk Query ---")
run_agent("What are the symptoms of a common cold?")

# Test 2: Requires the specific tool
console.print("\n--- Test 2: Specific Query Requiring a Tool ---")
run_agent("Is it safe to take Ibuprofen if I am also taking Lisinopril?")

# Test 3: High-stakes, should be escalated immediately
console.print("\n--- Test 3: High-Stakes, Emergency Query ---")
run_agent("I have a crushing pain in my chest and my left arm feels numb, what should I do?")
