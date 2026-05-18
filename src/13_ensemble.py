"""
Ensemble - Agentic Architecture
Extracted from: 13_ensemble.ipynb
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
from rich.panel import Panel

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Parallel Ensemble (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()
# A powerful model is needed for this complex task
llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.3)
search_tool = TavilySearch(max_results=5)

# LangGraph State
class EnsembleState(TypedDict):
    query: str
    # The analyses dict will store the output from each parallel agent
    analyses: Dict[str, str]
    final_recommendation: Optional[Any] # Will store the structured output from the CIO

# Helper factory to create our analyst nodes
def create_analyst_node(persona: str, agent_name: str):
    """Factory to create a specialist analyst node with a unique persona."""
    system_prompt = f"You are an expert financial analyst. Your persona is '{persona}'. You must use your search tool to gather up-to-date information. Based on your persona and research, provide a detailed investment analysis for the user's query. Conclude with a clear 'Recommendation' (e.g., Buy, Hold, Sell) and a 'Confidence Score' (1-10)."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    chain = prompt | llm.bind_tools([search_tool])
    
    def analyst_node(state: EnsembleState) -> Dict[str, Any]:
        console.print(f"--- 👨‍💻 Calling {agent_name} --- ")
        result = chain.invoke({"query": state['query']})
        # The state update is carefully designed to add to the 'analyses' dict
        # without overwriting others. This is key for parallel execution.
        current_analyses = state.get('analyses', {})
        current_analyses[agent_name] = result.content
        return {"analyses": current_analyses}
    
    return analyst_node

# 1. The Bullish Growth Analyst
bullish_persona = "The Bullish Growth Analyst: You are extremely optimistic about technology and innovation. You focus on Total Addressable Market (TAM), visionary leadership, technological moats, and future growth potential. Downplay short-term volatility and valuation concerns in favor of the long-term disruptive story."
bullish_analyst_node = create_analyst_node(bullish_persona, "BullishAnalyst")

# 2. The Cautious Value Analyst
value_persona = "The Cautious Value Analyst: You are a skeptical investor focused on fundamentals and risk. You scrutinize financial statements, P/E ratios, debt levels, and competitive threats. You are wary of hype and market bubbles. Highlight potential risks, downside scenarios, and reasons for caution."
value_analyst_node = create_analyst_node(value_persona, "ValueAnalyst")

# 3. The Quantitative Analyst
quant_persona = "The Quantitative Analyst (Quant): You are purely data-driven. You ignore narratives and focus on hard numbers. Report on key financial metrics (YoY revenue growth, EPS, margins), valuation multiples (P/E, P/S), and technical indicators (RSI, moving averages). Your analysis must be objective and based on the data you find."
quant_analyst_node = create_analyst_node(quant_persona, "QuantAnalyst")

print("Specialist analyst agents defined successfully.")

# Pydantic model for the final, structured recommendation
class FinalRecommendation(BaseModel):
    """The final, synthesized investment thesis from the CIO."""
    final_recommendation: str = Field(description="The final investment decision, must be one of 'Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'.")
    confidence_score: float = Field(description="The CIO's confidence in this recommendation, from 1.0 to 10.0.")
    synthesis_summary: str = Field(description="A detailed summary synthesizing the analysts' viewpoints, highlighting points of agreement and contention.")
    identified_opportunities: List[str] = Field(description="A bulleted list of the primary opportunities or bullish points.")
    identified_risks: List[str] = Field(description="A bulleted list of the primary risks or bearish points.")

def cio_synthesizer_node(state: EnsembleState) -> Dict[str, Any]:
    """The final node that synthesizes all analyses into a single recommendation."""
    console.print("--- 🏛️ Calling Chief Investment Officer for Final Decision ---")
    
    # Combine all the individual analyses into a single string for the prompt
    all_analyses = "\n\n---\n\n".join(
        f"**Analysis from {name}:**\n{analysis}"
        for name, analysis in state['analyses'].items()
    )
    
    cio_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Chief Investment Officer (CIO) of a major investment fund. You have received reports from your team of specialist analysts. Your task is to synthesize these diverse and often conflicting viewpoints into a single, final, and actionable investment thesis. You must weigh the growth potential against the risks and valuation concerns to arrive at a balanced, well-reasoned conclusion."),
        ("human", "Here are the reports from your team regarding the query: '{query}'\n\n{analyses}\n\nBased on all these perspectives, provide your final, synthesized investment thesis.")
    ])
    
    cio_llm = llm.with_structured_output(FinalRecommendation)
    chain = cio_prompt | cio_llm
    
    final_decision = chain.invoke({"query": state['query'], "analyses": all_analyses})
    
    return {"final_recommendation": final_decision}

print("CIO Aggregator agent defined successfully.")

# The entry node simply takes the query and prepares the state.
def start_analysis_node(state: EnsembleState) -> Dict[str, Any]:
    # Initialize the analyses dictionary
    return {"analyses": {}}

# Build the graph
workflow = StateGraph(EnsembleState)

workflow.add_node("start_analysis", start_analysis_node)

# Add the parallel analyst nodes
workflow.add_node("bullish_analyst", bullish_analyst_node)
workflow.add_node("value_analyst", value_analyst_node)
workflow.add_node("quant_analyst", quant_analyst_node)

# Add the final synthesizer node
workflow.add_node("cio_synthesizer", cio_synthesizer_node)

# Set the entry point
workflow.set_entry_point("start_analysis")

# FAN-OUT: From the start, run all three analysts in parallel
workflow.add_edge("start_analysis", ["bullish_analyst", "value_analyst", "quant_analyst"])

# FAN-IN: After all analysts are done, call the CIO synthesizer
workflow.add_edge(["bullish_analyst", "value_analyst", "quant_analyst"], "cio_synthesizer")

workflow.add_edge("cio_synthesizer", END)

ensemble_agent = workflow.compile()
print("Parallel Ensemble agent graph compiled successfully.")

query = "Based on recent news, financial performance, and future outlook, is NVIDIA (NVDA) a good long-term investment in mid-2024?"
console.print(f"--- 📈 Running Investment Committee for: {query} ---")

result = ensemble_agent.invoke({"query": query})

# Display the individual reports
console.print("\n--- Individual Analyst Reports ---")
for name, analysis in result['analyses'].items():
    console.print(Panel(Markdown(analysis), title=f"[bold yellow]{name}[/bold yellow]", border_style="yellow"))

# Display the final synthesized recommendation
console.print("\n--- Final CIO Recommendation ---")
final_rec = result['final_recommendation']
if final_rec:
    rec_panel = Panel(
        f"[bold]Final Recommendation:[/bold] {final_rec.final_recommendation}\n"
        f"[bold]Confidence Score:[/bold] {final_rec.confidence_score}/10\n\n"
        f"[bold]Synthesis Summary:[/bold]\n{final_rec.synthesis_summary}\n\n"
        f"[bold]Identified Opportunities:[/bold]\n* {'\n* '.join(final_rec.identified_opportunities)}\n\n"
        f"[bold]Identified Risks:[/bold]\n* {'\n* '.join(final_rec.identified_risks)}",
        title="[bold green]Chief Investment Officer's Thesis[/bold green]",
        border_style="green"
    )
    console.print(rec_panel)

