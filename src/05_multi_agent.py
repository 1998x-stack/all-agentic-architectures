"""
Multi-Agent - Agentic Architecture
Extracted from: 05_multi_agent.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv langchain-tavily

import os
from typing import List, Annotated, TypedDict, Optional
from dotenv import load_dotenv

# LangChain components
from langchain_nebius import ChatNebius
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Multi-Agent (Nebius)"

for key in ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} not found. Please create a .env file and set it.")

print("Environment variables loaded and tracing is set up.")

console = Console()

# Define the shared state for both agents
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the tool and LLM
search_tool = TavilySearch(max_results=3, name="web_search")
llm = ChatNebius(model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0)
llm_with_tools = llm.bind_tools([search_tool])

# Define the monolithic agent node
def monolithic_agent_node(state: AgentState):
    console.print("--- MONOLITHIC AGENT: Thinking... ---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode([search_tool])

# Build the ReAct graph for the monolithic agent
mono_graph_builder = StateGraph(AgentState)
mono_graph_builder.add_node("agent", monolithic_agent_node)
mono_graph_builder.add_node("tools", tool_node)
mono_graph_builder.set_entry_point("agent")

def tools_condition_with_end(state):
    result = tools_condition(state)
    if isinstance(result, str):
        # Older versions return just "tools" or "agent"
        return {result: "tools", "__default__": END}
    elif isinstance(result, dict):
        # Newer versions return a mapping
        result["__default__"] = END
        return result
    else:
        raise TypeError(f"Unexpected type from tools_condition: {type(result)}")

mono_graph_builder.add_conditional_edges("agent", tools_condition_with_end)
mono_graph_builder.add_edge("tools", "agent")

monolithic_agent_app = mono_graph_builder.compile()

print("Monolithic 'generalist' agent compiled successfully.")

company = "NVIDIA (NVDA)"
monolithic_query = f"Create a brief but comprehensive market analysis report for {company}. The report should include three sections: 1. A summary of recent news and market sentiment. 2. A basic technical analysis of the stock's price trend. 3. A look at the company's recent financial performance."

console.print(f"[bold yellow]Testing MONOLITHIC agent on a multi-faceted task:[/bold yellow]\n'{monolithic_query}'\n")

final_mono_output = monolithic_agent_app.invoke({
    "messages": [
        SystemMessage(content="You are a single, expert financial analyst. You must create a comprehensive report covering all aspects of the user's request."),
        HumanMessage(content=monolithic_query)
    ]
})

console.print("\n--- [bold red]Final Report from Monolithic Agent[/bold red] ---")
console.print(Markdown(final_mono_output['messages'][-1].content))

# The state for our multi-agent system will hold the outputs of each specialist
class MultiAgentState(TypedDict):
    user_request: str
    news_report: Optional[str]
    technical_report: Optional[str]
    financial_report: Optional[str]
    final_report: Optional[str]

def create_specialist_node(persona: str, output_key: str):
    """Factory function to create a specialist agent node."""
    system_prompt = persona + "\n\nYou have access to a web search tool. Your output MUST be a concise report section, formatted in markdown, focusing only on your area of expertise."

    # ✅ Build a ChatPromptTemplate instead of a plain list
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_request}")
    ])

    agent = prompt_template | llm_with_tools

    def specialist_node(state: MultiAgentState):
        console.print(f"--- CALLING {output_key.replace('_report','').upper()} ANALYST ---")
        result = agent.invoke({"user_request": state["user_request"]})
        content = result.content if result.content else f"No direct content, tool calls: {result.tool_calls}"
        return {output_key: content}

    return specialist_node


# Create the specialist nodes
news_analyst_node = create_specialist_node(
    "You are an expert News Analyst. Your specialty is scouring the web for the latest news, articles, and social media sentiment about a company.",
    "news_report"
)
technical_analyst_node = create_specialist_node(
    "You are an expert Technical Analyst. You specialize in analyzing stock price charts, trends, and technical indicators.",
    "technical_report"
)
financial_analyst_node = create_specialist_node(
    "You are an expert Financial Analyst. You specialize in interpreting financial statements and performance metrics.",
    "financial_report"
)

def report_writer_node(state: MultiAgentState):
    """The manager agent that synthesizes the specialist reports."""
    console.print("--- CALLING REPORT WRITER ---")
    prompt = f"""You are an expert financial editor. Your task is to combine the following specialist reports into a single, professional, and cohesive market analysis report. Add a brief introductory and concluding paragraph.
    
    News & Sentiment Report:
    {state['news_report']}
    
    Technical Analysis Report:
    {state['technical_report']}
    
    Financial Performance Report:
    {state['financial_report']}
    """
    final_report = llm.invoke(prompt).content
    return {"final_report": final_report}

print("Specialist agent nodes and Report Writer node defined.")

multi_agent_graph_builder = StateGraph(MultiAgentState)

# Add all the nodes
multi_agent_graph_builder.add_node("news_analyst", news_analyst_node)
multi_agent_graph_builder.add_node("technical_analyst", technical_analyst_node)
multi_agent_graph_builder.add_node("financial_analyst", financial_analyst_node)
multi_agent_graph_builder.add_node("report_writer", report_writer_node)

# Define the workflow sequence
multi_agent_graph_builder.set_entry_point("news_analyst")
multi_agent_graph_builder.add_edge("news_analyst", "technical_analyst")
multi_agent_graph_builder.add_edge("technical_analyst", "financial_analyst")
multi_agent_graph_builder.add_edge("financial_analyst", "report_writer")
multi_agent_graph_builder.add_edge("report_writer", END)

multi_agent_app = multi_agent_graph_builder.compile()
print("Multi-agent specialist team compiled successfully.")

multi_agent_query = f"Create a brief but comprehensive market analysis report for {company}."
initial_multi_agent_input = {"user_request": multi_agent_query}

console.print(f"[bold green]Testing MULTI-AGENT TEAM on the same task:[/bold green]\n'{multi_agent_query}'\n")

final_multi_agent_output = multi_agent_app.invoke(initial_multi_agent_input)

console.print("\n--- [bold green]Final Report from Multi-Agent Team[/bold green] ---")
console.print(Markdown(final_multi_agent_output['final_report']))

class ReportEvaluation(BaseModel):
    """Schema for evaluating a financial report."""
    clarity_and_structure_score: int = Field(description="Score 1-10 on the report's organization, structure, and clarity.")
    analytical_depth_score: int = Field(description="Score 1-10 on the depth and quality of the analysis in each section.")
    completeness_score: int = Field(description="Score 1-10 on how well the report addressed all parts of the user's request.")
    justification: str = Field(description="A brief justification for the scores.")

judge_llm = llm.with_structured_output(ReportEvaluation)

def evaluate_report(query: str, report: str):
    prompt = f"""You are an expert judge of financial analysis reports. Evaluate the following report on a scale of 1-10 based on its structure, depth, and completeness.
    
    **Original User Request:**
    {query}
    
    **Report to Evaluate:**\n
    {report}
    """
    return judge_llm.invoke(prompt)

console.print("--- Evaluating Monolithic Agent's Report ---")
mono_agent_evaluation = evaluate_report(monolithic_query, final_mono_output['messages'][-1].content)
console.print(mono_agent_evaluation.model_dump())

console.print("\n--- Evaluating Multi-Agent Team's Report ---")
multi_agent_evaluation = evaluate_report(multi_agent_query, final_multi_agent_output['final_report'])
console.print(multi_agent_evaluation.model_dump())
