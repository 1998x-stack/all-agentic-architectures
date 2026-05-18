"""
Blackboard - Agentic Architecture
Extracted from: 07_blackboard.ipynb
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

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Blackboard (Nebius)"

for key in ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} not found. Please create a .env file and set it.")

print("Environment variables loaded and tracing is set up.")

console = Console()
# Using a more capable model to handle complex instructions better
llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0)
search_tool = TavilySearch(max_results=2)

# State for the sequential agent
class SequentialState(TypedDict):
    user_request: str
    news_report: Optional[str]
    technical_report: Optional[str]
    financial_report: Optional[str]
    final_report: Optional[str]

# --- CORRECTED SPECIALIST NODES FOR SEQUENTIAL AGENT ---
# The key change is that each agent now gets context from previous steps, not just the original request.

def news_analyst_node_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING NEWS ANALYST ---")
    prompt = f"Your task is to act as an expert News Analyst. Find the latest major news about the topic in the user's request and provide a concise summary.\n\nUser Request: {state['user_request']}"
    agent = llm.bind_tools([search_tool])
    result = agent.invoke(prompt)
    return {"news_report": result.content}

def technical_analyst_node_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING TECHNICAL ANALYST ---")
    # This agent now uses the news report as context.
    prompt = f"Your task is to act as an expert Technical Analyst. Based on the following news report, conduct a technical analysis of the company's stock.\n\nNews Report:\n{state['news_report']}"
    agent = llm.bind_tools([search_tool])
    result = agent.invoke(prompt)
    return {"technical_report": result.content}

def financial_analyst_node_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING FINANCIAL ANALYST ---")
    # This agent also uses the news report as context.
    prompt = f"Your task is to act as an expert Financial Analyst. Based on the following news report, analyze the company's recent financial performance.\n\nNews Report:\n{state['news_report']}"
    agent = llm.bind_tools([search_tool])
    result = agent.invoke(prompt)
    return {"financial_report": result.content}


def report_writer_node_seq(state: SequentialState):
    console.print("--- (Sequential) CALLING REPORT WRITER ---")
    prompt = f"""You are an expert report writer. Your task is to synthesize the information from the News, Technical, and Financial analysts into a single, cohesive report that directly answers the user's original request.

User Request: {state['user_request']}

Here are the reports to combine:
---
News Report: {state['news_report']}
---
Technical Report: {state['technical_report']}
---
Financial Report: {state['financial_report']}
"""
    report = llm.invoke(prompt).content
    return {"final_report": report}

# Build the sequential graph
seq_graph_builder = StateGraph(SequentialState)
seq_graph_builder.add_node("news", news_analyst_node_seq)
seq_graph_builder.add_node("tech", technical_analyst_node_seq)
seq_graph_builder.add_node("finance", financial_analyst_node_seq)
seq_graph_builder.add_node("writer", report_writer_node_seq)

# The rigid, hardcoded sequence
seq_graph_builder.set_entry_point("news")
seq_graph_builder.add_edge("news", "tech")
seq_graph_builder.add_edge("tech", "finance")
seq_graph_builder.add_edge("finance", "writer")
seq_graph_builder.add_edge("writer", END)

sequential_app = seq_graph_builder.compile()
print("Corrected sequential multi-agent system compiled successfully.")

dynamic_query = "Find the latest major news about Nvidia. Based on the sentiment of that news, conduct either a technical analysis (if the news is neutral or positive) or a financial analysis of their recent performance (if the news is negative)."

console.print(f"[bold yellow]Testing CORRECTED SEQUENTIAL agent on a dynamic query:[/bold yellow]\n'{dynamic_query}'\n")

# Run the graph
final_seq_output = sequential_app.invoke({"user_request": dynamic_query})

console.print("\n--- [bold red]Final Report from Sequential Agent[/bold red] ---")
console.print(Markdown(final_seq_output['final_report']))

# The Blackboard State holds all information
class BlackboardState(TypedDict):
    user_request: str
    # The central blackboard where agents post their findings as strings
    blackboard: List[str]
    # List of available agents for the controller to choose from
    available_agents: List[str]
    # The controller's next decision
    next_agent: Optional[str]

# Pydantic model for the Controller's decision
# CORRECTION: Added the list of available agents to the field description to guide the LLM's choice.
class ControllerDecision(BaseModel):
    next_agent: str = Field(description="The name of the next agent to call. Must be one of ['News Analyst', 'Technical Analyst', 'Financial Analyst', 'Report Writer'] or 'FINISH'.")
    reasoning: str = Field(description="A brief reason for choosing the next agent.")

# Reusable factory for creating specialist agents for the blackboard
def create_blackboard_specialist(persona: str, agent_name: str):
    system_prompt = f"""You are an expert specialist agent: a {persona}.
Your task is to contribute to a larger goal by performing your specific function.
Read the initial User Request and the current Blackboard for context.
Use your tools to find the required information.
Finally, post your concise markdown report back to the blackboard. Your report should be signed with your name '{agent_name}'.
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Request: {user_request}\n\nBlackboard (previous reports):\n{blackboard_str}")
    ])
    agent = prompt_template | llm.bind_tools([search_tool])

    def specialist_node(state: BlackboardState):
        console.print(f"--- (Blackboard) AGENT '{agent_name}' is working... ---")
        blackboard_str = "\n---\n".join(state["blackboard"])
        result = agent.invoke({"user_request": state["user_request"], "blackboard_str": blackboard_str})
        report = f"**Report from {agent_name}:**\n{result.content}"
        # Append the new report to the list of blackboard entries
        return {"blackboard": state["blackboard"] + [report]}
    return specialist_node

# Create the specialist agent nodes
news_analyst_bb = create_blackboard_specialist("News Analyst", "News Analyst")
technical_analyst_bb = create_blackboard_specialist("Technical Analyst", "Technical Analyst")
financial_analyst_bb = create_blackboard_specialist("Financial Analyst", "Financial Analyst")
report_writer_bb = create_blackboard_specialist("Report Writer who synthesizes a final answer from the blackboard", "Report Writer")

# --- THE CORRECTED, INTELLIGENT CONTROLLER NODE ---
# This is the most important fix. The prompt is now much more sophisticated.
def controller_node(state: BlackboardState):
    console.print("--- CONTROLLER: Analyzing blackboard... ---")

    # Use a structured output LLM to make the decision
    controller_llm = llm.with_structured_output(ControllerDecision)

    blackboard_content = "\n\n".join(state['blackboard'])
    agent_list = state['available_agents']

    # The new prompt is state-aware and goal-oriented.
    prompt = f"""You are the central controller of a multi-agent system. Your job is to analyze the shared blackboard and the original user request to decide which specialist agent should run next.

**Original User Request:**
{state['user_request']}

**Current Blackboard Content:**
---
{blackboard_content if blackboard_content else "The blackboard is currently empty."}
---

**Available Specialist Agents:**
{', '.join(agent_list)}

**Your Task:**
1.  Read the user request and the current blackboard content carefully.
2.  Determine what the *next logical step* is to move closer to a complete answer.
3.  Choose the single best agent to perform that step from the list of available agents.
4.  If the user's request has been fully addressed and a final report has been written, choose 'FINISH'. Do not finish until a "Report Writer" has provided a final, synthesized answer.

Provide your decision in the required format.
"""
    decision_result = controller_llm.invoke(prompt)
    console.print(f"--- CONTROLLER: Decision is to call '{decision_result.next_agent}'. Reason: {decision_result.reasoning} ---")

    # The dictionary returned here updates the 'next_agent' key in the graph's state
    return {"next_agent": decision_result.next_agent}

print("Blackboard components and corrected Controller node defined.")

bb_graph_builder = StateGraph(BlackboardState)

# Add all nodes to the graph
bb_graph_builder.add_node("Controller", controller_node)
bb_graph_builder.add_node("News Analyst", news_analyst_bb)
bb_graph_builder.add_node("Technical Analyst", technical_analyst_bb)
bb_graph_builder.add_node("Financial Analyst", financial_analyst_bb)
bb_graph_builder.add_node("Report Writer", report_writer_bb)

bb_graph_builder.set_entry_point("Controller")

# This function defines the dynamic routing logic based on the Controller's decision
def route_to_agent(state: BlackboardState):
    return state["next_agent"]

# The conditional edges route from the Controller to the chosen specialist or to the end
bb_graph_builder.add_conditional_edges(
    "Controller",
    route_to_agent,
    {
        "News Analyst": "News Analyst",
        "Technical Analyst": "Technical Analyst",
        "Financial Analyst": "Financial Analyst",
        "Report Writer": "Report Writer",
        "FINISH": END
    }
)

# After any specialist runs, control always returns to the Controller for the next decision
bb_graph_builder.add_edge("News Analyst", "Controller")
bb_graph_builder.add_edge("Technical Analyst", "Controller")
bb_graph_builder.add_edge("Financial Analyst", "Controller")
bb_graph_builder.add_edge("Report Writer", "Controller")

blackboard_app = bb_graph_builder.compile()
print("Blackboard system compiled successfully.")

console.print(f"[bold green]Testing BLACKBOARD system on the same dynamic query:[/bold green]\n'{dynamic_query}'\n")

agent_list = ["News Analyst", "Technical Analyst", "Financial Analyst", "Report Writer"]
initial_bb_input = {"user_request": dynamic_query, "blackboard": [], "available_agents": agent_list}

# We use stream to observe the step-by-step process
final_bb_output = None
for chunk in blackboard_app.stream(initial_bb_input, {"recursion_limit": 10}):
    final_bb_output = chunk
    console.print("\n--- [bold purple]Current Blackboard State[/bold purple] ---")
    # Pretty print each report on the blackboard
    for i, report in enumerate(final_bb_output.get('blackboard', [])):
        console.print(f"--- Report {i+1} ---")
        console.print(Markdown(report))
    console.print("\n")

console.print("\n--- [bold green]Final Report from Blackboard System[/bold green] ---")
# The final report is the last item posted to the blackboard by the writer
final_report_content = final_bb_output['blackboard'][-1]
console.print(Markdown(final_report_content))

class ProcessLogicEvaluation(BaseModel):
    """Schema for evaluating an agent's logical process."""
    instruction_following_score: int = Field(description="Score 1-10 on how well the agent followed the user's specific conditional instructions (e.g., the 'either/or' logic).")
    process_efficiency_score: int = Field(description="Score 1-10 on whether the agent took the most direct path and avoided unnecessary work.")
    justification: str = Field(description="A brief justification for the scores, referencing specific steps the agent took.")

# Use a strong model for judging
judge_llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0).with_structured_output(ProcessLogicEvaluation)

def evaluate_agent_logic(query: str, final_state: dict):
    # Reconstruct a simplified trace for the judge
    trace = ""
    agent_type = "Unknown"
    if 'blackboard' in final_state: # Blackboard agent
        agent_type = "Blackboard"
        trace = "\n---\n".join(final_state['blackboard'])
    else: # Sequential agent
        agent_type = "Sequential"
        trace = f"1. News Report Generated: {final_state.get('news_report')}\n---\n2. Technical Report Generated: {final_state.get('technical_report')}\n---\n3. Financial Report Generated: {final_state.get('financial_report')}"

    prompt = f"""You are an expert judge of AI agent processes. Your task is to evaluate an agent's performance based on its generated content trace.

**User's Original Task:**
"{query}"

**Agent's Type:** {agent_type}
**Agent's Generated Content Trace:**
```
{trace}
```

**Evaluation Criteria:**
1.  **Instruction Following:** Did the agent respect the conditional logic in the user's task? (e.g., "either a technical analysis... or a financial analysis"). A high score means it followed the logic perfectly. A low score means it ignored it.
2.  **Process Efficiency:** Did the agent avoid doing unnecessary work? A high score means it only ran the required specialists. A low score means it ran specialists that the user's logic explicitly said to skip.

Based on the trace, provide your evaluation.
"""
    return judge_llm.invoke(prompt)

console.print("--- [bold]Evaluating Sequential Agent's Process[/bold] ---")
seq_agent_evaluation = evaluate_agent_logic(dynamic_query, final_seq_output)
console.print(seq_agent_evaluation.dict())

console.print("\n--- [bold]Evaluating Blackboard System's Process[/bold] ---")
bb_agent_evaluation = evaluate_agent_logic(dynamic_query, final_bb_output)
console.print(bb_agent_evaluation.dict())
