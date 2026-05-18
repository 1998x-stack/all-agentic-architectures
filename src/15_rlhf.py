"""
RLHF - Agentic Architecture
Extracted from: 15_RLHF.ipynb
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
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Self-Improvement Loop (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()
llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.4)

# --- Pydantic Models for Structured Data ---
class MarketingEmail(BaseModel):
    """Represents a marketing email draft."""
    subject: str = Field(description="A catchy and concise subject line for the email.")
    body: str = Field(description="The full body text of the email, written in markdown.")

class Critique(BaseModel):
    """A structured critique of the marketing email draft."""
    score: int = Field(description="Overall quality score from 1 (poor) to 10 (excellent).")
    feedback_points: List[str] = Field(description="A bulleted list of specific, actionable feedback points for improvement.")
    is_approved: bool = Field(description="A boolean indicating if the draft is approved (score >= 8). This is redundant with the score but useful for routing.")

# --- 1. The Generator: Junior Copywriter ---
def get_generator_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a junior marketing copywriter. Your task is to write a first draft of a marketing email based on the user's request. Be creative, but focus on getting the core message across."),
        ("human", "Write a marketing email about the following topic:\n\n{request}")
    ])
    return prompt | llm.with_structured_output(MarketingEmail)

# --- 2. The Critic: Senior Editor ---
def get_critic_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior marketing editor and brand manager. Your job is to critique an email draft written by a junior copywriter. 
        Evaluate the draft against the following criteria:
        1.  **Catchy Subject:** Is the subject line engaging and likely to get opened?
        2.  **Clarity & Persuasiveness:** Is the body text clear, compelling, and persuasive?
        3.  **Strong Call-to-Action (CTA):** Is there a clear, single action for the user to take?
        4.  **Brand Voice:** Is the tone professional yet approachable?
        Provide a score from 1-10. A score of 8 or higher means the draft is approved for sending. Provide specific, actionable feedback to help the writer improve."""
        ),
        ("human", "Please critique the following email draft:\n\n**Subject:** {subject}\n\n**Body:**\n{body}")
    ])
    return prompt | llm.with_structured_output(Critique)

# --- 3. The Reviser (Generator in 'Revise' Mode) ---
def get_reviser_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the junior marketing copywriter who wrote the original draft. You have just received feedback from your senior editor. Your task is to carefully revise your draft to address every single point of feedback. Produce a new, improved version of the email."),
        ("human", "Original Request: {request}\n\nHere is your original draft:\n**Subject:** {original_subject}\n**Body:**\n{original_body}\n\nHere is the feedback from your editor:\n{feedback}\n\nPlease provide the revised email.")
    ])
    return prompt | llm.with_structured_output(MarketingEmail)

print("Generator and Critic components defined successfully.")

# LangGraph State
class AgentState(TypedDict):
    user_request: str
    draft_email: Optional[MarketingEmail]
    critique: Optional[Critique]
    revision_number: int

# Graph Nodes
def generate_node(state: AgentState) -> Dict[str, Any]:
    console.print(Panel("📝 Junior Copywriter is generating the initial draft.", title="[yellow]Step: Generate[/yellow]", border_style="yellow"))
    chain = get_generator_chain()
    draft = chain.invoke({"request": state['user_request']})
    console.print(Panel(f"[bold]Subject:[/bold] {draft.subject}\n\n{draft.body}", title="Draft 1"))
    return {"draft_email": draft, "revision_number": 1}

def critique_node(state: AgentState) -> Dict[str, Any]:
    title = f"[yellow]Step: Critique (Revision #{state['revision_number']})[/yellow]"
    console.print(Panel(f"🧐 Senior Editor is critiquing draft #{state['revision_number']}.", title=title, border_style="yellow"))
    chain = get_critic_chain()
    critique_result = chain.invoke(state['draft_email'].dict())
    feedback_text = "\n- ".join(critique_result.feedback_points)
    console.print(Panel(f"[bold]Score:[/bold] {critique_result.score}/10\n[bold]Feedback:[/bold]\n- {feedback_text}", title="Critique Result"))
    return {"critique": critique_result}

def revise_node(state: AgentState) -> Dict[str, Any]:
    console.print(Panel("✍️ Junior Copywriter is revising the draft based on feedback.", title="[yellow]Step: Revise[/yellow]", border_style="yellow"))
    chain = get_reviser_chain()
    feedback_str = "\n- ".join(state['critique'].feedback_points)
    revised_draft = chain.invoke({
        "request": state['user_request'],
        "original_subject": state['draft_email'].subject,
        "original_body": state['draft_email'].body,
        "feedback": feedback_str,
    })
    console.print(Panel(f"[bold]Subject:[/bold] {revised_draft.subject}\n\n{revised_draft.body}", title=f"Draft {state['revision_number'] + 1}"))
    return {"draft_email": revised_draft, "revision_number": state['revision_number'] + 1}

# Conditional Edge
def should_continue(state: AgentState) -> str:
    console.print(Panel("⚖️ Decision Point: Does the draft meet quality standards?", title="[yellow]Step: Decide[/yellow]", border_style="yellow"))
    if state['critique'].is_approved:
        console.print("[green]Conclusion: Critique APPROVED! Finishing process.[/green]")
        return "end"
    if state['revision_number'] >= 3: # Set a max revision limit
        console.print("[red]Conclusion: Max revisions reached. Finishing with last draft.[/red]")
        return "end"
    else:
        console.print("[yellow]Conclusion: Critique requires revision. Looping back.[/yellow]")
        return "continue"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("generate", generate_node)
workflow.add_node("critique", critique_node)
workflow.add_node("revise", revise_node)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "critique")
workflow.add_conditional_edges(
    "critique",
    should_continue,
    {"continue": "revise", "end": END}
)
workflow.add_edge("revise", "critique")

self_refine_agent = workflow.compile()
print("Self-Refinement agent graph compiled successfully.")

def run_agent(request: str):
    initial_state = {"user_request": request}
    # stream() allows us to see the intermediate steps
    final_state = None
    for step in self_refine_agent.stream(initial_state):
        # The final state is the one just before END is called
        if END not in step:
            final_state = list(step.values())[0]
    return final_state

request = "Write a marketing email announcing our new revolutionary AI-powered data analytics platform, 'InsightSphere'."
console.print(f"--- 🚀 Kicking off the Self-Refinement Process ---")
final_result = run_agent(request)

# Display the final, approved result
console.print("\n--- Final Approved Email ---")
final_email = final_result['draft_email']
final_critique = final_result['critique']
email_panel = Panel(
    f"[bold]Subject:[/bold] {final_email.subject}\n\n---\n\n{final_email.body}",
    title="[bold green]Approved Email[/bold green]",
    subtitle=f"[green]Final Score: {final_critique.score}/10[/green]",
    border_style="green"
)
console.print(email_panel)

class GoldStandardMemory:
    """A simple in-memory store for high-quality examples."""
    def __init__(self):
        self.examples: List[MarketingEmail] = []
        
    def add_example(self, email: MarketingEmail):
        self.examples.append(email)
        
    def get_formatted_examples(self) -> str:
        if not self.examples:
            return "No examples available yet."
        formatted = "\n\n---\n\n".join([
            f"Example Subject: {ex.subject}\nExample Body:\n{ex.body}"
            for ex in self.examples
        ])
        return formatted

# Instantiate our persistent memory
gold_standard_memory = GoldStandardMemory()

# New generator node that uses the memory
def generate_node_with_memory(state: AgentState) -> Dict[str, Any]:
    title = "[yellow]Step: Generate[/yellow]"
    console.print(Panel("📝 Junior Copywriter is generating the initial draft (Informed by Past Successes).", title=title, border_style="yellow"))
    examples = gold_standard_memory.get_formatted_examples()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a junior marketing copywriter. Your task is to write a first draft of a marketing email based on the user's request. You should learn from the style and quality of past successful examples."),
        ("human", "Here are some examples of high-quality emails that were approved by your editor:\n\n{examples}\n\nNow, write a marketing email about the following topic:\n\n{request}")
    ])
    chain = prompt | llm.with_structured_output(MarketingEmail)
    draft = chain.invoke({"request": state['user_request'], "examples": examples})
    console.print(Panel(f"[bold]Subject:[/bold] {draft.subject}\n\n{draft.body}", title=f"Draft {state.get('revision_number', 1)}"))
    return {"draft_email": draft, "revision_number": 1}

# Build the new graph with the memory-enabled generator
workflow_with_memory = StateGraph(AgentState)
workflow_with_memory.add_node("generate", generate_node_with_memory)
workflow_with_memory.add_node("critique", critique_node)
workflow_with_memory.add_node("revise", revise_node)
workflow_with_memory.set_entry_point("generate")
workflow_with_memory.add_edge("generate", "critique")
workflow_with_memory.add_conditional_edges("critique", should_continue, {"continue": "revise", "end": END})
workflow_with_memory.add_edge("revise", "critique")
self_improving_agent = workflow_with_memory.compile()
print("Persistent memory components defined successfully.")

# --- DEMONSTRATION OF LONG-TERM IMPROVEMENT ---

# 1. Save our previously approved email to the memory
console.print(Panel("The high-quality, editor-approved email for 'InsightSphere' has been saved. It will now be used as a reference for future generations.", title="[bold]🏆 Saving approved email to Gold Standard Memory[/bold]", border_style="magenta"))
gold_standard_memory.add_example(final_result['draft_email'])

# 2. Run the agent again on a NEW task
new_request = "Write a promotional email for our new AI-powered CRM called 'Visionary'."
console.print("\n--- 🚀 Kicking off the Self-Refinement Process with Memory ---")
new_final_result = run_agent(new_request)

# 3. Display the new result. The key thing to notice is if it gets approved faster.
console.print("\n--- Final Approved Email (Generated with Memory) ---")
new_final_email = new_final_result['draft_email']
new_critique = new_final_result['critique']
email_panel_2 = Panel(
    f"[bold]Subject:[/bold] {new_final_email.subject}\n\n---\n\n{new_final_email.body}",
    title="[bold green]Approved Email[/bold green]",
    subtitle=f"[green]Final Score: {new_critique.score}/10[/green]",
    border_style="green"
)
console.print(email_panel_2)
