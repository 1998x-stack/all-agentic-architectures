"""
Dry Run - Agentic Architecture
Extracted from: 14_dry_run.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv

import os
import datetime
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
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Dry-Run Harness (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()

# Structured model for the agent's proposed post
class SocialMediaPost(BaseModel):
    content: str = Field(description="The full text content of the social media post.")
    hashtags: List[str] = Field(description="A list of relevant hashtags, without the '#'.")

# The key component: A tool with a dry_run flag
class SocialMediaAPI:
    """A mock social media API that supports a dry-run mode."""
    
    def publish_post(self, post: SocialMediaPost, dry_run: bool = True) -> Dict[str, Any]:
        """Publishes a post to the social media feed."""
        timestamp = datetime.datetime.now().isoformat()
        hashtags_str = ' '.join([f'#{h}' for h in post.hashtags])
        full_post_text = f"{post.content}\n\n{hashtags_str}"
        
        if dry_run:
            # In dry-run mode, we don't execute, we just return the plan and logs
            log_message = f"[DRY RUN] At {timestamp}, would publish the following post:\n--- PREVIEW ---\n{full_post_text}\n--- END PREVIEW ---"
            console.print(Panel(log_message, title="[yellow]Dry Run Log[/yellow]", border_style="yellow"))
            return {"status": "DRY_RUN_SUCCESS", "log": log_message, "proposed_post": full_post_text}
        else:
            # In live mode, we execute the action
            log_message = f"[LIVE] At {timestamp}, successfully published post!"
            console.print(Panel(log_message, title="[green]Live Execution Log[/green]", border_style="green"))
            # Here you would have the actual API call, e.g., twitter_client.create_tweet(...)
            return {"status": "LIVE_SUCCESS", "log": log_message, "post_id": f"post_{hash(full_post_text)}"}

social_media_tool = SocialMediaAPI()
print("Dry-run capable SocialMediaAPI tool defined successfully.")

llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5)

# LangGraph State
class AgentState(TypedDict):
    user_request: str
    proposed_post: Optional[SocialMediaPost]
    dry_run_log: Optional[str]
    review_decision: Optional[str] # 'approve' or 'reject'
    final_status: str

# Graph Nodes
def propose_post_node(state: AgentState) -> Dict[str, Any]:
    """The creative agent that drafts the social media post."""
    console.print("--- 📝 Social Media Agent Drafting Post ---")
    prompt = ChatPromptTemplate.from_template(
        "You are a creative and engaging social media manager for a major AI company. Based on the user's request, draft a compelling social media post, including relevant hashtags.\n\nRequest: {request}"
    )
    post_generator_llm = llm.with_structured_output(SocialMediaPost)
    chain = prompt | post_generator_llm
    proposed_post = chain.invoke({"request": state['user_request']})
    return {"proposed_post": proposed_post}

def dry_run_review_node(state: AgentState) -> Dict[str, Any]:
    """Performs the dry run and prompts for human review."""
    console.print("--- 🧐 Performing Dry Run & Awaiting Human Review ---")
    dry_run_result = social_media_tool.publish_post(state['proposed_post'], dry_run=True)
    
    # Present the plan for review
    review_panel = Panel(
        f"[bold]Proposed Post:[/bold]\n{dry_run_result['proposed_post']}",
        title="[bold yellow]Human-in-the-Loop: Review Required[/bold yellow]",
        border_style="yellow"
    )
    console.print(review_panel)
    
    # Get human approval
    decision = ""
    while decision.lower() not in ["approve", "reject"]:
        decision = console.input("Type 'approve' to publish or 'reject' to cancel: ")
        
    return {"dry_run_log": dry_run_result['log'], "review_decision": decision.lower()}

def execute_live_post_node(state: AgentState) -> Dict[str, Any]:
    """Executes the live post after approval."""
    console.print("--- ✅ Post Approved, Executing Live ---")
    live_result = social_media_tool.publish_post(state['proposed_post'], dry_run=False)
    return {"final_status": f"Post successfully published! ID: {live_result.get('post_id')}"}

def post_rejected_node(state: AgentState) -> Dict[str, Any]:
    """Handles the case where the post is rejected."""
    console.print("--- ❌ Post Rejected by Human Reviewer ---")
    return {"final_status": "Action was rejected by the reviewer and not executed."}

# Conditional Edge
def route_after_review(state: AgentState) -> str:
    """Routes to execution or rejection based on the human review."""
    return "execute_live" if state["review_decision"] == "approve" else "reject"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("propose_post", propose_post_node)
workflow.add_node("dry_run_review", dry_run_review_node)
workflow.add_node("execute_live", execute_live_post_node)
workflow.add_node("reject", post_rejected_node)

workflow.set_entry_point("propose_post")
workflow.add_edge("propose_post", "dry_run_review")
workflow.add_conditional_edges("dry_run_review", route_after_review, {"execute_live": "execute_live", "reject": "reject"})
workflow.add_edge("execute_live", END)
workflow.add_edge("reject", END)

dry_run_agent = workflow.compile()
print("Dry-Run Harness agent graph compiled successfully.")

def run_agent_with_harness(request: str):
    initial_state = {"user_request": request}
    # Note: You will be prompted to type in the console below the cell.
    result = dry_run_agent.invoke(initial_state)
    console.print(f"\n[bold]Final Status:[/bold] {result['final_status']}")

# Test 1: A safe post that we will approve.
console.print("--- ✅ Test 1: Safe Post (Approve) ---")
run_agent_with_harness("Draft a positive launch announcement for our new AI model, 'Nebula'.")

# Test 2: A risky post that we will reject.
console.print("\n--- ❌ Test 2: Risky Post (Reject) ---")
run_agent_with_harness("Draft a post that emphasizes how much better our new 'Nebula' model is than the competition.")
