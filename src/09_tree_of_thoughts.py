"""
Tree of Thoughts - Agentic Architecture
Extracted from: 09_tree_of_thoughts.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from collections import defaultdict

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
from rich.tree import Tree

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Tree-of-Thoughts (Nebius)"

# Check for required environment variables
required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()

class PuzzleState(BaseModel):
    "Represents the state of the Wolf, Goat, and Cabbage puzzle."
    # Using sets for unordered collections of items on each bank
    left_bank: set[str] = Field(default_factory=lambda: {"wolf", "goat", "cabbage"})
    right_bank: set[str] = Field(default_factory=set)
    boat_location: str = "left"
    move_description: str = "Initial state."

    def is_valid(self) -> bool:
        """Checks if the current state is valid (no one gets eaten)."""
        # Check left bank
        if self.boat_location == "right":
            if "wolf" in self.left_bank and "goat" in self.left_bank:
                return False
            if "goat" in self.left_bank and "cabbage" in self.left_bank:
                return False
        # Check right bank
        if self.boat_location == "left":
            if "wolf" in self.right_bank and "goat" in self.right_bank:
                return False
            if "goat" in self.right_bank and "cabbage" in self.right_bank:
                return False
        return True

    def is_goal(self) -> bool:
        """Checks if the goal state has been reached."""
        return self.right_bank == {"wolf", "goat", "cabbage"}
    
    def __hash__(self):
        # Make the state hashable to check for visited states
        return hash((frozenset(self.left_bank), frozenset(self.right_bank), self.boat_location))
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

def get_possible_moves(state: PuzzleState) -> list[PuzzleState]:
    """Generates all possible valid next states from the current state."""
    moves = []
    current_bank = state.left_bank if state.boat_location == "left" else state.right_bank
    
    # Option 1: Move one item in the boat
    for item in current_bank:
        new_state = state.model_copy(deep=True)
        if state.boat_location == "left":
            new_state.left_bank.remove(item)
            new_state.right_bank.add(item)
            new_state.boat_location = "right"
            new_state.move_description = f"Move {item} to the right bank."
        else:
            new_state.right_bank.remove(item)
            new_state.left_bank.add(item)
            new_state.boat_location = "left"
            new_state.move_description = f"Move {item} to the left bank."
        if new_state.is_valid():
            moves.append(new_state)
            
    # Option 2: Move the boat empty
    empty_move_state = state.model_copy(deep=True)
    if state.boat_location == "left":
        empty_move_state.boat_location = "right"
        empty_move_state.move_description = "Move the boat empty to the right bank."
    else:
        empty_move_state.boat_location = "left"
        empty_move_state.move_description = "Move the boat empty to the left bank."
    if empty_move_state.is_valid():
        moves.append(empty_move_state)
        
    return moves

print("Puzzle environment defined successfully.")

llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.4)

# Pydantic model for the LLM's choice of move
class MoveChoice(BaseModel):
    best_move_index: int = Field(description="The index of the best move from the provided list of possible moves.")
    reasoning: str = Field(description="Brief reasoning for why this is the most promising move.")

# LangGraph State
class ToTState(TypedDict):
    problem_description: str
    # Each path is a list of PuzzleState objects
    active_paths: List[List[PuzzleState]]
    # We'll store the final solution here
    solution: Optional[List[PuzzleState]]

# Graph Nodes
def initialize_search(state: ToTState) -> Dict[str, Any]:
    """Node to set up the initial state of the search."""
    initial_puzzle_state = PuzzleState()
    return {"active_paths": [[initial_puzzle_state]]}

def expand_paths(state: ToTState) -> Dict[str, Any]:
    """The 'Thought Generator'. Expands each active path with a promising next move."""
    console.print("--- Expanding Paths ---")
    new_paths = []
    choice_llm = llm.with_structured_output(MoveChoice)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert logic puzzle solver. Your goal is to solve the Wolf, Goat, and Cabbage problem. Analyze the current path and choose the most promising next move from the list of options to reach the goal."),
        ("human", "Problem: {problem}\n\nCurrent Path History:\n{path_history}\n\nFrom the final state, choose the best next move from this list:\n{possible_moves}")
    ])
    
    for path in state['active_paths']:
        last_state = path[-1]
        possible_next_states = get_possible_moves(last_state)
        
        if not possible_next_states:
            continue # This path is a dead end
            
        path_history_str = " -> ".join([s.move_description for s in path])
        possible_moves_str = "\n".join([f"{i}: {s.move_description}" for i, s in enumerate(possible_next_states)])
        
        # For simplicity and to show breadth, we can explore multiple moves.
        # A more advanced ToT might use the LLM to pick only the single best one.
        # Here, we'll let all valid moves branch out to demonstrate the tree structure.
        for next_state in possible_next_states:
            new_paths.append(path + [next_state])

    console.print(f"[cyan]Expanded to {len(new_paths)} potential paths.[/cyan]")
    return {"active_paths": new_paths}

def prune_paths(state: ToTState) -> Dict[str, Any]:
    """The 'State Evaluator'. Prunes paths that are invalid or contain cycles."""
    console.print("--- Pruning Paths ---")
    pruned_paths = []
    for path in state['active_paths']:
        # Check for cycles: if the last state has appeared before in the path
        if path[-1] in path[:-1]:
            continue # Found a cycle, prune this path
        
        # The get_possible_moves function already ensures validity, but this is a good place for extra checks.
        pruned_paths.append(path)
        
    console.print(f"[green]Pruned down to {len(pruned_paths)} valid, non-cyclical paths.[/green]")
    return {"active_paths": pruned_paths}

# Conditional Node
def check_for_solution(state: ToTState) -> str:
    """Checks if any path has reached the goal and routes execution."""
    for path in state['active_paths']:
        if path[-1].is_goal():
            console.print("[bold green]Solution Found![/bold green]")
            # Side effect: update the solution in the state. LangGraph copies this out.
            state['solution'] = path
            return "solution_found"
    return "continue_search"

# Build the graph
workflow = StateGraph(ToTState)

workflow.add_node("initialize", initialize_search)
workflow.add_node("expand", expand_paths)
workflow.add_node("prune", prune_paths)

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "expand")
workflow.add_edge("expand", "prune")

workflow.add_conditional_edges(
    "prune",
    check_for_solution,
    {
        "solution_found": END,
        "continue_search": "expand"
    }
)

tot_agent = workflow.compile()
print("Tree-of-Thoughts agent graph compiled successfully.")

problem = "A farmer wants to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one other item. The farmer cannot leave the wolf alone with the goat, nor the goat alone with the cabbage. How can the farmer get everyone across safely?"

console.print("--- 🌳 Running Tree-of-Thoughts Agent ---")
# The recursion limit prevents infinite loops in our graph
config = {"recursion_limit": 15}
final_state = tot_agent.invoke({"problem_description": problem}, config=config)

console.print("\n--- ✅ ToT Agent Solution ---")
if final_state.get('solution'):
    solution_path = final_state['solution']
    # Use rich.Tree for a nice visual output
    tree = Tree("[bold magenta]Wolf, Goat, and Cabbage Solution Path[/bold magenta]")
    for i, state in enumerate(solution_path):
        tree.add(f"[green]{i+1}.[/green] {state.move_description}")
    console.print(tree)
else:
    console.print("[bold red]No solution found within the step limit.[/bold red]")

console.print("\n--- 🤔 Running Simple Chain-of-Thought Agent ---")
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class logic puzzle solver. Provide a step-by-step solution to the user's puzzle."),
    ("human", "{problem}")
])
cot_chain = cot_prompt | llm
cot_result = cot_chain.invoke({"problem": problem}).content
console.print(Markdown(cot_result))
