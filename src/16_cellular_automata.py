"""
Cellular Automata - Agentic Architecture
Extracted from: 16_cellular_automata.ipynb
"""

# !pip install -q -U langchain-nebius rich python-dotenv numpy

import os
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from IPython.display import clear_output

# LangChain for optional final summary
from langchain_nebius import ChatNebius
from langchain_core.prompts import ChatPromptTemplate

# For pretty printing and visualization
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Cellular Automata (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()

class CellAgent:
    """A single agent in our grid. Its only job is to update its value based on neighbors."""
    def __init__(self, cell_type: str, item: Optional[str] = None):
        self.type = cell_type # 'EMPTY', 'OBSTACLE', 'SHELF', 'PACKING_STATION'
        self.item = item
        self.pathfinding_value = float('inf')

    def update_value(self, neighbors: List['CellAgent']):
        """The core local rule: my new value is 1 + the minimum value of my non-obstacle neighbors."""
        if self.type == 'OBSTACLE':
            return float('inf')
        
        min_neighbor_value = float('inf')
        for neighbor in neighbors:
            if neighbor.pathfinding_value < min_neighbor_value:
                min_neighbor_value = neighbor.pathfinding_value
        
        # The +1 represents the cost of moving from a neighbor to this cell
        return min(self.pathfinding_value, min_neighbor_value + 1)

class WarehouseGrid:
    """Manages the entire grid of CellAgents and the simulation loop."""
    def __init__(self, layout: List[str]):
        self.height = len(layout)
        self.width = len(layout[0])
        self.grid = self._create_grid_from_layout(layout)
        self.item_locations = self._get_item_locations()

    def _create_grid_from_layout(self, layout):
        grid = np.empty((self.height, self.width), dtype=object)
        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                if char == ' ':
                    grid[r, c] = CellAgent('EMPTY')
                elif char == '#':
                    grid[r, c] = CellAgent('OBSTACLE')
                elif char == 'P':
                    grid[r, c] = CellAgent('PACKING_STATION')
                else: # It's an item
                    grid[r, c] = CellAgent('SHELF', item=char)
        return grid

    def _get_item_locations(self) -> Dict[str, Tuple[int, int]]:
        locations = {}
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c].type == 'SHELF':
                    locations[self.grid[r, c].item] = (r, c)
                if self.grid[r, c].type == 'PACKING_STATION':
                    locations['P'] = (r, c)
        return locations

    def get_neighbors(self, r: int, c: int) -> List[CellAgent]:
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # N, S, E, W
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbors.append(self.grid[nr, nc])
        return neighbors

    def tick(self) -> bool:
        """Performs one synchronous update of all cells. Returns True if any value changed."""
        changed = False
        # First, calculate all new values based on the current state
        new_values = np.empty((self.height, self.width))
        for r in range(self.height):
            for c in range(self.width):
                neighbors = self.get_neighbors(r, c)
                new_values[r, c] = self.grid[r, c].update_value(neighbors)
        
        # Then, apply all the new values
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c].pathfinding_value != new_values[r, c]:
                    self.grid[r, c].pathfinding_value = new_values[r, c]
                    changed = True
        return changed

    def visualize(self, show_values: bool = False, title: str = "Warehouse Grid"):
        """Displays the grid state using Rich."""
        table = Table(title=title, show_header=False, show_edge=True, padding=0)
        for _ in range(self.width):
            table.add_column(justify="center")
        
        for r in range(self.height):
            row_renderables = []
            for c in range(self.width):
                cell = self.grid[r, c]
                val = cell.pathfinding_value
                display_char = ''
                if cell.type == 'EMPTY': display_char = '[grey70]·[/grey70]'
                elif cell.type == 'OBSTACLE': display_char = '[red]█[/red]'
                elif cell.type == 'PACKING_STATION': display_char = '[bold green]P[/bold green]'
                elif cell.type == 'SHELF': display_char = f'[bold blue]{cell.item}[/bold blue]'

                if show_values and val != float('inf'):
                    # Color code the path values
                    color = int(255 - (val * 5) % 255)
                    row_renderables.append(f"[rgb({color},{color},{color}) on rgb(30,30,60)]{int(val):^3}[/]")
                else:
                    row_renderables.append(f" {display_char} ")
            table.add_row(*row_renderables)
        console.print(table)

print("Cellular Automata environment defined successfully.")

def propagate_path_wave(grid: WarehouseGrid, target_pos: Tuple[int, int], visualize_steps: bool = False):
    """Resets and then runs the simulation until the pathfinding values stabilize."""
    # Reset all pathfinding values
    for r in range(grid.height):
        for c in range(grid.width):
            grid.grid[r, c].pathfinding_value = float('inf')
            
    # Set the target's value to 0 to start the wave
    grid.grid[target_pos].pathfinding_value = 0
    
    tick_count = 0
    while True:
        tick_count += 1
        if visualize_steps:
            clear_output(wait=True)
            grid.visualize(show_values=True, title=f"Path Wave Propagation (Tick #{tick_count})")
            time.sleep(0.1)
        
        changed = grid.tick()
        if not changed:
            break
    if visualize_steps:
        clear_output(wait=True)
        grid.visualize(show_values=True, title=f"Path Wave Propagation (Stabilized at Tick #{tick_count})")

def trace_and_move_item(grid: WarehouseGrid, start_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Follows the gradient from the start position back to the target (value 0)."""
    path = [start_pos]
    r, c = start_pos
    
    while grid.grid[r, c].pathfinding_value > 0:
        neighbors = grid.get_neighbors(r, c)
        best_neighbor_pos = None
        min_val = grid.grid[r, c].pathfinding_value
        
        # Find the neighbor with the lowest pathfinding value
        for neighbor_cell in neighbors:
            # Find the position of the neighbor cell
            pos_list = np.where(grid.grid == neighbor_cell)
            if len(pos_list[0]) > 0:
                nr, nc = pos_list[0][0], pos_list[1][0]
                if neighbor_cell.pathfinding_value < min_val:
                    min_val = neighbor_cell.pathfinding_value
                    best_neighbor_pos = (nr, nc)
        
        if best_neighbor_pos:
            path.append(best_neighbor_pos)
            r, c = best_neighbor_pos
        else:
            console.print("[red]Error: Path tracing got stuck. No downhill neighbor found.[/red]")
            break
            
    return path

print("Emergent behavior functions defined successfully.")

def fulfill_order(layout: List[str], order: List[str], visualize_waves: bool = False):
    """The main orchestration function."""
    grid = WarehouseGrid(layout)
    console.print("--- Initial Warehouse State ---")
    grid.visualize()
    
    packing_station_pos = grid.item_locations['P']
    
    for i, item_id in enumerate(order):
        panel_title = f"[bold]Step {i+1}: Fulfill Item '{item_id}'[/bold]"
        log_messages = []
        
        item_pos = grid.item_locations.get(item_id)
        if not item_pos:
            console.print(Panel(f"[red]Error: Item '{item_id}' not found in warehouse.[/red]", title=panel_title))
            continue
            
        # 1. Compute the path wave from the packing station
        log_messages.append("🌊 Computing path wave from Packing Station...")
        propagate_path_wave(grid, packing_station_pos, visualize_steps=visualize_waves)
        
        # 2. Trace the path for the current item
        log_messages.append(f"🚚 Found path for item {item_id}. Moving along gradient...")
        path = trace_and_move_item(grid, item_pos)
        path_str = ' -> '.join(map(str, path))
        log_messages.append(f"Path: {path_str}")

        # 3. Update the grid state (item is now at packing station)
        grid.grid[item_pos].type = 'EMPTY'
        grid.grid[item_pos].item = None
        log_messages.append(f"✅ Item '{item_id}' has been moved to the packing station.")
        console.print(Panel('\n'.join(log_messages), title=panel_title, border_style="blue"))
        
    console.print(Panel(f"The system successfully fulfilled the order for items {order} by emergently computing paths through local cell interactions.", title="[bold green]🎉 Order Fulfillment Complete![/bold green]", border_style="green"))
    return grid

# --- Main Execution ---
warehouse_layout = [
    "#######",
    "# D   #",
    "# ### #",
    "#A#C# #",
    "# # #B#",
    "#  P  #",
    "#######",
]
order_to_fulfill = ['A', 'B']
final_grid = fulfill_order(warehouse_layout, order_to_fulfill, visualize_waves=True)

# --- Optional: LLM Interpretation ---
console.print("\n--- 🤖 LLM Interpretation of the Final State ---")
llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1")
summary_prompt = ChatPromptTemplate.from_template("You are a logistics manager. Briefly summarize the outcome of the following order fulfillment report.\n\nOrder: {order}\nFinal Warehouse State: All items from the order have been moved to the packing station. Items A and B were retrieved. Original locations were {loc_A} and {loc_B}. The floor is now clear.")
summary_chain = summary_prompt | llm
final_summary = summary_chain.invoke({
    "order": order_to_fulfill, 
    "loc_A": WarehouseGrid(warehouse_layout).item_locations['A'],
    "loc_B": WarehouseGrid(warehouse_layout).item_locations['B']
}).content
console.print(Markdown(final_summary))
