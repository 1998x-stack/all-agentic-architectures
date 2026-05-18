"""
Mental Loop - Agentic Architecture
Extracted from: 10_mental_loop.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv numpy

import os
import random
import numpy as np
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
from rich.table import Table

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Simulator (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()

class Portfolio(BaseModel):
    cash: float = 10000.0
    shares: int = 0
    
    def value(self, current_price: float) -> float:
        return self.cash + self.shares * current_price

class MarketSimulator(BaseModel):
    """A simple simulation of a stock market for one asset."""
    day: int = 0
    price: float = 100.0
    volatility: float = 0.1 # Standard deviation for price changes
    drift: float = 0.01 # General trend
    market_news: str = "Market is stable."
    portfolio: Portfolio = Field(default_factory=Portfolio)

    def step(self, action: str, amount: float = 0.0):
        """Advance the simulation by one day, executing a trade first."""
        # 1. Execute trade
        if action == "buy": # amount is number of shares
            shares_to_buy = int(amount)
            cost = shares_to_buy * self.price
            if self.portfolio.cash >= cost:
                self.portfolio.shares += shares_to_buy
                self.portfolio.cash -= cost
        elif action == "sell": # amount is number of shares
            shares_to_sell = int(amount)
            if self.portfolio.shares >= shares_to_sell:
                self.portfolio.shares -= shares_to_sell
                self.portfolio.cash += shares_to_sell * self.price
        
        # 2. Update market price (Geometric Brownian Motion)
        daily_return = np.random.normal(self.drift, self.volatility)
        self.price *= (1 + daily_return)
        
        # 3. Advance time
        self.day += 1
        
        # 4. Potentially update news
        if random.random() < 0.1: # 10% chance of new news
            self.market_news = random.choice(["Positive earnings report expected.", "New competitor enters the market.", "Macroeconomic outlook is strong.", "Regulatory concerns are growing."])
            # News affects drift
            if "Positive" in self.market_news or "strong" in self.market_news:
                self.drift = 0.05
            else:
                self.drift = -0.05
        else:
             self.drift = 0.01 # Revert to normal drift

    def get_state_string(self) -> str:
        return f"Day {self.day}: Price=${self.price:.2f}, News: {self.market_news}\nPortfolio: ${self.portfolio.value(self.price):.2f} ({self.portfolio.shares} shares, ${self.portfolio.cash:.2f} cash)"

print("Market simulator environment defined successfully.")

llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.4)

# Pydantic models for structured LLM outputs
class ProposedAction(BaseModel):
    """The high-level strategy proposed by the analyst.""" 
    strategy: str = Field(description="A high-level trading strategy, e.g., 'buy aggressively', 'sell cautiously', 'hold'.")
    reasoning: str = Field(description="Brief reasoning for the proposed strategy.")

class FinalDecision(BaseModel):
    """The final, concrete action to be executed."""
    action: str = Field(description="The final action to take: 'buy', 'sell', or 'hold'.")
    amount: float = Field(description="The number of shares to buy or sell. Should be 0 if holding.")
    reasoning: str = Field(description="Final reasoning, referencing the simulation results.")

# LangGraph State
class AgentState(TypedDict):
    real_market: MarketSimulator
    proposed_action: Optional[ProposedAction]
    simulation_results: Optional[List[Dict]]
    final_decision: Optional[FinalDecision]

# Graph Nodes
def propose_action_node(state: AgentState) -> Dict[str, Any]:
    """Observes the market and proposes a high-level strategy."""
    console.print("--- 🧐 Analyst Proposing Action ---")
    prompt = ChatPromptTemplate.from_template(
        "You are a sharp financial analyst. Based on the current market state, propose a trading strategy.\n\nMarket State:\n{market_state}"
    )
    proposer_llm = llm.with_structured_output(ProposedAction)
    chain = prompt | proposer_llm
    proposal = chain.invoke({"market_state": state['real_market'].get_state_string()})
    console.print(f"[yellow]Proposal:[/yellow] {proposal.strategy}. [italic]Reason: {proposal.reasoning}[/italic]")
    return {"proposed_action": proposal}

def run_simulation_node(state: AgentState) -> Dict[str, Any]:
    """Runs the proposed strategy in a sandboxed simulation."""
    console.print("--- 🤖 Running Simulations ---")
    strategy = state['proposed_action'].strategy
    num_simulations = 5
    simulation_horizon = 10 # days
    results = []

    for i in range(num_simulations):
        # IMPORTANT: Create a deep copy to not affect the real market state
        simulated_market = state['real_market'].model_copy(deep=True)
        initial_value = simulated_market.portfolio.value(simulated_market.price)

        # Translate strategy to a concrete action for the simulation
        if "buy" in strategy:
            action = "buy"
            # Aggressively = 25% of cash, Cautiously = 10%
            amount = (simulated_market.portfolio.cash * (0.25 if "aggressively" in strategy else 0.1)) / simulated_market.price
        elif "sell" in strategy:
            action = "sell"
            # Aggressively = 25% of shares, Cautiously = 10%
            amount = simulated_market.portfolio.shares * (0.25 if "aggressively" in strategy else 0.1)
        else:
            action = "hold"
            amount = 0
        
        # Run the simulation forward
        simulated_market.step(action, amount)
        for _ in range(simulation_horizon - 1):
            simulated_market.step("hold") # Just hold after the initial action
        
        final_value = simulated_market.portfolio.value(simulated_market.price)
        results.append({"sim_num": i+1, "initial_value": initial_value, "final_value": final_value, "return_pct": (final_value - initial_value) / initial_value * 100})
    
    console.print("[cyan]Simulation complete. Results will be passed to the risk manager.[/cyan]")
    return {"simulation_results": results}

def refine_and_decide_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes simulation results and makes a final, refined decision."""
    console.print("--- 🧠 Risk Manager Refining Decision ---")
    results_summary = "\n".join([f"Sim {r['sim_num']}: Initial=${r['initial_value']:.2f}, Final=${r['final_value']:.2f}, Return={r['return_pct']:.2f}%" for r in state['simulation_results']])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a cautious risk manager. Your analyst proposed a strategy. You have run simulations to test it. Based on the potential outcomes, make a final, concrete decision. If results are highly variable or negative, reduce risk (e.g., buy/sell fewer shares, or hold).\n\nInitial Proposal: {proposal}\n\nSimulation Results:\n{results}\n\nReal Market State:\n{market_state}"
    )
    decider_llm = llm.with_structured_output(FinalDecision)
    chain = prompt | decider_llm
    final_decision = chain.invoke({
        "proposal": state['proposed_action'].strategy,
        "results": results_summary,
        "market_state": state['real_market'].get_state_string()
    })
    console.print(f"[green]Final Decision:[/green] {final_decision.action} {final_decision.amount:.0f} shares. [italic]Reason: {final_decision.reasoning}[/italic]")
    return {"final_decision": final_decision}

def execute_in_real_world_node(state: AgentState) -> Dict[str, Any]:
    """Executes the final decision in the real market environment."""
    console.print("--- 🚀 Executing in Real World ---")
    decision = state['final_decision']
    real_market = state['real_market']
    real_market.step(decision.action, decision.amount)
    console.print(f"[bold]Execution complete. New market state:[/bold]\n{real_market.get_state_string()}")
    return {"real_market": real_market}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("propose", propose_action_node)
workflow.add_node("simulate", run_simulation_node)
workflow.add_node("refine", refine_and_decide_node)
workflow.add_node("execute", execute_in_real_world_node)

workflow.set_entry_point("propose")
workflow.add_edge("propose", "simulate")
workflow.add_edge("simulate", "refine")
workflow.add_edge("refine", "execute")
workflow.add_edge("execute", END)

simulator_agent = workflow.compile()
print("Simulator-in-the-loop agent graph compiled successfully.")

real_market = MarketSimulator()
console.print("--- Initial Market State ---")
console.print(real_market.get_state_string())

# --- Day 1 Run ---
console.print("\n--- Day 1: Good News Hits! ---")
real_market.market_news = "Positive earnings report expected."
real_market.drift = 0.05
initial_state = {"real_market": real_market}
final_state = simulator_agent.invoke(initial_state)
real_market = final_state['real_market']

# --- Day 2 Run ---
console.print("\n--- Day 2: Bad News Hits! ---")
real_market.market_news = "New competitor enters the market."
real_market.drift = -0.05
initial_state = {"real_market": real_market}
final_state = simulator_agent.invoke(initial_state)
real_market = final_state['real_market']
