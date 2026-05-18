"""
Episodic + Semantic - Agentic Architecture
Extracted from: 08_episodic_with_semantic.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv langchain_community langchain-openai neo4j faiss-cpu tiktoken

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Pydantic for data modeling
from pydantic import BaseModel, Field

# LangChain components
from langchain_nebius import ChatNebius, NebiusEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
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
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Memory Stack (Nebius)"

# Check for required environment variables
required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()
llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0)
embeddings = NebiusEmbeddings()

# --- 1. Vector Store for Episodic Memory ---
# In a real application, you'd persist this. For this example, it's in-memory.
try:
    episodic_vector_store = FAISS.from_texts(["Initial document to bootstrap the store"], embeddings)
except ImportError:
    console.print("[bold red]FAISS not installed. Please run `pip install faiss-cpu`.")[/bold red]
    episodic_vector_store = None

# --- 2. Graph DB for Semantic Memory ---
try:
    graph = Neo4jGraph(
        url=os.environ.get("NEO4J_URI"),
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD")
    )
    # Clear the graph for a clean run
    graph.query("MATCH (n) DETACH DELETE n")
except Exception as e:
    console.print(f"[bold red]Failed to connect to Neo4j: {e}. Please check your credentials and connection.[/bold red]")
    graph = None

# --- 3. Pydantic Models for the "Memory Maker" ---
# Define the structure of knowledge we want to extract.
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node, which can be a person's name, a company ticker, or a concept.")
    type: str = Field(description="The type of the node (e.g., 'User', 'Company', 'InvestmentPhilosophy').")
    properties: Dict[str, Any] = Field(description="A dictionary of properties for the node.")

class Relationship(BaseModel):
    source: Node = Field(description="The source node of the relationship.")
    target: Node = Field(description="The target node of the relationship.")
    type: str = Field(description="The type of the relationship (e.g., 'IS_A', 'INTERESTED_IN').")
    properties: Dict[str, Any] = Field(description="A dictionary of properties for the relationship.")

class KnowledgeGraph(BaseModel):
    """Represents the structured knowledge extracted from a conversation."""
    relationships: List[Relationship] = Field(description="A list of relationships to be added to the knowledge graph.")

# --- 4. The "Memory Maker" Agent ---
def create_memories(user_input: str, assistant_output: str):
    conversation = f"User: {user_input}\nAssistant: {assistant_output}"
    
    # 4a. Create Episodic Memory (Summarization)
    console.print("--- Creating Episodic Memory (Summary) ---")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarization expert. Create a concise, one-sentence summary of the following user-assistant interaction. This summary will be used as a memory for future recall."),
        ("human", "Interaction:\n{interaction}")
    ])
    summarizer = summary_prompt | llm
    episodic_summary = summarizer.invoke({"interaction": conversation}).content
    
    new_doc = Document(page_content=episodic_summary, metadata={"created_at": uuid.uuid4().hex})
    episodic_vector_store.add_documents([new_doc])
    console.print(f"[green]Episodic memory created:[/green] '{episodic_summary}'")
    
    # 4b. Create Semantic Memory (Fact Extraction)
    console.print("--- Creating Semantic Memory (Graph) ---")
    extraction_llm = llm.with_structured_output(KnowledgeGraph)
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge extraction expert. Your task is to identify key entities and their relationships from a conversation and model them as a graph. Focus on user preferences, goals, and stated facts."),
        ("human", "Extract all relationships from this interaction:\n{interaction}")
    ])
    extractor = extraction_prompt | extraction_llm
    try:
        kg_data = extractor.invoke({"interaction": conversation})
        if kg_data.relationships:
            for rel in kg_data.relationships:
                graph.add_graph_documents([rel], include_source=True)
            console.print(f"[green]Semantic memory created:[/green] Added {len(kg_data.relationships)} relationships to the graph.")
        else:
            console.print("[yellow]No new semantic memories identified in this interaction.[/yellow]")
    except Exception as e:
        console.print(f"[red]Could not extract or save semantic memory: {e}[/red]")

if episodic_vector_store and graph:
    print("Memory components initialized successfully.")

# Define the state for our LangGraph agent
class AgentState(TypedDict):
    user_input: str
    retrieved_memories: Optional[str]
    generation: str

# Define the nodes of the graph

def retrieve_memory(state: AgentState) -> Dict[str, Any]:
    """Node that retrieves memories from both episodic and semantic stores."""
    console.print("--- Retrieving Memories ---")
    user_input = state['user_input']
    
    # Retrieve from episodic memory
    retrieved_docs = episodic_vector_store.similarity_search(user_input, k=2)
    episodic_memories = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Retrieve from semantic memory
    # This is a simple retrieval; more advanced would involve entity extraction from the query
    try:
        graph_schema = graph.get_schema
        # Using a fulltext index for better retrieval. Neo4j automatically creates one on node properties.
        # A more robust solution might involve extracting entities from user_input first.
        semantic_memories = str(graph.query("""
            UNWIND $keywords AS keyword
            CALL db.index.fulltext.queryNodes("entity", keyword) YIELD node, score
            MATCH (node)-[r]-(related_node)
            RETURN node, r, related_node LIMIT 5
            """, {'keywords': user_input.split()}))
    except Exception as e:
        semantic_memories = f"Could not query graph: {e}"
        
    retrieved_content = f"Relevant Past Conversations (Episodic Memory):\n{episodic_memories}\n\nRelevant Facts (Semantic Memory):\n{semantic_memories}"
    console.print(f"[cyan]Retrieved Context:\n{retrieved_content}[/cyan]")
    
    return {"retrieved_memories": retrieved_content}

def generate_response(state: AgentState) -> Dict[str, Any]:
    """Node that generates a response using the retrieved memories."""
    console.print("--- Generating Response ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and personalized financial assistant. Use the retrieved memories to inform your response and tailor it to the user. If the memories indicate a user's preference (e.g., they are a conservative investor), you MUST respect it."),
        ("human", "My question is: {user_input}\n\nHere are some memories that might be relevant:\n{retrieved_memories}")
    ])
    generator = prompt | llm
    generation = generator.invoke(state).content
    console.print(f"[green]Generated Response:\n{generation}[/green]")
    return {"generation": generation}

def update_memory(state: AgentState) -> Dict[str, Any]:
    """Node that updates the memory with the latest interaction."""
    console.print("--- Updating Memory ---")
    create_memories(state['user_input'], state['generation'])
    return {}

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_memory)
workflow.add_node("generate", generate_response)
workflow.add_node("update", update_memory)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "update")
workflow.add_edge("update", END)

memory_agent = workflow.compile()
print("Memory-augmented agent graph compiled successfully.")

def run_interaction(query: str):
    result = memory_agent.invoke({"user_input": query})
    return result['generation']

console.print("\n--- 💬 INTERACTION 1: Seeding Memory ---")
run_interaction("Hi, my name is Alex. I'm a conservative investor, and I'm mainly interested in established tech companies.")

console.print("\n--- 💬 INTERACTION 2: Asking a specific question ---")
run_interaction("What do you think about Apple (AAPL)?")

console.print("\n--- 🧠 INTERACTION 3: THE MEMORY TEST ---")
run_interaction("Based on my goals, what's a good alternative to that stock?")

console.print("--- 🔍 Inspecting Episodic Memory (Vector Store) ---")
# We'll do a similarity search for a general concept to see what comes up
retrieved_docs = episodic_vector_store.similarity_search("User's investment strategy", k=3)
for i, doc in enumerate(retrieved_docs):
    print(f"{i+1}. {doc.page_content}")

console.print("\n--- 🕸️ Inspecting Semantic Memory (Graph Database) ---")
print(f"Graph Schema:\n{graph.get_schema}")

# Cypher query to see who is interested in what
query_result = graph.query("MATCH (n:User)-[r:INTERESTED_IN|HAS_GOAL]->(m) RETURN n, r, m")
print(f"Relationships in Graph:\n{query_result}")
