"""
Graph - Agentic Architecture
Extracted from: 12_graph.ipynb
"""

# !pip install -q -U langchain-nebius langchain langgraph rich python-dotenv langchain_community neo4j

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Pydantic for data modeling
from pydantic import BaseModel, Field

# LangChain components
from langchain_nebius import ChatNebius
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as V1BaseModel

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown

# --- API Key and Tracing Setup ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Graph Memory (Nebius)"

required_vars = ["NEBIUS_API_KEY", "LANGCHAIN_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
for var in required_vars:
    if var not in os.environ:
        print(f"Warning: Environment variable {var} not set.")

print("Environment variables loaded and tracing is set up.")

console = Console()
llm = ChatNebius(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0)

# Connect to our Neo4j database
try:
    graph = Neo4jGraph()
    # Clear the graph for a clean run
    graph.query("MATCH (n) DETACH DELETE n")
except Exception as e:
    console.print(f"[bold red]Failed to connect to Neo4j: {e}. Please check your credentials and connection.[/bold red]")
    graph = None

# Pydantic models for structured extraction (using LangChain's v1 BaseModel for compatibility with older structured output methods)
class Node(V1BaseModel):
    id: str = Field(description="Unique name or identifier for the entity.")
    type: str = Field(description="The type or label of the entity (e.g., Person, Company, Product).")

class Relationship(V1BaseModel):
    source: Node
    target: Node
    type: str = Field(description="The type of relationship (e.g., WORKS_FOR, ACQUIRED).")

class KnowledgeGraph(V1BaseModel):
    """A graph of nodes and relationships."""
    relationships: List[Relationship]

# The Graph Maker Agent
def get_graph_maker_chain():
    extractor_llm = llm.with_structured_output(KnowledgeGraph)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting information from text and building a knowledge graph. Extract all entities (nodes) and relationships from the provided text. The relationship type should be a verb in all caps, like 'WORKS_FOR' or 'ACQUIRED'."),
        ("human", "Extract a knowledge graph from the following text:\n\n{text}")
    ])
    return prompt | extractor_llm

graph_maker_agent = get_graph_maker_chain()
print("Successfully connected to Neo4j and defined the Graph Maker Agent.")

unstructured_documents = [
    "On May 15, 2023, global tech giant AlphaCorp announced its acquisition of startup BetaSolutions, a leader in cloud-native database technology.",
    "Dr. Evelyn Reed, a renowned AI researcher, has been the Chief Science Officer at AlphaCorp since 2021. She leads the team responsible for the QuantumLeap AI platform.",
    "Innovate Inc.'s flagship product, NeuraGen, is seen as a direct competitor to AlphaCorp's QuantumLeap AI. Meanwhile, Innovate Inc. recently hired Johnathan Miles as their new CTO."
]
for i, doc in enumerate(unstructured_documents):
    console.print(f"--- Ingesting Document {i+1} ---")
    try:
        kg_data = graph_maker_agent.invoke({"text": doc})
        if kg_data.relationships:
            graph.add_graph_documents(graph_documents=kg_data.relationships, include_source=False)
            console.print(f"[green]Successfully added {len(kg_data.relationships)} relationships to the graph.[/green]")
        else:
             console.print("[yellow]No relationships extracted.[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to process document: {e}[/red]")

console.print("--- ✅ Knowledge Graph Ingestion Complete ---")
console.print("\n--- Graph Schema ---")
console.print(graph.schema)

# LangChain has a built-in chain for this, but we'll inspect the components
# to understand how it works.
cypher_generation_prompt = ChatPromptTemplate.from_template(
    """You are an expert Neo4j Cypher query developer. Your task is to convert a user's natural language question into a valid Cypher query.
You must use the provided graph schema to construct the query. Do not use any node labels or relationship types that are not in the schema.
Return ONLY the Cypher query, with no additional text or explanations.

Graph Schema:
{schema}

User Question:
{question}
"""
)

cypher_response_prompt = ChatPromptTemplate.from_template(
    """You are an assistant that provides clear, natural language answers based on query results from a knowledge graph.
Use the context from the graph query result to answer the user's original question.

User Question: {question}
Query Result: {context}
"""
)

def query_graph(question: str) -> Dict[str, Any]:
    """The full Text-to-Cypher and synthesis pipeline."""
    console.print(f"\n[bold]Question:[/bold] {question}")
    
    # 1. Generate Cypher Query
    console.print("--- ➡️ Generating Cypher Query ---")
    cypher_chain = cypher_generation_prompt | llm
    generated_cypher = cypher_chain.invoke({"schema": graph.schema, "question": question}).content
    console.print(f"[cyan]Generated Cypher:\n{generated_cypher}[/cyan]")
    
    # 2. Execute Cypher Query
    console.print("--- ⚡ Executing Query ---")
    try:
        context = graph.query(generated_cypher)
        console.print(f"[yellow]Query Result:\n{context}[/yellow]")
    except Exception as e:
        console.print(f"[red]Cypher Query Failed: {e}[/red]")
        return {"answer": "I was unable to execute a query to find the answer to your question."}
    
    # 3. Synthesize Final Answer
    console.print("--- 🗣️ Synthesizing Final Answer ---")
    synthesis_chain = cypher_response_prompt | llm
    answer = synthesis_chain.invoke({"question": question, "context": context}).content
    
    return {"answer": answer}

print("Graph-Querying Agent defined successfully.")

# Test 1: Simple fact retrieval (requires info from doc 2)
result1 = query_graph("Who works for AlphaCorp?")
console.print("\n--- Final Answer ---")
console.print(Markdown(result1['answer']))

# Test 2: Another simple fact retrieval (requires info from doc 1)
result2 = query_graph("What company did AlphaCorp acquire?")
console.print("\n--- Final Answer ---")
console.print(Markdown(result2['answer']))

# Test 3: The multi-hop reasoning question (requires info from all 3 docs)
result3 = query_graph("What companies compete with the products made by the company that acquired BetaSolutions?")
console.print("\n--- Final Answer ---")
console.print(Markdown(result3['answer']))
