"""
Build a knowledge graph from DataFrames using LangChain/LLMs.
Generate and execute Neo4j Cypher commands.
"""
import pandas as pd
from langchain.llms import OpenAI
# from neo4j import GraphDatabase

def describe_all_tables_llm(metadata2):
    """
    Generate a human-readable description for all tables and their columns from metadata2 using LangChain OpenAI LLM.
    """
    descriptions = []
    llm = OpenAI(temperature=0)
    for table, columns in metadata2.items():
        prompt = (
            f"You are a telco expert. Given the following table schema, generate a concise, human-readable description of the table and a one-line description for each column.\n\n"
            f"Table: {table}\n"
            f"Columns:\n"
        )
        for col in columns:
            prompt += f"  - {col}\n"
        prompt += (
            "\nFormat your response as:\n"
            "Table Description: ...\n"
            "Column Descriptions:\n"
            "  - column_name: description\n"
        )
        description = llm.invoke(prompt)
        descriptions.append(f"Table: {table}\n{description}\n")
    return "\n".join(descriptions)


def extract_entities(df):
    # Dummy function to extract entities
    return ["Cell", "Metric"]

def generate_cypher(entities):
    # Dummy function to generate Cypher
    return "CREATE (c:Cell {id: 1})"