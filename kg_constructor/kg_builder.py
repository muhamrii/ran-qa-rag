import os
import json
import hashlib
from typing import Dict, List, Tuple, Any

import pandas as pd
from langchain.llms import OpenAI
from langchain_community import llms
from transformers import pipeline
# from langchain.embeddings import OpenAIEmbeddings  # Uncomment if embeddings are needed
#from neo4j import GraphDatabase

# --------------------------------------------------
# Configuration
# --------------------------------------------------
# CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
# if os.path.exists(CONFIG_PATH):
#     with open(CONFIG_PATH, "r") as f:
#         config = yaml.safe_load(f)
#     NEO4J_URI = config["neo4j"]["uri"]
#     NEO4J_USER = config["neo4j"]["user"]
#     NEO4J_PASS = config["neo4j"]["password"]
# else:
#     raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

CACHE_DIR = os.getenv("KG_CACHE_DIR", ".kg_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


### LOCAL LLM WRAPPER
class LocalLLM:
    """
    Simple wrapper for Hugging Face text-generation pipeline to mimic OpenAI's interface.
    """
    def __init__(self, model_name="facebook/opt-1.3b", max_new_tokens=512):
        self.generator = pipeline("text-generation", model=model_name, device_map="auto")
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt):
        response = self.generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=True)
        return response[0]['generated_text'][len(prompt):].strip()


# --------------------------------------------------
# Helper: Schema fingerprint
# --------------------------------------------------
def fingerprint_schema(df: pd.DataFrame) -> str:
    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
    raw = json.dumps(schema, sort_keys=True).encode('utf-8')
    return hashlib.md5(raw).hexdigest()

# --------------------------------------------------
# Parser & Description
# --------------------------------------------------
def describe_tables(dfs: Dict[str, pd.DataFrame], llm: LocalLLM) -> Dict[str, str]:
    descriptions = {}
    for name, df in dfs.items():
        fp = fingerprint_schema(df)
        cache_file = os.path.join(CACHE_DIR, f"{name}_{fp}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                descriptions[name] = json.load(f)["description"]
            continue

        summary = [f"Table: {name}", f"Shape: {df.shape}"]
        summary.append("Columns:")
        for col in df.columns:
            sample = df[col].dropna().unique()[:3].tolist()
            summary.append(f"- {col} (dtype={df[col].dtype}, sample={sample})")
        
        prompt = (
            "You are a data expert. Given the table schema below, generate a one-line description of the table and each column.\n"
            + "\n".join(summary)
        )
        response = llm(prompt)
        descriptions[name] = response.strip()
        with open(cache_file, 'w') as f:
            json.dump({"schema_fp": fp, "description": response}, f)
    return descriptions

# --------------------------------------------------
# Entity & Relationship Extraction
# --------------------------------------------------
def extract_entities_and_relations(descriptions: Dict[str, str], llm: LocalLLM) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt = (
        "You are a graph modeling expert. Given the following table descriptions, "
        "1) Identify core entity types and their key properties. "
        "2) Identify relationships between these entities. "
        "\nDescriptions:\n" + json.dumps(descriptions, indent=2)
    )
    output = llm(prompt)
    data = json.loads(output)
    return data.get('entities', []), data.get('relationships', [])

# --------------------------------------------------
# Cypher Generation
# --------------------------------------------------
def build_cypher_commands(entities: List[Dict], relations: List[Dict]) -> List[str]:
    cyphers = []
    for ent in entities:
        props = ", ".join([f"{k}: ${{{k}}}" for k in ent.get('properties', [])])
        cyphers.append(f"MERGE (n:{ent['label']} {{{props}}});")
    for rel in relations:
        cyphers.append(
            f"MATCH (a:{rel['start_label']} {{id: $start_id}}),"
            f" (b:{rel['end_label']} {{id: $end_id}})\n"
            f"MERGE (a)-[r:{rel['type']}]->(b);"
        )
    return cyphers

# --------------------------------------------------
# Execution on Neo4j
# --------------------------------------------------
# def execute_cypher(commands: List[str], params: Dict[str, Any] = {}):
#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
#     with driver.session() as session:
#         for cy in commands:
#             session.run(cy, **params)
#     driver.close()