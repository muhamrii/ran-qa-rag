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
# Google AI Studio LLM Wrapper (Gemini 1.5 Flash, no-cost tier)
# --------------------------------------------------
import requests

class GoogleAIStudioLLM:
    """
    Wrapper for Google AI Studio API (Gemini 1.5 Flash, no-cost tier).
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def __call__(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        # Extract the generated text
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
# --------------------------------------------------
# Configuration
# --------------------------------------------------

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    GOOGLE_API_KEY = config["google_ai_studio"]["api_key"]
    GOOGLE_MODEL = config["google_ai_studio"].get("model", "gemini-1.5-flash-latest")
    NEO4J_URI = config["neo4j"]["uri"]
    NEO4J_USER = config["neo4j"]["user"]
    NEO4J_PASS = config["neo4j"]["password"]
else:
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

# Usage:
llm = GoogleAIStudioLLM(api_key=GOOGLE_API_KEY, model=GOOGLE_MODEL)

CACHE_DIR = os.getenv("KG_CACHE_DIR", ".kg_cache")
os.makedirs(CACHE_DIR, exist_ok=True)



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
import time
import requests
from requests.exceptions import HTTPError

def describe_tables(dfs: Dict[str, pd.DataFrame], llm: "GoogleAIStudioLLM") -> Dict[str, str]:
    """
    Describe tables using Google AI Studio LLM (Gemini 1.5 Flash, no-cost tier).
    Handles rate limits by waiting and retrying on HTTP 429 errors.
    Caches results to avoid repeated calls for the same data.
    """
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
            "You are a data and radio access network telecommunication expert. Given the table schema below, generate a one-line description of the table and each column.\n"
            + "\n".join(summary)
        )

        # Retry logic for rate limiting
        max_retries = 5
        retry_wait = 60  # seconds to wait after 429
        for attempt in range(max_retries):
            try:
                response = llm(prompt)
                break
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit (429). Waiting {retry_wait} seconds before retrying...")
                        time.sleep(retry_wait)
                        continue
                    else:
                        raise
                else:
                    raise
        else:
            raise RuntimeError("Max retries exceeded for LLM API.")

        descriptions[name] = response.strip()
        with open(cache_file, 'w') as f:
            json.dump({"schema_fp": fp, "description": response}, f)
        time.sleep(2)  # Reduce frequency of requests
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