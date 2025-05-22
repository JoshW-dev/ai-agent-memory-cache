from typing import List, Optional, Dict, TypedDict, Tuple
from openai import OpenAI # P2-T1: Switched to OpenAI
import os # For API Key
import uuid # P2-T2
from datetime import datetime, timezone # P2-T2
import numpy as np # P2-T3
import chromadb # Added import
import json # Added for P3-T3

# P1-T6: Define ActionSequence Type (List[str])
ActionSequence = List[str]

# Configuration Constants
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # P2-T1
SIMILARITY_THRESHOLD_TAU = 0.60 # P2-T3 - Lowered further to 0.70 for testing, NOW 0.60 based on P3-T4 testing
SCORE_THRESHOLD_EPSILON = 0.2 # For P2-T4, but good to have for lookup logic
REWARD_ALPHA = 0.3 # P2-T4 EMA factor
TOP_K_RESULTS = 3 # For ChromaDB queries

# P2-T2: Define CacheEntry structure
class CacheEntry(TypedDict):
    id: uuid.UUID
    prompt_raw: str
    embedding: List[float] # Assuming _generate_embedding returns List[float] or None
    actions: ActionSequence
    score: float
    created_at: datetime
    updated_at: datetime

class MemoryCache:
    def __init__(self):
        self._openai_client = OpenAI()
        # self._cache: List[CacheEntry] = [] # Will be replaced by ChromaDB
        
        # Initialize ChromaDB client and collection
        self._chroma_client = chromadb.Client() # For in-memory client
        # For persistent client:
        # self._chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
        self._collection = self._chroma_client.get_or_create_collection(
            name="memory_cache_collection",
            # Optionally, specify the embedding function if not using OpenAI's default with Chroma
            # metadata={"hnsw:space": "cosine"} # Ensure cosine distance if needed
        )
        print("ChromaDB client and collection initialized.")

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Helper function to generate embedding using OpenAI."""
        try:
            response = self._openai_client.embeddings.create(
                input=text,
                model=OPENAI_EMBEDDING_MODEL
            )
            # OpenAI embeddings are already normalized
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding for '{text}': {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Assumes embeddings are already L2-normalized (OpenAI embeddings are)
        # For normalized vectors, cosine similarity is the dot product.
        return np.dot(np.array(vec1), np.array(vec2))

    def lookup(self, prompt: str) -> Optional[Tuple[uuid.UUID, ActionSequence]]:
        """
        Performs a similarity search in ChromaDB for the given prompt.
        If a sufficiently similar and high-scoring entry is found, 
        its ID and actions are returned as a tuple (uuid.UUID, ActionSequence).
        """
        # print(f"DEBUG: lookup() called with prompt: '{prompt}'") # Reduced verbosity
        query_embedding = self._generate_embedding(prompt)
        if query_embedding is None:
            print(f"Error: Failed to generate embedding for lookup prompt: '{prompt}'.") # Keep error
            return None

        if self._collection.count() == 0:
            # print("DEBUG: ChromaDB collection is empty. Nothing to lookup.") # Reduced verbosity
            return None

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K_RESULTS,
                include=["metadatas", "distances"] 
            )
        except Exception as e:
            print(f"Error querying ChromaDB: {e}") # Keep error
            return None

        # print(f"DEBUG: ChromaDB query results for '{prompt}':") # Reduced verbosity

        if not results or not results.get('ids') or not results['ids'][0]:
            # print("DEBUG: No results returned from ChromaDB query or results format unexpected.") # Reduced verbosity
            return None

        result_ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for i in range(len(result_ids)):
            entry_id_str = result_ids[i]
            distance = distances[i]
            metadata = metadatas[i]
            
            similarity = 1 - distance 
            prompt_raw = metadata.get("prompt_raw", "[prompt_raw not found]")
            score = metadata.get("score", 0.0)
            actions_json = metadata.get("actions_json", "[]")

            # print(f"  Candidate {i+1}: ID={entry_id_str}, Prompt='{prompt_raw}', Similarity={similarity:.4f}, Score={score:.2f}") # Reduced verbosity

            if similarity >= SIMILARITY_THRESHOLD_TAU:
                if score >= SCORE_THRESHOLD_EPSILON:
                    # print(f"    DEBUG: Potential HIT! ID={entry_id_str}. Similarity and Score meet thresholds.") # Reduced verbosity
                    try:
                        actions = json.loads(actions_json)
                        # print(f"DEBUG: HIT! Prompt: '{prompt}'. Best match: '{prompt_raw}' (ID: {entry_id_str}). Similarity: {similarity:.4f}, Score: {score:.2f}") # Keep high-level HIT from demo
                        return uuid.UUID(entry_id_str), actions
                    except json.JSONDecodeError as e:
                        print(f"Error decoding actions_json for entry ID {entry_id_str}: {e}. Skipping this entry.") # Keep error
                        continue 
                # else:
                    # print(f"    DEBUG: MISS (Score too low). ID={entry_id_str}, Score={score:.2f} < {SCORE_THRESHOLD_EPSILON}") # Reduced verbosity
            # else:
                # print(f"    DEBUG: MISS (Similarity too low). ID={entry_id_str}, Similarity={similarity:.4f} < {SIMILARITY_THRESHOLD_TAU}") # Reduced verbosity
        
        # print(f"DEBUG: MISS. No entry in top {len(result_ids)} results for '{prompt}' met both similarity and score thresholds.") # Reduced verbosity
        return None

    def store(self, prompt: str, actions: ActionSequence) -> Optional[uuid.UUID]:
        """
        Generates an embedding for the prompt and stores the entry
        (embedding, prompt, actions, score, timestamps) in the ChromaDB collection.
        Returns the UUID of the stored entry, or None if embedding fails.
        """
        # print(f"DEBUG: store() called with prompt: '{prompt}'") # Reduced verbosity
        embedding = self._generate_embedding(prompt)
        if embedding is None:
            print(f"Error: Failed to generate embedding for prompt: '{prompt}'. Not storing.") # Keep error
            return None

        entry_id = uuid.uuid4()
        current_time = datetime.now(timezone.utc)
        initial_score = 1.0

        metadata = {
            "prompt_raw": prompt,
            "actions_json": json.dumps(actions), 
            "score": initial_score,
            "created_at_iso": current_time.isoformat(),
            "updated_at_iso": current_time.isoformat()
        }

        try:
            self._collection.add(
                ids=[str(entry_id)],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            # print(f"DEBUG: Successfully stored entry ID {entry_id} in ChromaDB.") # Reduced verbosity
            # print(f"  Prompt: '{prompt}'") # Reduced verbosity
            # print(f"  Actions: {actions}") # Reduced verbosity
            # print(f"  Metadata sent to Chroma: {metadata}") # Reduced verbosity
            return entry_id
        except Exception as e:
            print(f"Error storing entry ID {entry_id} in ChromaDB: {e}") # Keep error
            return None

    def update_reward(self, entry_id: uuid.UUID, success: bool) -> bool:
        """
        Updates the score of a cache entry in ChromaDB based on success/failure.
        If the score falls below a threshold, the entry is removed from ChromaDB.
        """
        # print(f"DEBUG: update_reward() called for entry_id: {entry_id}, success: {success}") # Reduced verbosity

        try:
            entry_data = self._collection.get(
                ids=[str(entry_id)],
                include=["metadatas"]
            )

            if not entry_data or not entry_data['ids'] or not entry_data['ids'][0]:
                # print(f"DEBUG: update_reward: Entry ID {entry_id} not found in ChromaDB.") # Reduced verbosity
                return False 
            
            current_metadata = entry_data['metadatas'][0]
            old_score = current_metadata.get("score", 0.0)
            if not isinstance(old_score, (float, int)):
                print(f"Warning: old_score for {entry_id} is not a number: {old_score}. Defaulting to 0.0 for EMA.") # Keep warning
                old_score = 0.0

            new_score = (REWARD_ALPHA * float(success)) + ((1 - REWARD_ALPHA) * old_score)
            # print(f"DEBUG: Entry {entry_id} score update: Old={old_score:.4f}, Success={success}, New={new_score:.4f}") # Reduced verbosity

            updated_metadata = current_metadata.copy()
            updated_metadata["score"] = new_score
            updated_metadata["updated_at_iso"] = datetime.now(timezone.utc).isoformat()

            if new_score < SCORE_THRESHOLD_EPSILON:
                # print(f"DEBUG: Entry {entry_id} new score ({new_score:.4f}) is below EPSILON ({SCORE_THRESHOLD_EPSILON}). Deleting from ChromaDB.") # Reduced verbosity
                self._collection.delete(ids=[str(entry_id)])
            else:
                # print(f"DEBUG: Updating entry {entry_id} in ChromaDB with new score: {new_score:.4f}") # Reduced verbosity
                self._collection.update(
                    ids=[str(entry_id)],
                    metadatas=[updated_metadata]
                )
            return True

        except Exception as e:
            print(f"Error during update_reward for entry ID {entry_id} in ChromaDB: {e}") # Keep error
            return False 