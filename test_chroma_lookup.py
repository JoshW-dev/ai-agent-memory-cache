from memory_cache import MemoryCache, ActionSequence, SIMILARITY_THRESHOLD_TAU
from dotenv import load_dotenv
import os
import time

def main():
    load_dotenv()
    print(f"--- Testing ChromaDB lookup() method (TAU={SIMILARITY_THRESHOLD_TAU}) ---")

    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY not found.")
        return

    cache = MemoryCache()
    # It's good practice to clear the collection for a clean test run, if possible,
    # or use a test-specific collection name.
    # For now, we assume the collection might have data from previous tests or runs.
    # To ensure a clean state for *this test*, you might manually delete ./chroma_db_data if using PersistentClient
    # or handle collection cleanup if ChromaDB client API supports it easily.
    # For an in-memory client, it's fresh each time. Current code uses in-memory.
    print(f"Initial collection count: {cache._collection.count()}")

    # 1. Store some entries
    print("\n--- Storing entries ---")
    prompts_actions = {
        "p1": ("How do I change the player speed?", ["set_player_attribute('speed', 10)"]),
        "p2": ("Make the skybox bright red.", ["set_sky_color('red')", "render_effects()"]),
        "p3": ("What is the health of the goblin chief?", ["get_npc_health('goblin_chief')"])
    }
    stored_ids = {}
    for key, (prompt, actions) in prompts_actions.items():
        entry_id = cache.store(prompt, actions)
        if entry_id:
            print(f"  Stored '{prompt}' with ID {entry_id}")
            stored_ids[key] = entry_id
        else:
            print(f"  Failed to store '{prompt}'")
        time.sleep(0.5) # Avoid hitting API rate limits if any
    
    print(f"Collection count after storing: {cache._collection.count()}")

    # 2. Perform lookups
    print("\n--- Performing lookups ---")
    test_lookups = [
        # Exact matches
        "How do I change the player speed?", 
        "Make the skybox bright red.",
        # Semantic matches
        "How can I alter player velocity?", # Similar to p1
        "I want a crimson sky.",           # Similar to p2
        # Semantic miss / different topic
        "Spawn a new enemy.",
        # Checking one of the original prompts that should be a clear hit
        "What is the health of the goblin chief?",
    ]

    for lookup_prompt in test_lookups:
        print(f"\n--- LOOKING UP: '{lookup_prompt}' ---")
        lookup_result = cache.lookup(lookup_prompt)
        
        if lookup_result:
            entry_id, actions_found = lookup_result
            print(f"  >> HIT! Entry ID: {entry_id}, Actions: {actions_found}")
        else:
            print(f"  >> MISS.")
        time.sleep(0.5) # Avoid hitting API rate limits

    print("\n--- Test complete ---")

if __name__ == "__main__":
    main() 