# test_store_v2.py
from dotenv import load_dotenv
load_dotenv() # Load .env file for OPENAI_API_KEY

from memory_cache import MemoryCache, ActionSequence # Ensure ActionSequence is imported if used explicitly for type hints here
import os

print("--- Testing new store() method (P2-T2) ---")

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
else:
    cache = MemoryCache()
    
    prompt1 = "Make the player character jump higher."
    actions1: ActionSequence = ["modify_jump_force(multiplier=1.5)", "play_jump_sound()"]
    
    print(f"\nCalling store for: \"{prompt1}\"")
    entry_id1 = cache.store(prompt1, actions1)
    
    if entry_id1:
        print(f"Store returned ID: {entry_id1}")
        # Let's inspect the cache (for debugging/testing purposes)
        if cache._cache: # Check if cache is not empty
            print("Current cache content (first item):")
            # Ensure to pretty print or handle dict display
            first_entry = cache._cache[0]
            print(f"  ID: {first_entry['id']}")
            print(f"  Prompt: {first_entry['prompt_raw']}")
            print(f"  Actions: {first_entry['actions']}")
            print(f"  Score: {first_entry['score']}")
            print(f"  Created At: {first_entry['created_at']}")
            print(f"  Updated At: {first_entry['updated_at']}")
            if 'embedding' in first_entry and first_entry['embedding']:
                 print(f"  Embedding stored, dimension: {len(first_entry['embedding'])}, first 5 values: {first_entry['embedding'][:5]}...")
            else:
                print("  Embedding not found or empty in stored entry.")
        else:
            print("Cache is empty after store attempt.")
    else:
        print("Store operation failed (likely embedding generation).")

    print("\n--- Test complete ---") 