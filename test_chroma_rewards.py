from memory_cache import MemoryCache, ActionSequence, SCORE_THRESHOLD_EPSILON, REWARD_ALPHA
from dotenv import load_dotenv
import os
import time
import uuid

def main():
    load_dotenv()
    print(f"--- Testing ChromaDB update_reward() method (EPSILON={SCORE_THRESHOLD_EPSILON}, ALPHA={REWARD_ALPHA}) ---")

    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY not found.")
        return

    cache = MemoryCache()
    # Using in-memory Chroma, so it's fresh each run.
    print(f"Initial collection count: {cache._collection.count()}")

    # 1. Store a new entry for eviction test
    print("\n--- Storing entry for eviction test ---")
    evict_prompt = "This prompt will be evicted soon."
    evict_actions: ActionSequence = ["action1", "action2"]
    evict_id = cache.store(evict_prompt, evict_actions)

    if not evict_id:
        print("Failed to store entry for eviction test. Aborting.")
        return
    print(f"Stored entry for eviction with ID: {evict_id}")
    
    # Verify initial score directly from Chroma (optional, but good for testing)
    try:
        entry_data = cache._collection.get(ids=[str(evict_id)], include=["metadatas"])
        initial_score = entry_data['metadatas'][0]['score']
        print(f"Initial score for {evict_id} from Chroma: {initial_score}")
    except Exception as e:
        print(f"Error fetching initial score for {evict_id}: {e}")

    # 2. Drive score down to trigger eviction
    print("\n--- Applying failures to trigger eviction ---")
    max_attempts = 10 # Safety break
    for i in range(max_attempts):
        print(f"Applying failure #{i+1} to entry {evict_id}")
        update_success = cache.update_reward(evict_id, success=False)
        if not update_success:
            print(f"  update_reward reported failure for {evict_id}, but could also mean item deleted.")
        
        # Check if entry still exists and its score
        current_entry_data = cache._collection.get(ids=[str(evict_id)], include=["metadatas"])
        if not current_entry_data or not current_entry_data['ids'] or not current_entry_data['ids'][0]:
            print(f"  Entry {evict_id} successfully evicted from ChromaDB after {i+1} failures.")
            break
        else:
            current_score = current_entry_data['metadatas'][0]['score']
            print(f"  Entry {evict_id} current score: {current_score:.4f}")
            if i == max_attempts - 1:
                print("Max attempts reached, eviction did not occur as expected within loop.")
        time.sleep(0.1)

    print(f"Collection count after eviction test: {cache._collection.count()}")

    # 3. Store another entry for score fluctuation test
    print("\n--- Storing entry for score fluctuation test ---")
    fluct_prompt = "This prompt will have its score changed."
    fluct_actions: ActionSequence = ["fluct_action"]
    fluct_id = cache.store(fluct_prompt, fluct_actions)

    if not fluct_id:
        print("Failed to store entry for fluctuation test. Skipping.")
    else:
        print(f"Stored entry for fluctuation test with ID: {fluct_id}")
        print(f"Applying success to {fluct_id}...")
        cache.update_reward(fluct_id, success=True)
        entry_data = cache._collection.get(ids=[str(fluct_id)], include=["metadatas"])
        print(f"  Score after 1st success: {entry_data['metadatas'][0]['score']:.4f}") # Should be 1.0

        print(f"Applying failure to {fluct_id}...")
        cache.update_reward(fluct_id, success=False)
        entry_data = cache._collection.get(ids=[str(fluct_id)], include=["metadatas"])
        print(f"  Score after 1 failure: {entry_data['metadatas'][0]['score']:.4f}") # 1.0*0.7 = 0.7

        print(f"Applying success to {fluct_id}...")
        cache.update_reward(fluct_id, success=True)
        entry_data = cache._collection.get(ids=[str(fluct_id)], include=["metadatas"])
        print(f"  Score after 1 failure, 1 success: {entry_data['metadatas'][0]['score']:.4f}") # 0.3*1 + 0.7*0.7 = 0.3+0.49 = 0.79

        print(f"Applying failure to {fluct_id}...")
        cache.update_reward(fluct_id, success=False)
        entry_data = cache._collection.get(ids=[str(fluct_id)], include=["metadatas"])
        print(f"  Score after 1f,1s,1f: {entry_data['metadatas'][0]['score']:.4f}") # 0.7*0.79 = 0.553

    print(f"Final collection count: {cache._collection.count()}")
    print("\n--- Test complete ---")

if __name__ == "__main__":
    main() 