# demo_phase2.py
from dotenv import load_dotenv
load_dotenv() # Load .env file for OPENAI_API_KEY

from memory_cache import MemoryCache, ActionSequence, SIMILARITY_THRESHOLD_TAU, SCORE_THRESHOLD_EPSILON, REWARD_ALPHA
import os
import time
import uuid

print(f"--- Phase 2 Demo: Similarity Lookup & Rewards (TAU={SIMILARITY_THRESHOLD_TAU}, EPSILON={SCORE_THRESHOLD_EPSILON}, ALPHA={REWARD_ALPHA}) ---")

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
else:
    cache = MemoryCache()
    entry_ids = {}

    # --- Storing initial entries ---
    print("\n--- Storing entries ---")
    prompts_actions = {
        "p1": ("Make the player character move faster.", ["set_player_speed(10)", "show_speed_effect()"]),
        "p2": ("Change the sky color to a deep blue.", ["set_skybox_color('deep_blue')", "update_lighting()"]),
        "p3": ("Increase the main enemy's health points.", ["target_enemy('boss_1').set_health(500)"]),
        "p4": ("Let the user's avatar sprint quickly.", ["enable_sprint_mode()", "set_player_speed(15)"])
    }

    for key, (p, a) in prompts_actions.items():
        entry_id = cache.store(p, a)
        if entry_id:
            entry_ids[key] = entry_id
        time.sleep(0.5)
    
    print(f"\nStored entry IDs: {entry_ids}")

    # --- Performing lookups (as before) ---
    print("\n--- Performing lookups ---")
    test_prompts = [
        "Make the player character move faster.", 
        "I want the avatar to sprint.",          
        "Render the heavens in an azure hue.",   
        "Give the boss more HP.",                
        "Make the background music louder.",     
        "How to make the player run?",           
        "Adjust skybox to be dark blue."
    ]
    for tp in test_prompts:
        print(f"\n--- LOOKING UP: '{tp}' ---")
        actions = cache.lookup(tp)
        if actions:
            print(f"  >> HIT! Actions: {actions}")
        else:
            print(f"  >> MISS.")
        time.sleep(0.5)

    # --- Testing Rewards and Eviction ---
    print("\n\n--- Testing Rewards and Eviction ---")
    
    # Test on p1: "Make the player character move faster."
    p1_id = entry_ids.get("p1")
    if p1_id:
        print(f"\nTargeting entry for '{prompts_actions['p1'][0]}' (ID: {p1_id})")
        
        print("\nSimulating a series of failures for p1...")
        initial_cache_size = len(cache._cache)
        current_score = 1.0 # Assuming initial score is 1.0
        
        # How many failures to drop score from 1.0 to < 0.2 with alpha = 0.3?
        # Iteration 1 (fail): new_score = 0.3*0 + 0.7*1.0 = 0.7
        # Iteration 2 (fail): new_score = 0.3*0 + 0.7*0.7 = 0.49
        # Iteration 3 (fail): new_score = 0.3*0 + 0.7*0.49 = 0.343
        # Iteration 4 (fail): new_score = 0.3*0 + 0.7*0.343 = 0.2401
        # Iteration 5 (fail): new_score = 0.3*0 + 0.7*0.2401 = 0.16807 (This should trigger eviction)

        for i in range(5):
            print(f"Applying failure #{i+1} to p1 (ID: {p1_id})")
            cache.update_reward(p1_id, success=False)
            # Find the entry to print its current score, if it still exists
            p1_entry_after_update = next((e for e in cache._cache if e["id"] == p1_id), None)
            if p1_entry_after_update:
                print(f"  p1 current score: {p1_entry_after_update['score']:.4f}")
            else:
                print(f"  p1 (ID: {p1_id}) appears to have been evicted.")
                break # Stop if evicted
            time.sleep(0.2)

        print(f"\nCache size before p1 eviction test: {initial_cache_size}")
        print(f"Cache size after p1 eviction test: {len(cache._cache)}")

        print(f"\nAttempting to lookup evicted prompt: '{prompts_actions['p1'][0]}' one more time...")
        actions_after_eviction = cache.lookup(prompts_actions['p1'][0])
        if actions_after_eviction:
            print(f"  >> ERROR: Still HIT! Actions: {actions_after_eviction}")
        else:
            print(f"  >> Correctly MISS after expected eviction.")

        # Test a successful update on another prompt (e.g., p2)
        p2_id = entry_ids.get("p2")
        if p2_id:
            print(f"\nApplying a success to p2 ('{prompts_actions['p2'][0]}', ID: {p2_id})")
            cache.update_reward(p2_id, success=True)
            p2_entry_after_update = next((e for e in cache._cache if e["id"] == p2_id), None)
            if p2_entry_after_update:
                 print(f"  p2 new score: {p2_entry_after_update['score']:.4f}") # Initial 1.0 -> 0.3*1 + 0.7*1 = 1.0 (no change if already 1 and success)
                                                                            # If it was < 1.0, it would increase.
                                                                            # Let's try one fail then one success to see change:
            cache.update_reward(p2_id, success=False) # score = 0.7*1 = 0.7
            p2_entry_after_update = next((e for e in cache._cache if e["id"] == p2_id), None)
            print(f"  p2 score after 1 fail: {p2_entry_after_update['score']:.4f}")
            cache.update_reward(p2_id, success=True) # score = 0.3*1 + 0.7*0.7 = 0.3 + 0.49 = 0.79
            p2_entry_after_update = next((e for e in cache._cache if e["id"] == p2_id), None)
            print(f"  p2 score after 1 fail then 1 success: {p2_entry_after_update['score']:.4f}")
    else:
        print("Could not find p1_id to test rewards.")

    print("\n--- Phase 2 Demo Complete ---") 