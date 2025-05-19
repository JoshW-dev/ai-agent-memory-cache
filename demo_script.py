from memory_cache import MemoryCache, ActionSequence

def run_demo():
    print("--- Basic MemoryCache Demo ---")
    cache = MemoryCache()

    # 1. Storing prompt-to-tool-call sequences
    prompt1 = "Make the player move faster"
    actions1: ActionSequence = ["increase_player_speed(factor=2)", "save_game_state()"]
    cache.store(prompt1, actions1)
    print(f"Stored: '{prompt1}' -> {actions1}")

    prompt2 = "Change sky color to blue"
    actions2: ActionSequence = ["set_sky_color(color='blue')", "render_scene()"]
    cache.store(prompt2, actions2)
    print(f"Stored: '{prompt2}' -> {actions2}")
    print("---")

    # 2. Retrieving a matching sequence
    print("Looking up an existing prompt:")
    retrieved_actions1 = cache.lookup(prompt1)
    if retrieved_actions1:
        print(f"Found for '{prompt1}': {retrieved_actions1}")
    else:
        print(f"No sequence found for '{prompt1}'")
    print("---")

    # 3. Attempting to retrieve a non-matching sequence
    prompt_non_existent = "Make the enemy stronger"
    print(f"Looking up a non-existent prompt: '{prompt_non_existent}'")
    retrieved_actions_non_existent = cache.lookup(prompt_non_existent)
    if retrieved_actions_non_existent:
        print(f"Found for '{prompt_non_existent}': {retrieved_actions_non_existent}")
    else:
        print(f"No sequence found for '{prompt_non_existent}'")
    print("---")

    # 4. Updating reward (conceptual)
    print("Updating reward for an existing prompt:")
    cache.update_reward(prompt1, success=True)
    cache.update_reward(prompt2, success=False)
    print("--- Demo Complete ---")

if __name__ == "__main__":
    run_demo() 