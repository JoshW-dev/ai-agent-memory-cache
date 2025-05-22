from memory_cache import MemoryCache, ActionSequence
from dotenv import load_dotenv
import os
import uuid

def main():
    load_dotenv()
    print("--- Testing ChromaDB store() method --- D")

    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY not found.")
        return

    cache = MemoryCache()

    test_prompt = "How to make a character jump in Unity?"
    test_actions: ActionSequence = [
        "Open PlayerController.cs script",
        "Add Rigidbody2D component to Player GameObject",
        "In Update(), check for Input.GetKeyDown(KeyCode.Space)",
        "If space pressed, apply upward force: rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse)"
    ]

    print(f"Attempting to store prompt: '{test_prompt}'")
    entry_id = cache.store(test_prompt, test_actions)

    if entry_id:
        print(f"Successfully called store(). Entry ID: {entry_id}")
        # To verify, we can also check the collection count if we want more rigor
        # print(f"Current collection count: {cache._collection.count()}")
    else:
        print("Failed to store entry. Check logs for errors.")
    
    # Try storing another one to see if IDs are unique and count increases
    test_prompt_2 = "Set background to red"
    test_actions_2: ActionSequence = ["Camera.main.backgroundColor = Color.red;"]
    print(f"\nAttempting to store prompt: '{test_prompt_2}'")
    entry_id_2 = cache.store(test_prompt_2, test_actions_2)
    if entry_id_2:
        print(f"Successfully called store(). Entry ID: {entry_id_2}")
        # print(f"Current collection count after second store: {cache._collection.count()}")
    else:
        print("Failed to store second entry.")


if __name__ == "__main__":
    main() 