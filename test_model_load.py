# test_model_load.py
from dotenv import load_dotenv
load_dotenv() # Load variables from .env file

from memory_cache import MemoryCache
import os

print("Attempting to create MemoryCache instance and test OpenAI embedding...")

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    print("Please set it before running the test: export OPENAI_API_KEY=\"your_key\"")
else:
    try:
        print("Initializing MemoryCache...")
        cache = MemoryCache()
        print("MemoryCache instance created.")
        
        test_sentence = "This is a test sentence for OpenAI embeddings."
        print(f"Attempting to generate embedding for: '{test_sentence}'")
        # Use the internal helper method directly for this test, though typically it's for internal class use
        test_embedding = cache._generate_embedding(test_sentence)
        
        if test_embedding:
            print(f"OpenAI embedding successful. Embedding vector starts with: {test_embedding[:5]}...")
            print(f"Embedding dimension: {len(test_embedding)}") # OpenAI text-embedding-3-small is 1536
        else:
            print("Failed to generate OpenAI embedding. Check API key and network.")
            
    except Exception as e:
        print(f"An error occurred during the test: {e}") 