from memory_cache import MemoryCache
from dotenv import load_dotenv
import os

def main():
    # Load environment variables from .env file (for OPENAI_API_KEY)
    load_dotenv()
    print("Attempting to initialize MemoryCache with ChromaDB...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY not found in environment variables.")
        print("Please ensure your .env file is correctly set up with OPENAI_API_KEY.")
        return

    try:
        cache = MemoryCache()
        print("MemoryCache initialization with ChromaDB appears successful.")
        # You can add more specific checks here if needed, e.g.,
        # print(f"Chroma client: {cache._chroma_client}")
        # print(f"Chroma collection: {cache._collection}")
        # print(f"Collection count (should be 0 initially): {cache._collection.count()}")
    except Exception as e:
        print(f"Error during MemoryCache initialization: {e}")
        print("Please ensure ChromaDB was installed correctly (pip3 install -r requirements.txt) and that there are no conflicts.")

if __name__ == "__main__":
    main() 