# Development Plan: AI Agent Memory Cache

This document tracks the development process for the AI Agent Memory Cache project.

## Core Objective

Build a memory-cache layer that remembers the exact sequence of tool calls an LLM agent used to satisfy a user request, allowing similar requests to be fulfilled instantly. The cache must support reward/penalty updates for self-healing.

## Phases & Tasks

We'll follow an iterative approach, starting simple and adding complexity, testing at each stage.
Constants for configuration (as per README.md, can be class members or global):
- τ (similarity threshold, e.g., 0.85)
- α (EMA factor for reward, e.g., 0.3)
- ε (eviction score threshold, e.g., 0.2)
- top_k (for retrieval, e.g., 3)

### Phase 1: Basic Cache Implementation (Option A - Required Minimum) - COMPLETED

| Task ID | Description                                                                 | Status      | Notes                                                                                     |
|---------|-----------------------------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------|
| P1-T1   | **Create `MemoryCache` Class Structure**                                    | Done        | Define the class with `lookup`, `store`, and `update_reward` method signatures.            |
| P1-T2   | **Implement In-Memory Storage**                                             | Done        | Used a simple Python dictionary for initial storage (prompt -> action_sequences).           |
| P1-T3   | **Implement `store()` Method**                                              | Done        | Basic storage of prompt and a list of action strings.                                     |
| P1-T4   | **Implement `lookup()` Method (Exact Match)**                               | Done        | Initial lookup based on exact prompt string matches, returning list of action strings.      |
| P1-T5   | **Implement `update_reward()` Method (Conceptual)**                         | Done        | Placeholder reward logic; printed feedback.                                               |
| P1-T6   | **Define `ActionSequence` Type**                                            | Done        | Represented as `List[str]`. Each string is a simplified tool call output.                 |
| P1-T7   | **Basic Demonstration Script**                                              | Done        | Script stored 2 sequences, retrieved one, showed reward update.                             |
| P1-T8   | **Manual Testing for Phase 1**                                              | Done        | Verified `store`, `lookup` (exact), and `update_reward` prints via `demo_script.py`.        |

### Phase 2: Similarity-Based Lookup & Rewards (Option B Elements)

**Goal:** Enhance cache with semantic search using embeddings and implement full reward logic. Cache remains in-memory for this phase.

| Task ID | Description                                                                 | Status      | Notes                                                                                                                                                             |
|---------|-----------------------------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| P2-T1   | **Integrate OpenAI Embeddings**                                             | Done        | Added `openai` & `python-dotenv` to `requirements.txt`. Initialized `OpenAI()` client in `MemoryCache.__init__` using `text-embedding-3-small`. API key via `.env`. |
| P2-T2   | **Evolve In-Memory Data Structure & `store()`**                             | Done        | Modified `self._cache` to `List[CacheEntry]`. `store()` generates embedding, creates full `CacheEntry` dict, appends to list. `uuid` & `datetime` imported and used. |
| P2-T3   | **Implement Similarity `lookup()`**                                         | Done        | Input `prompt: str`. Generate its embedding. Iterated `self._cache`, calculated cosine similarity. Found best entry >= τ (0.70). Ignores entries with score < ε. |
| P2-T4   | **Implement Full `update_reward()` Logic**                                  | Done        | Input `entry_id: UUID, success: bool`. Find entry in `self._cache` by `id`. Update `score` using EMA: `score_new = α*float(success) + (1‑α)*score_old`. Update `updated_at`. If `score_new < ε`, remove entry from `self._cache`. Define REWARD_ALPHA. |
| P2-T5   | **Manual Testing: Phase 2.**                                                | Done        | Ran `demo_phase2.py`. Verified exact and semantic matches work with TAU=0.70. Verified score updates (EMA) and eviction based on score < EPSILON work correctly. |

### Phase 2: Complete

### Phase 3: Vector DB Integration & Agent Simulation (Option B Elements)

**Goal:** Replace in-memory vector search with ChromaDB and simulate an agent using the cache.

| Task ID | Description                                                                 | Status      | Notes                                                                                                                                                                                            |
|---------|-----------------------------------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| P3-T1   | **Research: Vector Databases.**                                             | Done        | Identified ChromaDB, FAISS, LanceDB. Selected ChromaDB for its ease of use, Python client, and local persistence capabilities.                                                                     |
| P3-T2   | **Integrate ChromaDB**                                                      | Done        | Added `chromadb` to `requirements.txt`, installed. In `MemoryCache.__init__`, initialized `chromadb.Client()` and collection `memory_cache_collection`. Verified with `test_chroma_init.py`.      |
| P3-T3   | **Modify `store()` for ChromaDB**                                           | Done        | `store()` generates embedding, prepares metadata (actions as JSON string, datetimes as ISO strings), and adds entry to ChromaDB collection. Verified with `test_chroma_store.py`.               |
| P3-T4   | **Modify `lookup()` for ChromaDB**                                          | Done        | `lookup()` queries ChromaDB (top_k, includes metadata & distances), converts distance to similarity, checks against TAU (0.60) & EPSILON. Parses actions_json. Verified with `test_chroma_lookup.py`. |
| P3-T5   | **Modify `update_reward()` for ChromaDB**                                   | Done        | `update_reward()` gets entry from Chroma, calculates new score (EMA), then updates metadata in ChromaDB or deletes if score < EPSILON. Verified with `test_chroma_rewards.py`.                     |
| P3-T6   | **Create Mock Agent Script (`mock_agent_demo.py`)**                         | Done        | Script takes user prompt. Calls `cache.lookup()`. On HIT: print actions. On MISS: print "reasoning", generate dummy `ActionSequence`, call `cache.store()`. Verified with interactive session. |
| P3-T7   | **Integrate Rewards into Mock Agent**                                       | Done        | `mock_agent_demo.py` updated: `lookup()` now returns `(id, actions)`. After HIT or MISS+STORE, script asks for y/n feedback and calls `cache.update_reward()` with the relevant ID.              |
| P3-T8   | **Demonstrate Self-Healing in Mock Agent**                                  | Done        | Ran `mock_agent_demo.py`, repeatedly marked a cached entry as failed until its score dropped below epsilon, verified it was evicted, and a new plan was stored on subsequent identical prompt. Log verbosity reduced in `memory_cache.py`. |
| P3-T9   | **Manual Testing for Phase 3**                                              | Done        | Comprehensive testing via `mock_agent_demo.py` (including hit, miss, store, reward, eviction, self-healing) and earlier specific ChromaDB method tests (`test_chroma_*.py`) confirms Phase 3 functionality. |

### Phase 3: Complete

### Phase 4: Final Touches & Documentation

| Task ID | Description                               | Status      | Notes                                                        |
|---------|-------------------------------------------|-------------|--------------------------------------------------------------|
| P4-T1   | **Code Cleanup & Refinements**              | Done        | Reviewed `memory_cache.py` and demo scripts. Code is generally clean, well-commented, and type-hinted. Log verbosity was reduced. No major outstanding cleanup needed. |
| P4-T2   | **Finalize `README.md` (Main)**             | Done        | README updated with implemented features, project structure, setup/usage instructions, revised data model, algorithms, API, and testing strategy. |
| P4-T3   | **Consider Interactive Chat CLI for Demo**  | Skipped     | Current `input()` loop in `mock_agent_demo.py` is sufficient for demonstration purposes. Advanced CLI is out of scope for this phase. |
| P4-T4   | **Prepare Submission Package**            | Done        | User confirmed .env.example creation. All files ready. Project Complete pending this new phase. |

### Phase 5: Advanced Mock Agent Integration (mpaepper/llm_agents)

**Goal:** Enhance the mock agent demo by integrating a more structured agent based on the mpaepper/llm_agents library.

| Task ID | Description                                                         | Status      | Notes                                                                                                                            |
|---------|---------------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------|
| P5-T1   | **Vendor `mpaepper/llm_agents` Library**                              | Done        | `llm_module` created with `agent.py`, `llm.py`, and `tools/base.py` (and `__init__.py` files) from `mpaepper/llm_agents`.          |
| P5-T2   | **Update `llm_module.llms.ChatLLM` for OpenAI v1+ API**               | Done        | `llm_module/llm.py` updated and successfully tested.                                                                       |
| P5-T3   | **Define Custom Placeholder Tools**                                   | Done        | `llm_module/custom_tools.py` created with `WeatherTool`, `InventoryCheckTool`, `MessageHandlerTool`. Base tool class corrected. Successfully tested. |
| P5-T4   | **Create `CapturingAgent` to Extract Tool Usage History**             | Done        | `llm_module/capturing_agent.py` created and successfully tested. Agent uses LLM, selects tools, and captures history.               |

### Phase B: Integrating `CapturingAgent` with `MemoryCache` in `mock_agent_demo.py`

| Task ID | Description                                                      | Status      | Notes                                                                                                                             |
|---------|------------------------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------|
| P5-B1   | **Update `requirements.txt` (check for `tiktoken`)**               | Done        | `tiktoken` added to `requirements.txt` and user confirmed installation.                                                           |
| P5-B2   | **Modify `mock_agent_demo.py` to use `CapturingAgent`**            | Done        | `mock_agent_demo.py` successfully uses `CapturingAgent`. Cache miss/hit, agent tool use, history capture, storage, and reward feedback all tested and working. |

### Phase C: Testing and Refinement (for `CapturingAgent` integration)

| Task ID | Description                                                                 | Status      | Notes                                                                                                |
|---------|-----------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------|
| P5-C1   | **Comprehensive End-to-End Testing of `mock_agent_demo.py`**                | Done        | Tested cache miss/hit, agent tool use, self-healing (eviction), and direct answers. Confirmed by user. |
| P5-C2   | **Review and Refine `CapturingAgent` & `mock_agent_demo.py` (Initial Pass)**  | Done        | Log verbosity, error handling (basic), prompt engineering (default used). User feedback incorporated.|
| P5-C3   | **Update `README.md` to Reflect Phase 5 Changes**                         | Done        | `README.md` updated to accurately reflect the `CapturingAgent`, `llm_module`, and enhanced demo functionality from Phase 5. |

### Phase 6: UI/UX Enhancements & Advanced Features

**Goal:** Improve the user experience with a web frontend and refine core agent/cache mechanisms.

| Task ID | Description                                                                 | Status      | Notes                                                                                                                                 |
|---------|-----------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------|
| P6-T1   | **Research & Select Frontend Framework**                                      | Done        | Selected Streamlit for its ease of use and Python-native UI development. User installed Streamlit.                                  |
| P6-T2   | **Develop Basic Chatbot Frontend**                                            | Done        | Created `app.py` with a basic Streamlit chat interface. User confirmed. *To refine: Improve clarity of agent response display.*         |
| P6-T3   | **Integrate Frontend with `MemoryCache` and `CapturingAgent`**                | Done        | `app.py` now uses `MemoryCache` & `CapturingAgent`. Basic reward UI (buttons) implemented. *To refine: Display similarity score on HIT.* |
| P6-T4   | **Refine Reward System UX**                                                   | In Progress | Current: Thumbs up/down buttons in UI. Confirm if this meets simplicity goal or if other UX elements are desired.                   |
| P6-T5   | **Agent Polish: Improve `CapturingAgent` Robustness & Tool Relevance**        | In Progress | Enhance error handling, refine prompts. *Priority: Update `custom_tools.py` with video game specific tools.*                      |
| P6-T6   | **Advanced Reward System (Conceptual/Implementation)**                        | Not Started | Explore more nuanced reward signals beyond simple y/n. E.g., partial success, or feedback on specific tool calls.                  |
| P6-T7   | **Documentation Update for New Features**                                     | Not Started | Update README and other relevant docs for the new frontend and any advanced features.                                               |

## Environment Variables

- `OPENAI_API_KEY`: (Optional, if we decide to use OpenAI embeddings later).

Let me know when you're ready to provide the `OPENAI_API_KEY` if we decide to use OpenAI models directly for the mock agent part. For now, sentence-transformers will be our primary focus for embeddings. 