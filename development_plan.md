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

### Phase 7: Agent Tool Selection Refactor & Enhanced Testing (Current Focus)

**Goal:** Refine the agent's tool selection to be primarily driven by semantic similarity between the user prompt and tool descriptions/names. Improve agent output clarity and add targeted backend tests.

| Task ID | Description                                                                      | Status      | Notes                                                                                                                                                              |
|---------|----------------------------------------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| P7-T1   | **Modify `CapturingAgent` for Embedding-Based Tool Selection**                     | In Progress | Agent to embed tools (name+desc) on init. `run()` will find best tool via similarity to prompt embedding. Threshold: 0.6.                                        |
| P7-T2   | **LLM Role Shift: Tool Input Generation**                                          | Not Started | If tool selected by similarity, LLM's role is to generate the tool's input string based on user prompt and tool desc, not to select the tool itself.              |
| P7-T3   | **Handle No Tool Match: LLM Direct Answer**                                        | Not Started | If no tool meets similarity threshold, LLM generates a direct answer to the user's prompt.                                                                         |
| P7-T4   | **Update `ActionSequence` Format & Agent History**                                 | Not Started | `ActionSequence` to include tool selection similarity score if a tool is used. E.g., "Tool: <name> (Similarity: <score>), Input: <input>, Observation: <obs>".      |
| P7-T5   | **Enhance `app.py` Output for Agent Responses**                                    | Not Started | Display tool selection similarity in Streamlit UI when agent uses a tool (cache miss scenario).                                                                      |
| P7-T6   | **Create New Terminal-Based Tests for Tool Selection**                             | Not Started | New Python test script (e.g., `test_agent_tool_selection.py`) to verify: tool choice by similarity, score reporting, LLM input generation, fallback to direct answer, dynamic tool creation. |
| P7-T7   | **Update `README.md` for Phase 7 Changes**                                       | Not Started | Reflect new agent logic, `ActionSequence` format, and testing strategy in main README.                                                                               |

### Phase 8: Advanced Feedback & Self-Healing for Tool Selection (Conceptualized)

**Goal:** Implement a robust feedback mechanism that allows the agent to learn from user input on tool selection, improving accuracy and enabling self-healing of its tool preferences.

| Task ID | Description                                                                      | Status      | Notes                                                                                                                                                                                              |
|---------|----------------------------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| P8-T1   | **Agent Internal Tool Representation for Feedback**                                | Not Started | Modify `BaseTool` or agent's tool store to allow each tool to maintain a list of 'additional representative embeddings' derived from upvoted user prompts.                                     |
| P8-T2   | **Implement Upvote Logic for Tool Selection**                                      | Not Started | On user upvote for a chosen_tool/user_prompt pair: generate embedding for user_prompt, add it to chosen_tool's additional representative embeddings. Re-evaluate effective similarity for future. |
| P8-T3   | **Integrate Agent Tool Feedback with Cache Feedback**                              | Not Started | If a cached sequence (referencing a tool) is upvoted, apply agent-level tool upvote logic for the original prompt & tool. If downvoted, cache score penalizes the sequence.                      |
| P8-T4   | **Refine Agent's `run` Method for Multi-Embedding Tool Matching**                   | Not Started | Agent's tool selection to compare user prompt embedding against both primary tool embedding and all additional representative embeddings for each tool, taking the max similarity.          |
| P8-T5   | **Handle Downvotes for Tool Selection (Implicit Learning)**                        | Not Started | Downvoted tool-prompt pairs are not reinforced. Correct choices (via alternative existing tools or newly created ones) get upvoted, naturally out-competing prior incorrect choices.        |
| P8-T6   | **Update `app.py` / Demo for Tool Feedback**                                     | Not Started | UI to allow feedback specifically tied to the agent's tool choice when applicable (especially on cache miss).                                                                          |
| P8-T7   | **Testing Suite for Feedback Loop**                                                | Not Started | Create backend tests to simulate upvotes/downvotes and verify that the agent's tool preferences adapt correctly over several interactions with similar prompts.                             |
| P8-T8   | **Update `README.md` and `development_plan.md` for Phase 8**                     | Not Started | Document the new feedback mechanisms, learning strategy, and updated agent behavior.                                                                                                 |

## Environment Variables

- `OPENAI_API_KEY`: (Optional, if we decide to use OpenAI embeddings later).

Let me know when you're ready to provide the `OPENAI_API_KEY` if we decide to use OpenAI models directly for the mock agent part. For now, sentence-transformers will be our primary focus for embeddings. 