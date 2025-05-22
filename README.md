# AI Agent Memory Cache

The memory cache system provides a fast retrieval method for LLM agent tool sequences, enabling repeated requests to be fulfilled instantly without re-planning by using a vector database for similarity search and a reward mechanism for self-healing. The system now features a more sophisticated mock agent that uses LLM-driven reasoning and actual (placeholder) tool calls.

## System Diagrams

### Component Architecture
![Memory Cache Component Diagram](./diagrams/component_diagram.svg)

### Data Flow
![Memory Cache Data Flow Diagram](./diagrams/data_flow_diagram.svg)

## 1. Objective

Build a memory-cache layer that remembers the sequence of tool calls an LLM agent used to satisfy a user request. This allows repeated or semantically similar requests to be fulfilled instantly by replaying stored actions, bypassing expensive re-planning. The cache supports reward/penalty updates, enabling the agent to learn from feedback and adapt if a stored plan becomes obsolete or incorrect.

## 2. Background & Motivation

LLM agents often perform multi-step reasoning and tool use. The initial planning phase can be slow and costly. Caching successful action sequences allows the agent to:
1.  Instantly retrieve and replay actions for identical or very similar prompts.
2.  Learn from feedback: If a cached sequence fails, the agent can penalize the stored sequence, re-plan (using its LLM and tools), and cache the new, successful sequence.

## 3. Core Features Implemented

*   **Semantic Cache Hits:** Looks up prompts based on semantic similarity using OpenAI embeddings (`text-embedding-3-small`).
*   **Vector Store:** Utilizes ChromaDB (in-memory by default) for storing and querying prompt embeddings and associated metadata.
*   **Reward-Based Updates:** Agent actions are reinforced or penalized based on user feedback (`y/n`), updating a score for each cached entry.
*   **Self-Healing/Eviction:** Entries with scores falling below a defined threshold (`SCORE_THRESHOLD_EPSILON`) are automatically evicted.
*   **Interactive Demo with `CapturingAgent`:** A `mock_agent_demo.py` script showcases the cache's functionality. On a cache miss, it employs a `CapturingAgent` (from the `llm_module`) which:
    *   Uses an LLM (OpenAI's GPT model) for reasoning.
    *   Selects from a predefined set of placeholder tools (e.g., Weather, Inventory).
    *   Executes the chosen tools and processes their (simulated) observations.
    *   Captures the sequence of tool interactions (tool name, input, observation) or a direct LLM answer if no tools are used. This sequence is then stored in the cache.

## 4. Project Structure

```
├── README.md             # This file
├── development_plan.md   # Tracks tasks and progress
├── diagrams/             # SVG diagrams
│   ├── component_diagram.svg
│   └── data_flow_diagram.svg
├── llm_module/           # Core agent logic based on mpaepper/llm_agents
│   ├── __init__.py
│   ├── agent.py          # Base Agent class
│   ├── capturing_agent.py # Agent that captures tool usage history
│   ├── custom_tools.py   # Defines placeholder tools (Weather, Inventory, etc.)
│   ├── llm.py            # ChatLLM class (OpenAI API wrapper)
│   └── tools/
│       ├── __init__.py
│       └── base.py       # Base Tool class
├── memory_cache.py       # Core MemoryCache class implementation
├── mock_agent_demo.py    # Interactive script to demo cache with CapturingAgent
├── requirements.txt      # Python dependencies
├── .env.example          # Example for environment variables (OpenAI API Key)
└── test_*.py             # Individual test scripts for MemoryCache functionalities
```
*(Note: Components within `llm_module` also have `if __name__ == '__main__':` blocks for direct testing.)*

## 5. Setup and Usage

### Prerequisites
*   Python 3.10+
*   An OpenAI API Key

### Installation
1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up OpenAI API Key:**
    *   Rename `.env.example` to `.env`.
    *   Open the `.env` file and add your OpenAI API key:
        ```
        OPENAI_API_KEY="your_openai_api_key_here"
        ```

### Running the Demo
Execute the mock agent demo script:
```bash
python3 mock_agent_demo.py
```
Interact with the agent by typing prompts.
*   On a **cache miss**, the `CapturingAgent` will use its LLM and tools to determine actions. The sequence of tool interactions (or a direct LLM answer) will be displayed and then stored in the cache.
*   On a **cache hit**, the stored action sequence is retrieved and displayed.
After each sequence (retrieved or newly generated), you'll be asked if it worked ("y/n"). This feedback drives the reward mechanism. Type "quit" or "exit" to end the demo.

## 6. Glossary (Implemented)

-   **Prompt:** Original user message string.
-   **ActionSequence:** An ordered `List[str]`. Each string represents one step in the agent's process:
    *   If a tool was used: `"Tool: <tool_name>, Input: '<tool_input>', Observation: '<observation>'"`.
    *   If the agent answered directly: `"Direct Answer: <final_answer_from_llm>"`.
-   **Embedding:** A vector representation of a prompt, generated by OpenAI's `text-embedding-3-small` model.
-   **Reward Score:** A floating-point metric (typically 0.0 to 1.0) associated with each cached entry, updated via EMA based on feedback.
-   **ChromaDB:** An open-source embedding database used to store prompts, their embeddings, and associated metadata including actions and scores.
-   **`llm_module`:** A directory containing the core agent logic, including the `CapturingAgent`, `ChatLLM` wrapper, and placeholder tool definitions, inspired by the `mpaepper/llm_agents` library.
-   **`CapturingAgent`:** An agent class that uses an LLM for reasoning, selects tools from a predefined set, executes them, and captures the history of these interactions (tool name, input, observation).

## 7. Data Model (as stored in ChromaDB metadata)

| Metadata Key      | Type (in Python) | Notes                                                                 |
|-------------------|------------------|-----------------------------------------------------------------------|
| `prompt_raw`      | `str`            | The original prompt string.                                           |
| `actions_json`    | `str`            | A JSON string representation of the `ActionSequence` (list of strings reflecting tool use or direct answers). |
| `score`           | `float`          | Starts at 1.0, updated via EMA based on feedback.                     |
| `created_at_iso`  | `str`            | ISO 8601 timestamp of when the entry was first created.               |
| `updated_at_iso`  | `str`            | ISO 8601 timestamp of when the entry was last updated (score change).   |
_(The embedding itself is stored directly in ChromaDB alongside a unique UUID string as the ID.)_

## 8. Core Algorithms

### 8.1 Similarity-Based Lookup
1.  Generate embedding for the input `prompt` using OpenAI.
2.  Query ChromaDB with the prompt's embedding to find the `TOP_K_RESULTS` (e.g., 3) most similar entries.
3.  For each result, calculate similarity (1 - distance, as ChromaDB returns cosine distance).
4.  If a result's similarity ≥ `SIMILARITY_THRESHOLD_TAU` (e.g., 0.60) AND its `score` ≥ `SCORE_THRESHOLD_EPSILON` (e.g., 0.2):
    *   Return the `entry_id` and its `ActionSequence` (parsed from `actions_json`).
5.  If no such entry is found, return `None` (cache miss).

### 8.2 Reward Update & Eviction
1.  Given an `entry_id` and `success` (boolean feedback):
2.  Fetch the entry's current metadata (especially `score`) from ChromaDB.
3.  Calculate `new_score = (REWARD_ALPHA * float(success)) + ((1 - REWARD_ALPHA) * old_score)`.
    *   `REWARD_ALPHA` (e.g., 0.3) is the learning rate for the Exponential Moving Average.
4.  If `new_score < SCORE_THRESHOLD_EPSILON` (e.g., 0.2):
    *   Delete the entry from ChromaDB.
5.  Else:
    *   Update the entry in ChromaDB with the `new_score` and current timestamp for `updated_at_iso`.

### 8.3 Self-Healing Example Flow
1.  User prompt -> Cache MISS. The `CapturingAgent` runs (uses LLM, selects tools like `WeatherTool("Paris")`, gets observation), generates `ActionSequence A` (e.g., `["Tool: WeatherLookup, Input: 'Paris', Observation: '...sunny...'"]`). This is stored (ID `id_A`, score 1.0). User feedback: 'y'.
2.  Same user prompt -> Cache HIT (returns `ActionSequence A` from `id_A`). Agent executes (displays actions). User feedback: 'n'. Score for `id_A` drops (e.g., to 0.7).
3.  Repeat step 2 multiple times. Each 'n' drops the score of `id_A` further.
4.  Once score of `id_A` drops below `SCORE_THRESHOLD_EPSILON`, it's evicted.
5.  Same user prompt -> Cache MISS (as `id_A` is gone). `CapturingAgent` runs again, generates `ActionSequence B`, stores it (ID `id_B`, score 1.0). This demonstrates re-planning.

## 9. Public API (Python - `memory_cache.py`)

```python
from typing import List, Optional, Tuple
import uuid

# Simplified ActionSequence type
ActionSequence = List[str]

class MemoryCache:
    def __init__(self): ...

    def lookup(self, prompt: str) -> Optional[Tuple[uuid.UUID, ActionSequence]]:
        # Returns (entry_id, actions) if hit, else None
        ...

    def store(self, prompt: str, actions: ActionSequence) -> Optional[uuid.UUID]:
        # Stores prompt and actions, returns new entry_id or None on failure
        ...

    def update_reward(self, entry_id: uuid.UUID, success: bool) -> bool:
        # Updates score for entry_id, returns True if operation processed, False on error (e.g. ID not found)
        # Handles eviction if score drops below threshold.
        ...
```

## 10. Configuration / Tunables (Constants in `memory_cache.py` and `llm_module`)

| Parameter                 | File (`Context`)        | Value (Current)        | Rationale                                                     |
|---------------------------|-------------------------|------------------------|---------------------------------------------------------------|
| `OPENAI_EMBEDDING_MODEL`  | `memory_cache.py`       | `text-embedding-3-small` | Good balance of performance and cost for OpenAI embeddings.   |
| `SIMILARITY_THRESHOLD_TAU`| `memory_cache.py`       | `0.60`                 | Minimum similarity score for a cache hit. Tuned via testing.  |
| `SCORE_THRESHOLD_EPSILON` | `memory_cache.py`       | `0.2`                  | Entries with score below this are evicted.                    |
| `REWARD_ALPHA`            | `memory_cache.py`       | `0.3`                  | Learning rate for EMA score updates.                          |
| `TOP_K_RESULTS`           | `memory_cache.py`       | `3`                    | Number of results to fetch from vector DB for consideration.  |
| `DEFAULT_AGENT_PROMPT_TEMPLATE` | `llm_module/capturing_agent.py` | (See file)             | Instructs the `CapturingAgent` on reasoning and tool use. |
| `ChatLLM.model`           | `llm_module/llm.py`     | `gpt-3.5-turbo`        | Default LLM used by the `CapturingAgent`.                   |
| `Agent.max_loops`         | `llm_module/agent.py`   | `15`                   | Max iterations for the agent's internal thought-action loop.  |

## 11. Testing
The project uses several methods for testing:
-   **Unit/Component Tests for `MemoryCache`:** Individual `test_*.py` scripts (`test_chroma_init.py`, `test_chroma_store.py`, etc.) verify core functionalities of `MemoryCache` and its ChromaDB interactions.
-   **Component Tests for `llm_module`:** Files within `llm_module` (e.g., `llm.py`, `custom_tools.py`, `capturing_agent.py`) include `if __name__ == '__main__':` blocks for direct execution and testing of those specific components (LLM connectivity, tool behavior, agent's reasoning loop).
-   **End-to-End Integration Test (`mock_agent_demo.py`):** This script provides a command-line interface to interact with the integrated system. It allows manual verification of:
    *   Cache hits and misses.
    *   The `CapturingAgent`'s ability to use its LLM and tools to process prompts.
    *   Correct storage of the agent's `ActionSequence` (tool interactions or direct answers).
    *   Reward updates based on interactive feedback.
    *   The self-healing mechanism (eviction of low-scoring plans and subsequent re-planning by the `CapturingAgent`).

*(Original sections on Failure Scenarios, Security, and Future Extensions can be kept or adapted if desired, but the above covers the implemented system.)*

## 12. Future Extensions (from original plan and new ideas)
- Cross‑prompt generalisation.
- Multi‑modal embeddings.
- Dashboard for observability.
- Language bindings.
- **Web-based Chat Interface:** (See Phase 6 of `development_plan.md`)
- **Advanced Reward Mechanisms:** (See Phase 6 of `development_plan.md`)
- **Agent Polishing & Robustness:** (See Phase 6 of `development_plan.md`)