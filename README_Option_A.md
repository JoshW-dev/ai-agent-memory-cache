# AI Agent Memory Cache - Option A Prototype

This README describes the "Option A - Required Minimum" prototype for the AI Agent Memory Cache.

## Core Functionality

The `MemoryCache` module (`memory_cache.py`) stores and retrieves sequences of action strings (simplified tool calls) based on exact user prompt matches.

**Key Features:**
- **`MemoryCache` Class**:
    - In-memory dictionary for prompt -> action sequence storage.
    - `store(prompt: str, actions: List[str])`: Caches a prompt and its actions.
    - `lookup(prompt: str) -> Optional[List[str]]`: Retrieves actions for an exact prompt match; `None` on miss.
    - `update_reward(prompt: str, success: bool)`: Placeholder; prints conceptual reward feedback.
- **`ActionSequence`**: `List[str]`, where each string is a simplified action.

## Demonstration (`demo_script.py`)

The `demo_script.py` showcases:
1. Storing two prompt-action pairs.
2. Retrieving a stored sequence (cache hit by exact match).
3. Attempting to retrieve a non-existent sequence (cache miss).
4. Conceptual reward updates via `update_reward()`.

## Setup & Execution

1.  **Prerequisites**: Python 3.
2.  **Run Demo**: `python3 demo_script.py`

## Expected Demo Output

```
--- Basic MemoryCache Demo ---
Stored: 'Make the player move faster' -> ['increase_player_speed(factor=2)', 'save_game_state()']
Stored: 'Change sky color to blue' -> ["set_sky_color(color='blue')", 'render_scene()']
---
Looking up an existing prompt:
Found for 'Make the player move faster': ['increase_player_speed(factor=2)', 'save_game_state()']
---
Looking up a non-existent prompt: 'Make the enemy stronger'
No sequence found for 'Make the enemy stronger'
---
Updating reward for an existing prompt:
Reward update for prompt 'Make the player move faster': Success = True
Reward update for prompt 'Change sky color to blue': Success = False
--- Demo Complete ---
```
This output confirms basic store, exact lookup, and conceptual reward functionalities are working. 