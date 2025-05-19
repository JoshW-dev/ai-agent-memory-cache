from typing import List, Optional, Dict

# P1-T6: Define ActionSequence Type (List[str])
ActionSequence = List[str]

class MemoryCache:
    def __init__(self):
        # P1-T2: Implement In-Memory Storage
        self._cache: Dict[str, ActionSequence] = {}

    def lookup(self, prompt: str) -> Optional[ActionSequence]:
        # P1-T4: Implement lookup() Method (Exact Match)
        return self._cache.get(prompt)

    def store(self, prompt: str, actions: ActionSequence) -> None:
        # P1-T3: Implement store() Method
        self._cache[prompt] = actions

    def update_reward(self, prompt: str, success: bool) -> None:
        # P1-T5: Implement update_reward() Method (Conceptual)
        print(f"Reward update for prompt '{prompt}': Success = {success}")
        # In later phases, this will update scores and handle eviction. 