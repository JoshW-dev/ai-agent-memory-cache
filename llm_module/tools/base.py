from pydantic import BaseModel
from typing import List, Optional, Any

class Tool(BaseModel):
    name: str
    description: str
    primary_embedding: Optional[List[float]] = None
    additional_prompt_embeddings: List[List[float]] = []
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        if 'additional_prompt_embeddings' not in data:
            self.additional_prompt_embeddings = []

    def __call__(self, action_input: str) -> str:
        raise NotImplementedError("__call__() method not implemented in subclass")

    def add_representative_prompt_embedding(self, embedding: List[float]):
        if embedding not in self.additional_prompt_embeddings:
            self.additional_prompt_embeddings.append(embedding)
            print(f"Added representative prompt embedding to tool '{self.name}'. Count: {len(self.additional_prompt_embeddings)}")

    def get_all_embeddings(self) -> List[List[float]]:
        all_embs = []
        if self.primary_embedding:
            all_embs.append(self.primary_embedding)
        all_embs.extend(self.additional_prompt_embeddings)
        return all_embs
