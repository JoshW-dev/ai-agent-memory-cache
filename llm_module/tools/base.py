from pydantic import BaseModel

class Tool(BaseModel):
    name: str
    description: str
    
    def __call__(self, action_input: str) -> str:
        raise NotImplementedError("__call__() method not implemented in subclass")
