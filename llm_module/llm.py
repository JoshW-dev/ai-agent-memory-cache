import openai
import os

from pydantic import BaseModel
from typing import List

# Import the new OpenAI client
from openai import OpenAI


class ChatLLM(BaseModel):
    model: str = 'gpt-3.5-turbo'
    temperature: float = 0.0
    # openai.api_key = os.environ["OPENAI_API_KEY"] # Old way of setting API key

    # Add a client instance. Pydantic needs `validate_assignment=True` if we want to assign
    # non-model fields after init, or we can use a private attribute.
    # For simplicity, we'll initialize it here. We need to ensure Pydantic handles this.
    # A cleaner way for Pydantic is to use a private attribute or a context manager.
    # Let's try making it a private attribute to avoid Pydantic validation issues.
    _client: OpenAI = None # Field for the client instance

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenAI() # Initialize client using env var OPENAI_API_KEY

    def generate(self, prompt: str, stop: List[str] = None) -> str: # Added return type hint
        # response = openai.ChatCompletion.create( # Old API call
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=self.temperature,
        #     stop=stop
        # )
        # return response.choices[0].message.content # Old response parsing

        if self._client is None:
            # This case should ideally not be reached if __init__ is always called.
            # Or, if __init__ could fail to set it up, handle appropriately.
            # For now, let's assume __init__ sets it.
            raise RuntimeError("OpenAI client not initialized.")

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            # Decide on how to handle the error, e.g., return a default string, None, or re-raise
            return "Error: Could not get response from LLM."


if __name__ == '__main__':
    # Ensure OPENAI_API_KEY is set in your environment for this test to run
    try:
        llm = ChatLLM()
        result = llm.generate(prompt='Who is the current president of the USA?')
        print(f"LLM Response: {result}")
        
        # Test with a stop sequence (optional)
        # result_with_stop = llm.generate(prompt='Count to 5.', stop=["3"])
        # print(f"LLM Response with stop: {result_with_stop}")

    except Exception as e:
        print(f"Error in main: {e}")
        print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
