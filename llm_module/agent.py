import datetime
import re

from pydantic import BaseModel
from typing import List, Dict, Tuple
from .llm import ChatLLM
from .tools.base import Tool


FINAL_ANSWER_TOKEN = "Final Answer:"
OBSERVATION_TOKEN = "Observation:"
THOUGHT_TOKEN = "Thought:"
PROMPT_TEMPLATE = """Today is {today} and you can use tools to get new information. Answer the question as best as you can using the following tools: 

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: comment on what you want to do next
Action: the action to take, exactly one element of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation repeats N times, use it until you are sure of the answer)
Thought: I now know the final answer
Final Answer: your final answer to the original input question

Begin!

Question: {question}
Thought: {previous_responses}
"""


class Agent(BaseModel):
    llm: ChatLLM
    tools: List[Tool]
    prompt_template: str = PROMPT_TEMPLATE
    max_loops: int = 15
    stop_sequence: List[str] = [f'\n{OBSERVATION_TOKEN}']
    finish_tool_name: str = "Final Answer"
    agent_name: str = "AI Assistant"
    today_date: str = str(datetime.date.today())

    def _get_tools_description(self) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

    def _get_tool_names(self) -> str:
        return ",".join([tool.name for tool in self.tools])

    def _find_tool(self, tool_name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def run(self, question: str):
        print("Warning: Calling base Agent.run(). CapturingAgent has its own run() method.")
        
        previous_responses_str = ""
        num_loops = 0
        
        current_prompt = self.prompt_template.format(
                today=self.today_date,
                tool_description=self._get_tools_description(),
                tool_names=self._get_tool_names(),
                question=question,
                agent_scratchpad=previous_responses_str 
        )

        while num_loops < self.max_loops:
            num_loops += 1
            
            generated_text, tool_name, tool_input_text = self.decide_next_action(current_prompt)
            
            if tool_name == self.finish_tool_name:
                return tool_input_text
            
            tool_instance = self._find_tool(tool_name)
            if not tool_instance:
                print(f"Error: Tool '{tool_name}' not found by base Agent.run after LLM decision.")
                observation = f"Error: Tool '{tool_name}' is not available."
            else:
                observation = tool_instance(tool_input_text)
            
            current_prompt += f"{generated_text}\n{OBSERVATION_TOKEN} {observation}\n{THOUGHT_TOKEN}" 

        return "Error: Max loops reached in base Agent.run()"

    def decide_next_action(self, current_prompt: str) -> Tuple[str, str, str]:
        generated_llm_output = self.llm.generate(current_prompt, stop=self.stop_sequence)
        parsed_tool_name, parsed_tool_input = self._parse_llm_output(generated_llm_output)
        return generated_llm_output, parsed_tool_name, parsed_tool_input

    def _parse_llm_output(self, llm_output: str) -> Tuple[str, str]:
        if FINAL_ANSWER_TOKEN in llm_output:
            return self.finish_tool_name, llm_output.split(FINAL_ANSWER_TOKEN)[-1].strip()
        
        regex = r"Action:\s*(.*?)(?:\n|$)Action Input:\s*(.*?)(?:\n|$)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            return "ErrorNoAction", f"Could not parse Action/ActionInput from: {llm_output}"
            
        tool_name = match.group(1).strip()
        tool_input_text = match.group(2).strip(" \"")
        return tool_name, tool_input_text


if __name__ == '__main__':
    print("Running __main__ from llm_module/agent.py (Base Agent tests)")
    pass
