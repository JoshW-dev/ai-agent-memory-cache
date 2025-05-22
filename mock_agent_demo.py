from memory_cache import MemoryCache, ActionSequence
from dotenv import load_dotenv
import os
import time
import uuid
from typing import Optional, List, Dict

# Imports for the new CapturingAgent
from llm_module.llm import ChatLLM
from llm_module.custom_tools import WeatherTool, InventoryCheckTool, MessageHandlerTool
from llm_module.capturing_agent import CapturingAgent, DEFAULT_AGENT_PROMPT_TEMPLATE

def get_reward_feedback() -> bool:
    while True:
        feedback = input("Did this action sequence work? (y/n): ").strip().lower()
        if feedback == 'y':
            return True
        elif feedback == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def main():
    load_dotenv()
    print("--- Mock Agent Demo with CapturingAgent & ChromaDB Cache (with Rewards) ---")
    print("Type your request to the agent, or type 'quit' or 'exit' to end.")

    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    # Initialize MemoryCache
    cache = MemoryCache()
    print(f"MemoryCache initialized. Initial collection count: {cache._collection.count()}")

    # Initialize LLM, Tools, and CapturingAgent
    llm = ChatLLM() # Uses OPENAI_API_KEY from environment
    tools = [WeatherTool(), InventoryCheckTool(), MessageHandlerTool()]
    # We can use the default prompt template from capturing_agent.py or define one here
    agent_prompt = DEFAULT_AGENT_PROMPT_TEMPLATE 
    agent = CapturingAgent(llm=llm, tools=tools, prompt_template=agent_prompt)
    print("CapturingAgent initialized with Weather, Inventory, and Message tools.")

    while True:
        user_prompt = input("\nYour request: ").strip()

        if user_prompt.lower() in ["quit", "exit"]:
            print("Exiting agent demo.")
            break
        
        if not user_prompt:
            continue

        print(f"\nAgent processing prompt: '{user_prompt}'")
        lookup_result = cache.lookup(user_prompt)
        
        current_entry_id_for_reward: Optional[uuid.UUID] = None
        executed_actions_for_display: ActionSequence = []

        if lookup_result:
            entry_id, cached_actions = lookup_result
            current_entry_id_for_reward = entry_id
            executed_actions_for_display = cached_actions
            print("\n>>> Cache HIT!")
            print(f"  Retrieved actions for Entry ID: {entry_id}")
            for i, action_str in enumerate(executed_actions_for_display):
                print(f"    {i+1}. {action_str}")
            
            print("  Agent executes retrieved actions...") 
            time.sleep(0.5)
            # success = get_reward_feedback()
            # print(f"  Updating reward for Entry ID {entry_id} with success: {success}")
            # cache.update_reward(entry_id, success)

        else:
            print("\n>>> Cache MISS.")
            print("  Agent is thinking... (using LLM and tools via CapturingAgent)")
            # agent.run() returns -> Tuple[str, List[Dict[str, str]]]
            # The first element is the final answer string from the LLM.
            # The second is the history of tool calls.
            final_answer_from_agent, tool_history_dicts = agent.run(user_prompt)
            
            print(f"\n  Agent's Final Answer: {final_answer_from_agent}")
            print("  Actions taken by agent during this run:")
            
            # Convert tool_history_dicts to ActionSequence (List[str]) for caching and display
            newly_generated_action_sequence: ActionSequence = []
            for step in tool_history_dicts:
                action_str = f"Tool: {step.get('tool_name', 'N/A')}, " \
                             f"Input: '{step.get('tool_input', 'N/A')}', " \
                             f"Observation: '{step.get('observation', 'N/A')}'"
                newly_generated_action_sequence.append(action_str)
            
            executed_actions_for_display = newly_generated_action_sequence
            for i, action_str in enumerate(executed_actions_for_display):
                print(f"    {i+1}. {action_str}")
            
            if not tool_history_dicts and final_answer_from_agent.startswith("Error:"):
                print("  Agent encountered an error and took no actions. Not storing in cache.")
            elif not tool_history_dicts:
                print("  Agent provided an answer without using tools. Storing simple response.")
                # Decide if we want to store non-tool-use answers. For now, let's store it as a single action.
                # This makes it consistent with the cache expecting an ActionSequence.
                # If the final_answer_from_agent is all we have, wrap it.
                executed_actions_for_display = [f"Direct Answer: {final_answer_from_agent}"]
                print(f"    1. {executed_actions_for_display[0]}")
                new_entry_id = cache.store(user_prompt, executed_actions_for_display)
                if new_entry_id:
                    current_entry_id_for_reward = new_entry_id
                    print(f"  Stored direct answer with ID: {new_entry_id}")
                else:
                    print("  Failed to store direct answer.")
            else: # Agent used tools
                print("  Storing new plan (sequence of tool actions) in cache...")
                new_entry_id = cache.store(user_prompt, executed_actions_for_display)
                if new_entry_id:
                    current_entry_id_for_reward = new_entry_id
                    print(f"  New plan stored with ID: {new_entry_id}")
                else:
                    print("  Failed to store new plan.")

        # Common reward feedback logic, whether it was a HIT or MISS (if an action was stored/retrieved)
        if current_entry_id_for_reward and executed_actions_for_display:
            print("  Simulating execution of the above actions...") # Simulate execution
            time.sleep(0.5)
            success = get_reward_feedback()
            print(f"  Updating reward for Entry ID {current_entry_id_for_reward} with success: {success}")
            cache.update_reward(current_entry_id_for_reward, success)
        elif not executed_actions_for_display:
            print("  No actions were executed or retrieved, so no reward feedback needed.")
            
    print("\n--- Mock Agent Demo Complete ---")

if __name__ == "__main__":
    main() 