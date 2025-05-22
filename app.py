import streamlit as st
from dotenv import load_dotenv
import os
import time # For any brief simulated delays if needed
import uuid
from typing import Optional, List, Dict, Tuple

# Core application imports
from memory_cache import MemoryCache, ActionSequence, LookupResult
from llm_module.llm import ChatLLM
# from llm_module.custom_tools import WeatherTool, InventoryCheckTool, MessageHandlerTool # Old tools
from llm_module.custom_tools import SetPlayerAttributeTool, SpawnEntityTool, ChangeSkyboxTool, PlaySoundTool # New game-specific tools
from llm_module.capturing_agent import CapturingAgent, DEFAULT_AGENT_PROMPT_TEMPLATE

# --- Initialization of Agent and Cache (using Streamlit caching) ---
@st.cache_resource # Cache the resource across reruns
def get_memory_cache():
    print("Initializing MemoryCache...")
    return MemoryCache()

@st.cache_resource
def get_capturing_agent():
    print("Initializing CapturingAgent...")
    llm = ChatLLM()
    # tools = [WeatherTool(), InventoryCheckTool(), MessageHandlerTool()] # Old tools instantiation
    tools = [SetPlayerAttributeTool(), SpawnEntityTool(), ChangeSkyboxTool(), PlaySoundTool()] # New tools
    agent_prompt = DEFAULT_AGENT_PROMPT_TEMPLATE
    agent = CapturingAgent(llm=llm, tools=tools, prompt_template=agent_prompt)
    return agent

def main():
    st.set_page_config(page_title="AI Agent with Memory Cache", page_icon="🧠")
    st.title("🧠 AI Agent with Memory Cache")

    load_dotenv() # Ensure API keys are loaded
    if not os.getenv("OPENAI_API_KEY"):
        st.error("CRITICAL: OPENAI_API_KEY not found. Please set it in your .env file and restart.")
        st.stop()

    # Get (or create) cached instances of agent and memory cache
    cache = get_memory_cache()
    agent = get_capturing_agent()

    # Initialize chat history and other session variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_entry_id_for_reward" not in st.session_state:
        st.session_state.current_entry_id_for_reward = None
    if "last_executed_actions" not in st.session_state:
        st.session_state.last_executed_actions = [] # Store the ActionSequence that was last shown
    if "feedback_status" not in st.session_state:
        st.session_state.feedback_status = {}

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # For assistant messages that are action sequences, format them nicely
            if message["role"] == "assistant" and isinstance(message["content"], list):
                for action_item in message["content"]:
                    st.markdown(f"- {action_item}") # Display each action as a list item
            else:
                st.markdown(message["content"])

    # Add the "Available Tools" section to the sidebar
    st.sidebar.title("Available Tools")
    if agent.tools: # agent.tools is a List[Tool]
        for tool_instance in agent.tools: # Iterate over the list of tool objects
            with st.sidebar.expander(tool_instance.name): # Access tool_instance.name
                st.markdown(tool_instance.description) # Access tool_instance.description
    else:
        st.sidebar.markdown("No tools available for the agent.")

    # --- Placeholder Prompts ---
    placeholder_prompts = [
        "Make the skybox stormy",
        "Spawn a friendly dog",
        "Set player health to 50",
        "Play a happy sound effect"
    ]

    st.sidebar.title("Placeholder Prompts")
    st.sidebar.markdown("Click a prompt to try it out:")
    for i, p_prompt in enumerate(placeholder_prompts):
        if st.sidebar.button(p_prompt, key=f"placeholder_{i}"):
            # Simulate chat input submission when a placeholder button is clicked
            st.session_state.messages.append({"role": "user", "content": p_prompt})
            # Set the prompt to be processed as if it were typed
            # This requires a slight refactor of the input handling.
            # We'll trigger a rerun and process the prompt in the next cycle.
            st.session_state.process_prompt_now = p_prompt
            st.rerun()

    # --- User Input and Agent Logic ---
    user_input_prompt = st.chat_input("What can I help you with?")

    # Check if a placeholder prompt was clicked in the previous run
    if "process_prompt_now" in st.session_state and st.session_state.process_prompt_now:
        prompt = st.session_state.process_prompt_now
        st.session_state.process_prompt_now = None # Clear it after use
        # The user message for this prompt was already added when the button was clicked.
        # We just need to ensure it's displayed if not already.
        # This logic might need adjustment based on how message display is handled.
        # For now, let's assume the message was already added and will be displayed.
    elif user_input_prompt:
        prompt = user_input_prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    else:
        prompt = None # No input this cycle

    if prompt:
        # Agent processing
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # For "Thinking..." message
            message_placeholder.markdown("🧠 Agent thinking...")
            
            assistant_response_content: any = "Error: Could not get response."
            actions_to_display_and_store: ActionSequence = []
            entry_id_for_this_interaction: Optional[uuid.UUID] = None
            similarity_score_for_display: Optional[float] = None
            new_tool_defined_this_turn = False # Initialize flag

            lookup_result: Optional[LookupResult] = cache.lookup(prompt)

            if lookup_result:
                entry_id_for_this_interaction = lookup_result['entry_id']
                actions_to_display_and_store = lookup_result['actions']
                similarity_score_for_display = lookup_result['similarity_score']
                
                st.session_state.current_entry_id_for_reward = entry_id_for_this_interaction
                
                response_summary = f"Retrieved from cache (Similarity: {similarity_score_for_display:.2f}):"
            else:
                response_summary = "🧠 Generating new response with agent:"
                final_answer_from_agent, tool_history_dicts = agent.run(prompt)

                if tool_history_dicts:
                    for step in tool_history_dicts:
                        if step.get("tool_name") == "ToolDefinitionAgent":
                            new_tool_defined_this_turn = True
                        action_str = f"Tool: {step.get('tool_name', 'N/A')}, " \
                                     f"Similarity: {step.get('similarity_score', 'N/A')}, " \
                                     f"Input: '{step.get('tool_input', 'N/A')}', " \
                                     f"Observation: '{step.get('observation', 'N/A')}'"
                        actions_to_display_and_store.append(action_str)
                    response_summary += f"\nLLM Final Answer: {final_answer_from_agent}"
                elif not final_answer_from_agent.startswith("Error:"):
                    actions_to_display_and_store = [f"Direct Answer: {final_answer_from_agent}"]
                else: # Error from agent
                    actions_to_display_and_store = [f"Agent Error: {final_answer_from_agent}"]
                
                # Store in cache if actions were generated (even direct answers)
                if actions_to_display_and_store and not final_answer_from_agent.startswith("Error: Agent reached maximum loops") : # Avoid caching agent errors from loop exhaustion
                    new_entry_id = cache.store(prompt, actions_to_display_and_store)
                    if new_entry_id:
                        entry_id_for_this_interaction = new_entry_id
                        print(f"Stored new entry {new_entry_id} for prompt \'{prompt}\'")
                    else:
                        print(f"Failed to store entry for prompt \'{prompt}\'")
            
            # Update the placeholder with the actual response
            if actions_to_display_and_store:
                message_placeholder.markdown(response_summary) # Display summary first
                for action_item in actions_to_display_and_store:
                    st.markdown(f"- `{action_item}`") # Display each action, formatted as code for clarity
                assistant_response_content = actions_to_display_and_store # Store the list for history
            else:
                message_placeholder.markdown("No actions taken or retrieved.")
                assistant_response_content = "No actions taken or retrieved."

            st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})
            st.session_state.last_executed_actions = actions_to_display_and_store # Save for reward UI
            st.session_state.current_entry_id_for_reward = entry_id_for_this_interaction # Ensure it's set for this turn

            if new_tool_defined_this_turn:
                print("A new tool was defined by the agent. Triggering a rerun to update the tool list in the sidebar.")
                st.rerun()

    # --- Reward Feedback UI ---
    if st.session_state.last_executed_actions and st.session_state.current_entry_id_for_reward:
        entry_id = st.session_state.current_entry_id_for_reward
        current_feedback_status = st.session_state.feedback_status.get(entry_id)

        st.markdown("--- Provide Feedback ---")

        if current_feedback_status == "upvoted":
            st.success(f"👍 Feedback recorded: Worked Well (Entry ID: {entry_id})")
        elif current_feedback_status == "downvoted":
            st.error(f"👎 Feedback recorded: Did Not Work (Entry ID: {entry_id})")
        else:
            # Create two columns for buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Worked Well", key=f"worked_{entry_id}"):
                    cache.update_reward(entry_id, True)
                    st.session_state.feedback_status[entry_id] = "upvoted"
                    # Clear flags to hide this section for the current interaction turn and allow UI to update
                    st.session_state.last_executed_actions = [] 
                    st.session_state.current_entry_id_for_reward = None
                    st.rerun()
            with col2:
                if st.button("👎 Did Not Work", key=f"not_worked_{entry_id}"):
                    cache.update_reward(entry_id, False)
                    st.session_state.feedback_status[entry_id] = "downvoted"
                    # Clear flags to hide this section for the current interaction turn and allow UI to update
                    st.session_state.last_executed_actions = []
                    st.session_state.current_entry_id_for_reward = None
                    st.rerun()

if __name__ == "__main__":
    main() 