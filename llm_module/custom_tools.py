from typing import Type # Required for Python 3.8 compatibility if Type is used as a class
from .tools.base import Tool # Adjusted import path
# from tools.base import Tool # For direct script execution

class SetPlayerAttributeTool(Tool):
    name: str = "SetPlayerAttribute"
    description: str = (
        "Sets an attribute for the player character. "
        "Input should be in the format 'attribute_name=value' (e.g., 'speed=10', 'health=100', 'mana=50')."
    )

    def __call__(self, action_input: str) -> str:
        action_input = action_input.strip()
        if '=' not in action_input:
            return "Observation: Error - Invalid format for SetPlayerAttribute. Expected 'attribute_name=value'."
        try:
            attribute_name, value = action_input.split('=', 1)
            attribute_name = attribute_name.strip()
            value = value.strip()
            # In a real game, you might try to cast value to int/float or validate attribute_name
            if not attribute_name or not value:
                return "Observation: Error - Attribute name or value cannot be empty."
            return f"Observation: Player attribute '{attribute_name}' set to '{value}'."
        except ValueError:
            return "Observation: Error - Invalid format for SetPlayerAttribute. Expected 'attribute_name=value'."

class SpawnEntityTool(Tool):
    name: str = "SpawnEntity"
    description: str = (
        "Spawns an entity in the game world. "
        "Input should be in the format 'entity_type,x,y,z' (e.g., 'goblin,10,0,5', 'health_potion,0,1,0')."
    )

    def __call__(self, action_input: str) -> str:
        action_input = action_input.strip()
        parts = [p.strip() for p in action_input.split(',')]
        if len(parts) != 4:
            return "Observation: Error - Invalid format for SpawnEntity. Expected 'entity_type,x,y,z'."
        
        entity_type, x_str, y_str, z_str = parts
        if not entity_type:
            return "Observation: Error - Entity type cannot be empty."
        try:
            # Attempt to convert coordinates to numbers for validation, though not strictly necessary for mock tool
            float(x_str), float(y_str), float(z_str)
            return f"Observation: Entity '{entity_type}' spawned at coordinates ({x_str}, {y_str}, {z_str})."
        except ValueError:
            return "Observation: Error - Invalid coordinates for SpawnEntity. x, y, z must be numbers."

class ChangeSkyboxTool(Tool):
    name: str = "ChangeSkybox"
    description: str = (
        "Changes the skybox in the game. "
        "Input should be a skybox theme or color (e.g., 'night_sky', 'crimson_sunset', 'blue')."
    )

    def __call__(self, action_input: str) -> str:
        theme_or_color = action_input.strip()
        if not theme_or_color:
            return "Observation: Error - Skybox theme/color cannot be empty."
        return f"Observation: Skybox changed to '{theme_or_color}'."

class PlaySoundTool(Tool):
    name: str = "PlaySoundEffect"
    description: str = (
        "Plays a sound effect. "
        "Input should be the name of the sound effect (e.g., 'explosion', 'player_jump', 'door_open')."
    )

    def __call__(self, action_input: str) -> str:
        sound_name = action_input.strip()
        if not sound_name:
            return "Observation: Error - Sound effect name cannot be empty."
        return f"Observation: Sound effect '{sound_name}' played."


# Example usage (for testing purposes)
if __name__ == '__main__':
    set_attr_tool = SetPlayerAttributeTool()
    spawn_tool = SpawnEntityTool()
    skybox_tool = ChangeSkyboxTool()
    sound_tool = PlaySoundTool()

    print("--- SetPlayerAttribute Tool Testing ---")
    print(set_attr_tool(action_input="health=100"))
    print(set_attr_tool(action_input="speed = 25 "))
    print(set_attr_tool(action_input="invalid_format"))
    print(set_attr_tool(action_input="name="))

    print("\n--- SpawnEntity Tool Testing ---")
    print(spawn_tool(action_input="orc_warrior, 15.5, 2.0, -5.0"))
    print(spawn_tool(action_input="item_chest,0,0")) # Invalid format
    print(spawn_tool(action_input="tree,10,ground,20")) # Invalid coordinates
    print(spawn_tool(action_input=",10,10,10")) # Empty entity type

    print("\n--- ChangeSkybox Tool Testing ---")
    print(skybox_tool(action_input="stormy_night"))
    print(skybox_tool(action_input="  ")) # Empty input

    print("\n--- PlaySound Tool Testing ---")
    print(sound_tool(action_input="sword_clash_heavy"))
    print(sound_tool(action_input="")) 