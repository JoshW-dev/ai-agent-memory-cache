from typing import Type # Required for Python 3.8 compatibility if Type is used as a class
from .tools.base import Tool # Adjusted import path
# from tools.base import Tool # For direct script execution

class WeatherTool(Tool):
    name: str = "WeatherLookup"
    description: str = (
        "Useful for finding out the weather in a specific city. "
        "Input should be a city name (e.g., London, Paris, Tokyo)."
    )

    def __call__(self, action_input: str) -> str:
        # In a real scenario, this would call a weather API
        city_name = action_input.strip()
        if not city_name:
            return "Observation: Error - City name cannot be empty."
        return f"Observation: The weather in {city_name} is sunny and 25°C."

class InventoryCheckTool(Tool):
    name: str = "InventoryChecker"
    description: str = (
        "Useful for checking if an item is in stock. "
        "Input should be an item ID or product name."
    )

    def __call__(self, action_input: str) -> str:
        item_id = action_input.strip()
        if not item_id:
            return "Observation: Error - Item ID cannot be empty."
        # Simulate checking inventory
        if item_id.lower() == "widget-001":
            return f"Observation: Item {item_id} is currently in stock. Quantity: 100 units."
        elif item_id.lower() == "gadget-x":
            return f"Observation: Item {item_id} is out of stock. Expected restock in 2 weeks."
        else:
            return f"Observation: Item {item_id} not found in inventory system."

class MessageHandlerTool(Tool):
    name: str = "MessageProcessor"
    description: str = (
        "Useful for processing or sending a simple message. "
        "Input should be the message content."
    )

    def __call__(self, action_input: str) -> str:
        message = action_input.strip()
        if not message:
            return "Observation: Error - Message content cannot be empty."
        return f"Observation: Message \"{message}\" processed successfully."


# Example usage (for testing purposes)
if __name__ == '__main__':
    weather_tool = WeatherTool()
    inventory_tool = InventoryCheckTool()
    message_tool = MessageHandlerTool()

    print("--- Weather Tool Testing ---")
    print(f"Tool Name: {weather_tool.name}")
    print(f"Tool Description: {weather_tool.description}")
    print(weather_tool(action_input="New York"))
    print(weather_tool(action_input="")) # Test empty input

    print("\n--- Inventory Check Tool Testing ---")
    print(f"Tool Name: {inventory_tool.name}")
    print(f"Tool Description: {inventory_tool.description}")
    print(inventory_tool(action_input="widget-001"))
    print(inventory_tool(action_input="gadget-X")) # Test case-insensitivity and different item
    print(inventory_tool(action_input="unknown-item"))
    print(inventory_tool(action_input="  ")) # Test whitespace input

    print("\n--- Message Handler Tool Testing ---")
    print(f"Tool Name: {message_tool.name}")
    print(f"Tool Description: {message_tool.description}")
    print(message_tool(action_input="Hello World!"))
    print(message_tool(action_input="   Order confirmed.   ")) # Test with whitespace 