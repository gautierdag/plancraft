
from abc import ABC, abstractmethod
from copilot import CopilotClient
import re
from plancraft.environment.search import gold_search_recipe


class CopilotBaseAgent(ABC):
    """
    Abstract Base Class for Copilot-powered Plancraft agents.
    Mirrors BaseAgent but uses CopilotClient instead of genai.Client.
    """
    def __init__(self, model_name: str, client: CopilotClient):
        self.model_name = model_name
        self.client = client
        self.conversation = []  # Stores history as {"role": ..., "content": ...}
        self.step_count = 0
        self.example_id = None
        self.target = None

    def reset(self, example_id: str, target: str):
        """Resets the agent state for a new episode."""
        self.conversation = []
        self.step_count = 0
        self.example_id = example_id
        self.target = target

    @abstractmethod
    async def act(self, observation_text: str) -> str:
        """
        Receives an observation string and returns an action string.
        Async because the Copilot SDK is async.
        """
        pass

    def _oracle_search(self, query: str) -> str:
        """Performs search using the oracle recipe tool."""
        match = re.search(r"search:\s*(\S+)", query)
        if match:
            target = match.group(1).strip().lower()
            return gold_search_recipe(target)
        return "No results found."

    def log(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")
