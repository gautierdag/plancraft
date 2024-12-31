import abc

from plancraft.utils import History


class PlancraftBaseModel(abc.ABC):
    """
    Model class must implement the following methods to work with evaluator
    """

    @abc.abstractmethod
    def step(self, observation: dict, dialogue_history: History) -> str:
        """
        Model should output an action in text based on the types available
        We also pass history to the model to allow for chat models to track the dialogue
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        """
        Reset the model state - ready for a new episode
        """
        raise NotImplementedError()
