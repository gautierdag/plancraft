import abc


class PlancraftBaseModel(abc.ABC):
    """
    Model class must implement the following methods to work with evaluator
    """

    @abc.abstractmethod
    def step(self, observation: list[dict]) -> str:
        """
        Model should output an action in text based on the types available
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()
