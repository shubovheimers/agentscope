"""Base agent module for AgentScope."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class AgentBase(ABC):
    """Abstract base class for all agents in AgentScope.

    All custom agents should inherit from this class and implement
    the `reply` method.

    Example:
        >>> class MyAgent(AgentBase):
        ...     def reply(self, message):
        ...         return {"role": self.name, "content": "Hello!"}
    """

    def __init__(
        self,
        name: str,
        sys_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            name (str): The name of the agent.
            sys_prompt (Optional[str]): System prompt for the agent.
        """
        self.name = name
        self.sys_prompt = sys_prompt or ""
        self._memory: list[dict] = []

    @abstractmethod
    def reply(self, message: dict) -> dict:
        """Generate a reply to the given message.

        Args:
            message (dict): The input message with at least 'role' and
                'content' keys.

        Returns:
            dict: The agent's response message.
        """

    def observe(self, message: dict) -> None:
        """Observe a message without generating a reply.

        Stores the message in the agent's memory.

        Args:
            message (dict): The message to observe.
        """
        self._memory.append(message)

    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self._memory.clear()

    def __call__(self, message: dict) -> dict:
        """Make the agent callable, delegating to `reply`.

        Args:
            message (dict): The input message.

        Returns:
            dict: The agent's response.
        """
        response = self.reply(message)
        self._memory.append(message)
        self._memory.append(response)
        return response

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
