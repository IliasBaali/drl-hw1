import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch.distributions.categorical import Categorical

from src.utils import device

DEVICE = device()
HIDDEN_DIMENSION: int = 64


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)

class Model(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, out_dimension):
        super().__init__()
        self.input = nn.Linear(in_dimension,hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension,hidden_dimension//2)
        self.output = nn.Linear(hidden_dimension//2,out_dimension)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

def network(
        in_dimension: int, hidden_dimension: int, out_dimension: int
) -> nn.Module:
    """
    TODO: Design and implement the neural network architecture.

    Args:
        in_dimension (int): Dimension of the input layer.
        hidden_dimension (int): Dimension of the hidden layers.
        out_dimension (int): Dimension of the output layer.

    Returns:
        nn.Module: The constructed neural network model.
    """
    network = Model(in_dimension, hidden_dimension, out_dimension)
    return network


class Policy(nn.Module):
    def __init__(
            self,
            state_dimension: int,
            num_actions: int,
            hidden_dimension: int = HIDDEN_DIMENSION
    ):
        super(Policy, self).__init__()
        self.network = network(
            state_dimension, hidden_dimension, num_actions
        )

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the Policy network.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The output logits for each action.

        TODO: Implement the forward method to compute the network output for the given state.
        You can use the self.network to forward the input.
        """
        return self.network(tensor(state))

    def pi(self, state: np.ndarray) -> Categorical:
        """
        Computes the action distribution π(a|s) for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Categorical: The action distribution.

        TODO: Implement the pi method to create a Categorical distribution based on the network's output.
        """
        logits = self.forward(state)
        probs = F.softmax(logits)
        distribution = Categorical(probs=probs)
        return distribution

    def sample(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples an action from the policy and returns the action along with its log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Tuple[int, torch.Tensor]: The sampled action and its log probability.

        TODO: Implement the sample method to sample an action and compute its log probability.
        """
        distribution = self.pi(state)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return (action.item(), log_prob)

    def sample_multiple(self, states: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples actions for multiple states and returns the actions along with their log probabilities.

        Args:
            states (np.ndarray): The input states.

        Returns:
            Tuple[int, torch.Tensor]: The sampled actions and their log probabilities.

        TODO: Implement the sample_multiple method to handle multiple states.
        """
        actions =  []
        log_probs = []
        for state in states:
            (action, log_prob) = self.sample(state)
            actions.append(action)
            log_probs.append(log_prob)
        return (tensor(actions, type=int), torch.cat([t.view(1) for t in log_probs]))

    def action(self, state: np.ndarray) -> torch.Tensor:
        """
        Selects an action based on the policy without returning the log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The selected action.

        TODO: Implement the action method to return an action based on the sampled action.
        """
        distribution = self.pi(state)
        action = distribution.sample()
        return action.item()

class ValueFunctionQ(nn.Module):
    def __init__(
            self,
            state_dimension: int,
            num_actions: int,
            hidden_dimension: int = HIDDEN_DIMENSION
    ):
        super(ValueFunctionQ, self).__init__()
        self.network = network(
            state_dimension, hidden_dimension, num_actions
        )

    def __call__(
            self, state: np.ndarray, action: Optional[int] = None
    ) -> torch.Tensor:
        """
        Computes the Q-values Q(s, a) for given states and optionally for specific actions.

        Args:
            state (np.ndarray): The input state.
            action (Optional[int], optional): The action for which to compute the Q-value. Defaults to None.

        Returns:
            torch.Tensor: The Q-values.

        TODO: Implement the __call__ method to return Q-values for the given state and action.
        This method is intended to compute Q(s, a).
        """
        Q_values = self.forward(state)
        if action != None:
            return Q_values[action]
        return Q_values

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the ValueFunctionQ network.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The Q-values for each action.

        TODO: Implement the forward method to compute Q-values for the given state.
        You can use the self.network to forward the input.
        """
        return self.network(tensor(state))

    def greedy(self, state: np.ndarray) -> torch.Tensor:
        """
        Selects the action with the highest Q-value for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The action with the highest Q-value.

        TODO: Implement the greedy method to select the best action based on Q-values.
        This method is intended for greedy sampling.
        """
        Q_values = self(state)
        action = torch.argmax(Q_values)
        return action

    def action(self, state: np.ndarray) -> torch.Tensor:
        """
        Returns the greedy action for the given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The selected action.

        TODO: Implement the action method to return the greedy action.
        """
        return self.greedy(state)

    def V(self, state: np.ndarray, policy: Policy) -> float:
        """
        Computes the expected value V(s) of the state under the given policy.

        Args:
            state (np.ndarray): The input state.
            policy (Policy): The policy to evaluate.

        Returns:
            float: The expected value.

        TODO: Implement the V method to compute the expected value of the state under the policy.
        This method is intended to return V(s).
        """
        distribution = policy.pi(state)
        V = torch.sum(distribution.probs*self(state))
        return V.item()
