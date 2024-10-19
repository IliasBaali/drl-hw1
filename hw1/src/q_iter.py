import gym
import torch
import numpy as np
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ


DEVICE = device()
EPS_END: float = 0.01
EPS_START: float = 1.0
EPS_DECAY: float = 0.999_9
eps: float = EPS_START

# simple MSE loss
def loss(
        value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
    mean_square_error = (value - target)**2
    return mean_square_error


def greedy_sample(Q: ValueFunctionQ, state: np.array):
    with torch.no_grad():
        return Q.action(state)


def eps_greedy_sample(Q: ValueFunctionQ, state: np.array):
    global eps
    eps = max(EPS_END, EPS_DECAY * eps)

    # TODO: Implement epsilon-greedy action selection
    # Hint: With probability eps, select a random action
    # Hint: With probability (1 - eps), select the best action using greedy_sample
    Q_values = Q(state)
    best = torch.argmax(Q_values).item()
    if np.random.rand()>eps:
        return best
    else:
        return np.random.choice(np.arange(Q_values.shape[0])[np.arange(Q_values.shape[0])!=best])


def train_one_epoch(
        env: gym.Env,
        Q: ValueFunctionQ,
        optimizer: Optimizer,
        gamma: float = 0.99
    ) -> float:
    Q.train()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    episode_reward: float = 0.0

    for t in count():
        # TODO: Generate action using epsilon-greedy policy
        action = eps_greedy_sample(Q, state)
        # TODO: Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if terminated:
            next_state = None

        # Calculate the target
        with torch.no_grad():
            # TODO: Compute the target Q-value
            target = reward
            if not terminated:
                target += gamma*torch.amax(Q(next_state))

        # TODO: Compute the loss
        loss_ = loss(Q(state,action),target)
        
        # TODO: Perform backpropagation and update the network
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        # TODO: Update the state
        next_state = state

        # TODO: Handle episode termination
        if done:
            break

    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
