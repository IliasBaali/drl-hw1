import gym
import torch
import numpy as np
import torch.nn as nn
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ, Policy
from src.buffer import ReplayBuffer, Transition

DEVICE = device()


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


# Hint: loss you can use to update Q function
def loss_Q(
        value_batch: torch.Tensor, target_batch: torch.Tensor
) -> torch.Tensor:
    mse = nn.MSELoss()
    return mse(value_batch, target_batch)


# Hint: loss you can use to update policy
def loss_pi(
        log_probabilities: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    return -1.0 * (log_probabilities * advantages).mean()

# Hint: you can use similar implementation from dqn algorithm
def optimize_Q(
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        policy: Policy,
        gamma: float,
        batch: Transition,
        optimizer: Optimizer
):
    states = np.stack(batch.state)
    actions = np.stack(batch.action)
    rewards = np.stack(batch.reward)
    valid_next_states = np.stack(tuple(
        filter(lambda s: s is not None, batch.next_state)
    ))

    nonterminal_mask = tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        type=torch.bool
    )

    actions_, log_probabilities = policy.sample_multiple(states)
    actions_ = actions_.unsqueeze(-1)[nonterminal_mask]

    rewards = tensor(rewards)
    batch_size = len(rewards)
    # TODO: Update the Q-network

    # calculate the target
    targets = torch.zeros(size=(batch_size, 1), device=DEVICE)
    with torch.no_grad():
        # Hint: Compute the target Q-values
        targets = rewards
        #targets[nonterminal_mask] += gamma*target_Q(valid_next_states)[torch.arange(actions_.shape[1]),actions_.squeeze()]
        #targets +=  gamma*torch.amax(target_Q(valid_next_states),1)
        targets[nonterminal_mask]  +=  gamma*target_Q(valid_next_states)[torch.arange(valid_next_states.shape[0]),actions_.squeeze()]
        
    #forward pass
    y_pred = Q(states)[torch.arange(actions.shape[0]),actions.squeeze()]
    loss = loss_Q(y_pred,targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print("Q loss:", loss)




# Hint: you can use similar implementation from reinforce algorithm
def optimize_policy(
        policy: Policy,
        Q: ValueFunctionQ,
        batch: Transition,
        optimizer: Optimizer
):
    states = np.stack(batch.state)

    actions, log_probabilities = policy.sample_multiple(states)

    actions = actions.unsqueeze(-1)
    log_probabilities = log_probabilities.unsqueeze(-1)

    # TODO: Update the policy network

    with torch.no_grad():
        # Hint: Advantages
        values = torch.tensor([Q.V(state,policy) for state in states])
        advantages = Q(states)[torch.arange(states.shape[0]),actions.squeeze()]-values
    entropies = torch.stack([policy.pi(state).entropy() for state in states])
        
    loss = loss_pi(log_probabilities,advantages.unsqueeze(-1))#+1e-6*entropies.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print("PI loss:", loss)


def train_one_epoch(
        env: gym.Env,
        policy: Policy,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        memory: ReplayBuffer,
        optimizer_Q: Optimizer,
        optimizer_pi: Optimizer,
        gamma: float = 0.99,
) -> float:
    # make sure target isn't fitted
    policy.train(), Q.train(), target_Q.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset()
    
    for t in count():
        
        #Sample an action from the policy
        (action,log_prob) = policy.sample(state)
        
        #Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # TODO: Complete the train_one_epoch for actor-critic algorithm

        if terminated:
            next_state = None

        # TODO: Store the transition in memory
        # Hint: Use replay buffer!
        memory.push(state, action, next_state, reward)

        # Hint: Check if replay buffer has enough samples
        if len(memory) >= memory.batch_size and t%20 == 0:
            batch_transitions = memory.sample()
            batch = Transition(*zip(*batch_transitions))
            #Q.train(),policy.eval()
            optimize_Q(Q, target_Q, policy, gamma, batch, optimizer_Q)
            #Q.eval(), policy.train()
            optimize_policy(policy, Q, batch, optimizer_pi)
        state = next_state
        if terminated or truncated:
            break

    

    #memory.clear()
    
    # Placeholder return value (to be replaced with actual calculation)
    return 0.0
