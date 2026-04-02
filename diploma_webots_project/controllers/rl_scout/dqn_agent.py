import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==============================
# Q-Network (Dueling Architecture)
# ==============================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ==============================
# Replay Buffer
# ==============================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ==============================
# DQN Agent
# ==============================

class DQNAgent:
    def __init__(self, state_dim, action_dim):

        self.device = torch.device("cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Decai mai lent pentru explorare mai bună

        self.batch_size = 64

        self.replay_buffer = ReplayBuffer(50000)  # Buffer mai mare

        self.update_target_every = 500  # Update mai des pentru mai multă stabilitate
        self.step_counter = 0
        
        self.clip_grad_norm = 10.0  # Gradient clipping pentru stabilitate

    # -----------------------------

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Set to eval mode for inference (batch norm needs this for single sample)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()

        return q_values.argmax().item()

    # -----------------------------

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # -----------------------------

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a)
        current_q = self.q_network(states).gather(1, actions)

        # Double DQN: Folosim q_network pentru selectare, target_network pentru evaluare
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping pentru stabilitate
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clip_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()

        self.step_counter += 1

        # Update target network
        if self.step_counter % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())