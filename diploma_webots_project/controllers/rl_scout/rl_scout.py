import torch
import numpy as np
import os
from collections import deque
from dqn_agent import DQNAgent
from scout_env import ScoutEnv
from controller import Supervisor


# ===============================
# INIT WEBOTS
# ===============================

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Robot node
robot_node = robot.getFromDef("Pioneer3AT")
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# Motoare
fl = robot.getDevice("front left wheel")
bl = robot.getDevice("back left wheel")
fr = robot.getDevice("front right wheel")
br = robot.getDevice("back right wheel")

motors = [fl, bl, fr, br]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

# Lidar
lidar = robot.getDevice("lidar")
lidar.enable(timestep)


# ===============================
# ENVIRONMENT & AGENT
# ===============================

env = ScoutEnv(
    robot,
    motors,
    lidar,
    translation_field,
    rotation_field,
    timestep
)

state_dim = env.state_dim
action_dim = env.action_dim

agent = DQNAgent(state_dim, action_dim)

# Load existing model if available
model_path = "dqn_model.pth"
if os.path.exists(model_path):
    try:
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        print(f"✓ Model încărcat din {model_path}")
    except RuntimeError as e:
        print(f"⚠ Model incompatibil (arhitectură diferită): {str(e)[:100]}")
        print("→ Inițiaz model nou din arhitectura actuală")
        # Start fresh - old model architecture doesn't match new Dueling DQN


# ===============================
# TRAINING LOOP
# ===============================

num_episodes = 1000
max_steps = 1200

# Monitoring
episode_rewards = deque(maxlen=50)
best_reward = -float('inf')

for episode in range(num_episodes):

    state = env.reset()
    total_reward = 0

    for step in range(max_steps):

        robot.step(timestep)

        action = agent.select_action(state)

        next_state, reward, done = env.step(action)

        agent.store(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards)
    
    # Epsilon decay per episod (combinat cu decay din select_action)
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    
    # Monitoring și salvare
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1:4d} | Reward: {total_reward:8.2f} | "
              f"Avg50: {avg_reward:8.2f} | Epsilon: {agent.epsilon:.4f} | "
              f"Buffer: {len(agent.replay_buffer):6d} | "
              f"Goal: ({env.goal_x:6.2f}, {env.goal_y:6.2f}) | "
              f"RobotPos: ({env.x:6.2f}, {env.y:6.2f})")
    
    # Salvare checkpoint best model
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.q_network.state_dict(), "dqn_model_best.pth")
    
    # Salvare periodică
    if (episode + 1) % 100 == 0:
        torch.save(agent.q_network.state_dict(), f"dqn_model_ep{episode+1}.pth")

# ===============================
# SAVE FINAL MODEL
# ===============================

torch.save(agent.q_network.state_dict(), "dqn_model.pth")
print("\nAntrenament încheiat!")
print(f"Model final salvat în dqn_model.pth")
print(f"Best model salvat în dqn_model_best.pth (reward: {best_reward:.2f})")