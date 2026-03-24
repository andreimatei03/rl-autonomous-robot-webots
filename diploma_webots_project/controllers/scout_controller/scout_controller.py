import sys
import os
import random

from controller import Supervisor

# -------------------------------------------------
# Adăugăm path către rl_scout
# -------------------------------------------------

project_root = r"C:\Users\HeXer36\Documents\Proiect Licenta\diploma_webots_project\controllers"
sys.path.append(project_root)

from rl_scout.scout_env import ScoutEnv

# -------------------------------------------------
# Inițializare Webots
# -------------------------------------------------

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Obținem robot node
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

# -------------------------------------------------
# Creare environment
# -------------------------------------------------

env = ScoutEnv(robot,
               motors,
               lidar,
               translation_field,
               rotation_field,
               timestep)

# -------------------------------------------------
# TEST RANDOM AGENT
# -------------------------------------------------

state = env.reset()

episode = 0

while robot.step(timestep) != -1:

    action = 0

    next_state, reward, done = env.step(action)

    if done:
        episode += 1
        print(f"Episode {episode} finished")
        state = env.reset()
    else:
        state = next_state