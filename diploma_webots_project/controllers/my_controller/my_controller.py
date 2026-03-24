import math
import random
from controller import Robot
from controller import Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

# === ACCES NOD ROBOT (RESET FIZIC) ===
robot_node = robot.getFromDef("Pioneer3AT")
if robot_node is None:
    print("ERROR: Robot DEF not found!")
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# === PARAMETRI ROBOT ===
R = 0.0975
L = 0.33
MAX_WHEEL_SPEED = 6.4

# === DINAMICA SIMPLIFICATA ===
v_prev = 0.0
w_prev = 0.0

MAX_ACC_V = 1.5      # m/s^2
MAX_ACC_W = 4.0      # rad/s^2

# === MOTOARE ===
fl = robot.getDevice("front left wheel")
bl = robot.getDevice("back left wheel")
fr = robot.getDevice("front right wheel")
br = robot.getDevice("back right wheel")

motors = [fl, bl, fr, br]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

# === LIDAR ===
lidar = robot.getDevice("lidar")
lidar.enable(timestep)

# === ODOMETRIE ===
x, y, theta = 0.0, 0.0, 0.0

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

# === RESET EPISOD (CU RESET FIZIC REAL) ===
def reset_episode():
    global x, y, theta, goal_x, goal_y, step_count, prev_distance

    # 1. Oprește motoarele complet
    for m in motors:
        m.setVelocity(0.0)

    # 2. Resetează poziția
    translation_field.setSFVec3f([0.0, 0.0, 0.05])
    rotation_field.setSFRotation([0, 1, 0, 0])

    # 3. Reset fizic
    robot.simulationResetPhysics()

    # 4. Lasă un pas de stabilizare
    robot.step(timestep)

    # 5. Reset variabile interne
    x, y, theta = 0.0, 0.0, 0.0
    step_count = 0

    # Nou goal
    goal_x = random.uniform(-8, 8)
    goal_y = random.uniform(-8, 8)

    prev_distance = math.sqrt(goal_x**2 + goal_y**2)

    print("New episode started.")
    
reset_episode()

# === ACTION SPACE ===
def apply_action(action):
    global v_prev, w_prev

    # === v_target, w_target din actiune ===
    if action == 0:      # forward
        v_target = 1.0
        w_target = 0.0
    elif action == 1:    # left
        v_target = 0.0
        w_target = 2.0
    elif action == 2:    # right
        v_target = 0.0
        w_target = -2.0
    else:
        v_target = 0.0
        w_target = 0.0

    # === LIMITARE ACCELERATIE LINARA ===
    dv = v_target - v_prev
    dv = max(min(dv, MAX_ACC_V * dt), -MAX_ACC_V * dt)
    v = v_prev + dv

    # === LIMITARE ACCELERATIE ANGULARA ===
    dw = w_target - w_prev
    dw = max(min(dw, MAX_ACC_W * dt), -MAX_ACC_W * dt)
    w = w_prev + dw

    # actualizare viteze anterioare
    v_prev = v
    w_prev = w

    # === conversie la viteze roti ===
    v_left = (v - (L/2)*w) / R
    v_right = (v + (L/2)*w) / R

    v_left = max(min(v_left, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    v_right = max(min(v_right, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

    fl.setVelocity(v_left)
    bl.setVelocity(v_left)
    fr.setVelocity(v_right)
    br.setVelocity(v_right)

def get_state():
    global x, y, theta

    ranges = lidar.getRangeImage()
    n = len(ranges)

    front = ranges[n//2]
    left = ranges[3*n//4]
    right = ranges[n//4]

    dx = goal_x - x
    dy = goal_y - y
    distance = math.sqrt(dx**2 + dy**2)
    angle_to_goal = normalize_angle(math.atan2(dy, dx) - theta)

    return [front, left, right, distance, angle_to_goal]

def compute_reward(state):
    front, left, right, distance, angle = state

    global prev_distance

    reward = 0.0

    # progres către țintă
    reward += (prev_distance - distance) * 5.0

    # penalizare timp
    reward -= 0.01

    # coliziune
    if front < 0.4:
        return -100, True

    # atingere țintă
    if distance < 0.3:
        return 100, True

    prev_distance = distance

    return reward, False

# === LOOP PRINCIPAL ===
while robot.step(timestep) != -1:

    step_count += 1

    action = 0

    apply_action(action)

    # ODOMETRIE simplificată
    v = 1.0 if action == 0 else 0.0
    w = 2.0 if action == 1 else (-2.0 if action == 2 else 0.0)

    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += w * dt
    theta = normalize_angle(theta)

    state = get_state()
    reward, done = compute_reward(state)

    print("State:", state, "Reward:", reward)

    if done or step_count > 300:
        print("Episode finished. Waiting 1 second...")
        for _ in range(30):  # ~1 sec
            robot.step(timestep)
        reset_episode()