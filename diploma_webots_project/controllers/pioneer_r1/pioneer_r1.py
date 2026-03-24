import math
from controller import Robot

# ===============================
# INIT ROBOT
# ===============================
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

# ===============================
# PARAMETRI ROBOT
# ===============================
R = 0.0975      # raza roata (m)
L = 0.33        # distanta intre roti (m)
MAX_WHEEL_SPEED = 6.4

# LIMITARI
MAX_V = 1.2
MAX_W = 6.0
MAX_ACC_V = 2.0      # m/s^2
MAX_ACC_W = 6.0      # rad/s^2

# ===============================
# MOTOARE
# ===============================
fl = robot.getDevice("front left wheel")
bl = robot.getDevice("back left wheel")
fr = robot.getDevice("front right wheel")
br = robot.getDevice("back right wheel")

motors = [fl, bl, fr, br]

for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

# ===============================
# ODOMETRIE
# ===============================
x = 0.0
y = 0.0
theta = 0.0

# ===============================
# FUNCTIE NORMALIZARE UNGHI
# ===============================
def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

# ===============================
# LISTA PUNCTE TEST
# ===============================
goals = [(5,3), (-4,2), (3,-3), (-2,-4), (6,-1)]
goal_index = 0
goal_x, goal_y = goals[goal_index]

print("Start test multiple goals")

# ===============================
# MEMORIE VITEZE ANTERIOARE
# ===============================
v_prev = 0.0
w_prev = 0.0

# ===============================
# LOOP PRINCIPAL
# ===============================
while robot.step(timestep) != -1:

    # daca toate goal-urile au fost atinse
    if goal_index >= len(goals):
        for m in motors:
            m.setVelocity(0.0)
        continue

    # ===============================
    # EROARE POZITIE
    # ===============================
    dx = goal_x - x
    dy = goal_y - y

    distance_error = math.sqrt(dx**2 + dy**2)
    angle_to_goal = math.atan2(dy, dx)
    angle_error = normalize_angle(angle_to_goal - theta)

    # ===============================
    # CONTROL POLAR
    # ===============================
    k_rho = 1.0
    k_alpha = 4.0
    angle_threshold = 0.3

    if abs(angle_error) > angle_threshold:
        v = 0.0
        w = k_alpha * angle_error
    else:
        v = k_rho * distance_error
        w = k_alpha * angle_error

    # ===============================
    # LIMITARE ACCELERATIE
    # ===============================
    dv = v - v_prev
    dv = max(min(dv, MAX_ACC_V * dt), -MAX_ACC_V * dt)
    v = v_prev + dv

    dw = w - w_prev
    dw = max(min(dw, MAX_ACC_W * dt), -MAX_ACC_W * dt)
    w = w_prev + dw

    # ===============================
    # LIMITARE VITEZA
    # ===============================
    v = max(min(v, MAX_V), -MAX_V)
    w = max(min(w, MAX_W), -MAX_W)

    # salvam dupa limitari
    v_prev = v
    w_prev = w

    # ===============================
    # CONVERSIE v,w → VITEZE ROTI
    # ===============================
    v_left = (v - (L/2.0) * w) / R
    v_right = (v + (L/2.0) * w) / R

    v_left = max(min(v_left, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    v_right = max(min(v_right, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

    fl.setVelocity(v_left)
    bl.setVelocity(v_left)
    fr.setVelocity(v_right)
    br.setVelocity(v_right)

    # ===============================
    # ODOMETRIE (model cinematic)
    # ===============================
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += w * dt
    theta = normalize_angle(theta)

    # ===============================
    # VERIFICARE TINTA
    # ===============================
    if distance_error < 0.15:
        print(f"Reached goal {goal_index+1}: ({goal_x},{goal_y})")
        goal_index += 1

        if goal_index < len(goals):
            goal_x, goal_y = goals[goal_index]
        else:
            print("All goals completed.")

    # DEBUG
    print(f"x={x:.2f}, y={y:.2f}, theta={theta:.2f}, goal=({goal_x},{goal_y})")