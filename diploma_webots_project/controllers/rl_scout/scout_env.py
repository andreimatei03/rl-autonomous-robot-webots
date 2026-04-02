import math
import random


class ScoutEnv:

    def __init__(self, robot, motors, lidar,
                 translation_field, rotation_field,
                 timestep):

        self.robot = robot
        self.motors = motors
        self.lidar = lidar
        self.translation_field = translation_field
        self.rotation_field = rotation_field
        self.timestep = timestep
        self.dt = timestep / 1000.0

        # Parametri robot
        self.R = 0.0975
        self.L = 0.33
        self.MAX_WHEEL_SPEED = 6.4

        # Dinamica
        self.v = 0.0
        self.w = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0

        self.MAX_ACC_V = 1.5
        self.MAX_ACC_W = 4.0

        # Odometrie internă
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.prev_distance = 0.0
        self.step_count = 0
        self.max_steps = 1200

        self.state_dim = 9  # front, left, right, front_left, front_right, distance, angle
        self.action_dim = 5

    # -----------------------------

    def reset(self):

        # Oprire motoare
        for m in self.motors:
            m.setVelocity(0.0)

        # Reset poziție
        self.translation_field.setSFVec3f([0.0, 0.0, 0.05])
        self.rotation_field.setSFRotation([0, 1, 0, 0])

        self.robot.simulationResetPhysics()
        self.robot.step(self.timestep)

        # Reset variabile
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.v = 0.0
        self.w = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0

        self.step_count = 0

        # Goal random
        self.goal_x = random.uniform(-13, 13)
        self.goal_y = random.uniform(-13, 13)

        self.prev_distance = math.sqrt(
            self.goal_x ** 2 + self.goal_y ** 2
        )

        return self._get_state()

    # -----------------------------

    def step(self, action):

        self.step_count += 1

        self._apply_action(action)

        self.robot.step(self.timestep)

        self._update_odometry()

        state = self._get_state()
        reward, done = self._compute_reward(state)

        if self.step_count >= self.max_steps:
            done = True

        return state, reward, done

    # -----------------------------

    def _apply_action(self, action):

    #0 - forward
    #1 - forward-left
    #2 - forward-right
    #3 - left
    #4 - right

        if action == 0:      # forward
            v_target = 1.0
            w_target = 0.0
        elif action == 1:    # forward-left
            v_target = 1.0
            w_target = 1.5
        elif action == 2:    # forward-right
            v_target = 1.0
            w_target = -1.5
        elif action == 3:    # rotate left
            v_target = 0.0
            w_target = 2.0
        elif action == 4:    # rotate right
            v_target = 0.0
            w_target = -2.0

        # limitare accelerație
        dv = v_target - self.v_prev
        dv = max(min(dv, self.MAX_ACC_V * self.dt),
                 -self.MAX_ACC_V * self.dt)
        self.v = self.v_prev + dv

        dw = w_target - self.w_prev
        dw = max(min(dw, self.MAX_ACC_W * self.dt),
                 -self.MAX_ACC_W * self.dt)
        self.w = self.w_prev + dw

        self.v_prev = self.v
        self.w_prev = self.w

        # conversie roți
        v_left = (self.v - (self.L / 2.0) * self.w) / self.R
        v_right = (self.v + (self.L / 2.0) * self.w) / self.R

        v_left = max(min(v_left, self.MAX_WHEEL_SPEED),
                     -self.MAX_WHEEL_SPEED)
        v_right = max(min(v_right, self.MAX_WHEEL_SPEED),
                      -self.MAX_WHEEL_SPEED)

        self.motors[0].setVelocity(v_left)
        self.motors[1].setVelocity(v_left)
        self.motors[2].setVelocity(v_right)
        self.motors[3].setVelocity(v_right)

    # -----------------------------

    def _update_odometry(self):

        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta += self.w * self.dt
        self.theta = math.atan2(
            math.sin(self.theta),
            math.cos(self.theta)
        )

    # -----------------------------

    def _get_state(self):

        ranges = self.lidar.getRangeImage()
        n = len(ranges)

        # Senzori: front, left, right
        front = ranges[n // 2]
        left = ranges[3 * n // 4]
        right = ranges[n // 4]
        
        # Senzori adiționali pentru robustețe
        front_left = ranges[5 * n // 8]
        front_right = ranges[3 * n // 8]

        # Normalizare lidar
        max_range = 10.0
        front = min(front, max_range) / max_range
        left = min(left, max_range) / max_range
        right = min(right, max_range) / max_range
        front_left = min(front_left, max_range) / max_range
        front_right = min(front_right, max_range) / max_range
        
        # Min distanță pe toate senzori
        min_sensor = min(front, left, right, front_left, front_right)

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx) - self.theta
        angle = math.atan2(math.sin(angle), math.cos(angle))

        # Normalizare obiectiv
        distance_norm = min(distance, 40.0) / 40.0
        angle_norm = angle / math.pi

        v_norm = self.v / 2.0
        w_norm = self.w / 4.0

        return [
            front,
            left,
            right,
            front_left,
            front_right,
            distance_norm,
            angle_norm,
            v_norm,  # <--- Trebuie să adaugi această linie
            w_norm   # <--- Trebuie să adaugi această linie
        ]

    # -----------------------------

    def _compute_reward(self, state):
        front, left, right, front_left, front_right, distance_norm, angle_norm, v_norm, w_norm = state
        
        reward = 0.0

        # 1. STRONG progress reward (PRIMARY incentive)
        distance = math.sqrt(
            (self.goal_x - self.x) ** 2 + 
            (self.goal_y - self.y) ** 2
        )
        progress = self.prev_distance - distance
        reward += progress * 20.0 #Q-values stability

        # 2. Orientation reward (secondary)
        if abs(angle_norm) < 0.15:
            reward += 1.0  # BOOSTED: 0.5 → 1.0
        reward -= abs(angle_norm) * 0.02  # Softer penalty

        # 3. Safe movement bonus (scăzut de la 3 metri la 1.5 metri)
        min_dist = min(front, left, right, front_left, front_right)
        if min_dist > 0.15: # 0.15 * 10m = 1.5 metri
            reward += 0.2  
        
        # 4. Obstacle approaching warnings (distanțe mult mai mici)
        if front < 0.10:    # 1.0 metri (înainte era 2.5m)
            reward -= 1.0  
        if front < 0.05:    # 0.5 metri (înainte era 1.5m)
            reward -= 2.0  
        
        # Penalizare laterală ajustată la 0.5 metri
        if (left < 0.05 or right < 0.05 or 
            front_left < 0.05 or front_right < 0.05):
            reward -= 0.1
        
        # 5. Minimal living cost
        reward -= 0.0005  # REDUCED: 0.001 → 0.0005 (even softer)
        reward -= abs(self.w) * 0.002  # REDUCED: 0.005 → 0.002

        # 6. COLLISION (Permite-i să se apropie la 25 de centimetri)
        if min_dist < 0.025: # 0.025 * 10m = 25 centimetri de perete
            return -20.0, True

        # 7. GOAL reached (STRONG reward)
        if distance < 1:
            return 200.0, True  # BOOSTED: 150 → 200 (strong success signal)

        self.prev_distance = distance

        return reward, False