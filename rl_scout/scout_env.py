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
        self.max_steps = 300

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
        self.goal_x = random.uniform(-8, 8)
        self.goal_y = random.uniform(-8, 8)

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

        # 0 forward
        # 1 left
        # 2 right
        # 3 stop

        if action == 0:
            v_target = 1.0
            w_target = 0.0
        elif action == 1:
            v_target = 0.0
            w_target = 2.0
        elif action == 2:
            v_target = 0.0
            w_target = -2.0
        else:
            v_target = 0.0
            w_target = 0.0

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

        front = ranges[n // 2]
        left = ranges[3 * n // 4]
        right = ranges[n // 4]

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx) - self.theta
        angle = math.atan2(math.sin(angle),
                           math.cos(angle))

        return [
            front,
            left,
            right,
            distance,
            angle,
            self.v,
            self.w
        ]

    # -----------------------------

    def _compute_reward(self, state):

        front, left, right, distance, angle, v, w = state

        reward = 0.0

        # progres
        reward += (self.prev_distance - distance) * 5.0

        # penalizare timp
        reward -= 0.01

        # coliziune
        if front < 0.4:
            return -100.0, True

        # țintă atinsă
        if distance < 0.3:
            return 100.0, True

        self.prev_distance = distance

        return reward, False