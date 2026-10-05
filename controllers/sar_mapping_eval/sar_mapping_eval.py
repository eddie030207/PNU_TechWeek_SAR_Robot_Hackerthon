"""위치 추정과 지도 작성의 정확도를 검증하는 컨트롤러 (검증 전용).

Supervisor로 로봇의 실제 위치(Ground Truth)를 읽어 sar_mapping의 추정값과 비교한다.
로봇은 LiDAR 기반 단순 장애물 회피로 스스로 돌아다닌다.
대회용 컨트롤러가 아니며, worlds/apartment_mapping_eval.wbt에서만 사용한다.
"""
import math
import os
import sys

from controller import Supervisor
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sar_mapping"))

from occupancy_grid import OccupancyGrid
from odometry import Odometry, wrap_angle
from tb3_params import (
    WHEEL_RADIUS, WHEEL_SEPARATION, ROBOT_RADIUS,
    LIDAR_OFFSET_X, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE,
    compass_to_yaw, lidar_ray_angles,
)
from visualization import render_map


# 검증 시간 [s]. controllerArgs로 바꿀 수 있다.
DURATION = float(sys.argv[1]) if len(sys.argv) >= 2 else 180.0

BASE_SPEED = 4.0          # [rad/s]
DISTANCE_THRESHOLD = 0.4  # [m] 정면이 이 거리보다 가까우면 회전한다.
CLEAR_THRESHOLD = 0.6     # [m] 정면이 이 거리보다 멀어지면 다시 직진한다.


robot = Supervisor()
robot_node = robot.getSelf()

timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
left_encoder.enable(timestep)
right_encoder.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

lidar = robot.getDevice("LDS-01")
lidar.enable(timestep)

num_rays = lidar.getHorizontalResolution()
ray_angles = lidar_ray_angles(num_rays, lidar.getFov())


def ground_truth():
    position = robot_node.getPosition()
    orientation = robot_node.getOrientation()
    return position[0], position[1], math.atan2(orientation[3], orientation[0])


# 시작 지점은 제공되는 정보이므로 Ground Truth의 첫 값을 사용한다.
robot.step(timestep)
start_x, start_y, start_theta = ground_truth()

odometry = Odometry(start_x, start_y, start_theta, WHEEL_RADIUS, WHEEL_SEPARATION)
encoder_only = Odometry(start_x, start_y, start_theta, WHEEL_RADIUS, WHEEL_SEPARATION)
grid = OccupancyGrid(start_x, start_y)

estimated_path = [(start_x, start_y)]
true_path = [(start_x, start_y)]

position_errors = []
heading_errors = []
encoder_only_position_errors = []
encoder_only_heading_errors = []

turning = False
turn_dir = 1

# 막힘 감지 (LiDAR 높이보다 낮은 물체에 걸린 경우 후진 후 회전한다).
stuck_check_time = 0.0
stuck_check_position = (start_x, start_y)
escape_until = 0.0

prev_log_time = 0.0

while robot.step(timestep) != -1:
    curr_time = robot.getTime()

    if curr_time >= DURATION:
        break

    left_position = left_encoder.getValue()
    right_position = right_encoder.getValue()
    compass_yaw = compass_to_yaw(compass.getValues())

    x, y, theta = odometry.update(left_position, right_position, compass_yaw)
    enc_x, enc_y, enc_theta = encoder_only.update(left_position, right_position)
    true_x, true_y, true_theta = ground_truth()

    if not odometry.ready:
        continue

    # 유효하지 않은 scan은 빈 공간으로 해석하지 않고 정지한다.
    ranges = lidar.getRangeImage()
    if not ranges or len(ranges) != num_rays or np.isnan(ranges).any():
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        continue

    lidar_x = x + LIDAR_OFFSET_X * math.cos(theta)
    lidar_y = y + LIDAR_OFFSET_X * math.sin(theta)
    grid.update(lidar_x, lidar_y, theta, ranges, ray_angles, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE)

    estimated_path.append((x, y))
    true_path.append((true_x, true_y))

    position_errors.append(math.hypot(x - true_x, y - true_y))
    heading_errors.append(abs(wrap_angle(theta - true_theta)))
    encoder_only_position_errors.append(math.hypot(enc_x - true_x, enc_y - true_y))
    encoder_only_heading_errors.append(abs(wrap_angle(enc_theta - true_theta)))

    # --- LiDAR 기반 단순 장애물 회피 주행 ---
    distances = np.minimum(np.asarray(ranges, dtype=np.float64), LIDAR_MAX_RANGE)
    center = num_rays // 2
    front = distances[center - 30:center + 31].min()
    left = distances[center - 90:center - 30].mean()
    right = distances[center + 31:center + 91].mean()

    if curr_time - stuck_check_time >= 4.0:
        moved = math.hypot(true_x - stuck_check_position[0], true_y - stuck_check_position[1])
        if moved < 0.05 and curr_time >= escape_until:
            escape_until = curr_time + 3.0
            turn_dir = -turn_dir
        stuck_check_time = curr_time
        stuck_check_position = (true_x, true_y)

    if curr_time < escape_until:
        if escape_until - curr_time > 1.5:
            left_speed = right_speed = -BASE_SPEED
        else:
            left_speed = -turn_dir * BASE_SPEED
            right_speed = turn_dir * BASE_SPEED
    elif turning:
        if front > CLEAR_THRESHOLD:
            turning = False
        left_speed = -turn_dir * BASE_SPEED * 0.5
        right_speed = turn_dir * BASE_SPEED * 0.5
    elif front < DISTANCE_THRESHOLD:
        turning = True
        turn_dir = 1 if left > right else -1
        left_speed = right_speed = 0.0
    else:
        left_speed = right_speed = BASE_SPEED

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    if curr_time - prev_log_time >= 10.0:
        print(
            f"t={curr_time:6.1f}s  위치 오차: {position_errors[-1]:.3f} m  "
            f"방향 오차: {math.degrees(heading_errors[-1]):.2f} deg  "
            f"(Encoder만 사용 시: {encoder_only_position_errors[-1]:.3f} m, "
            f"{math.degrees(encoder_only_heading_errors[-1]):.2f} deg)",
            flush=True,
        )
        prev_log_time = curr_time

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

if not position_errors:
    print("유효한 LiDAR scan이 없어 검증 결과가 없습니다.", flush=True)
    robot.simulationQuit(1)
    sys.exit(1)

path_length = sum(
    math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(true_path[:-1], true_path[1:])
)

summary = "\n".join([
    "===== sar_mapping 검증 결과 =====",
    f"주행 시간: {robot.getTime():.1f} s, 실제 주행 거리: {path_length:.2f} m",
    "[Encoder + Compass]",
    f"  위치 오차  최종: {position_errors[-1]:.3f} m, 평균: {np.mean(position_errors):.3f} m, 최대: {np.max(position_errors):.3f} m",
    f"  방향 오차  최종: {math.degrees(heading_errors[-1]):.2f} deg, 최대: {math.degrees(np.max(heading_errors)):.2f} deg",
    "[Encoder만 사용]",
    f"  위치 오차  최종: {encoder_only_position_errors[-1]:.3f} m, 평균: {np.mean(encoder_only_position_errors):.3f} m, 최대: {np.max(encoder_only_position_errors):.3f} m",
    f"  방향 오차  최종: {math.degrees(encoder_only_heading_errors[-1]):.2f} deg, 최대: {math.degrees(np.max(encoder_only_heading_errors)):.2f} deg",
])
print(summary, flush=True)

output_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(output_dir, "eval_result.txt"), "w") as f:
    f.write(summary + "\n")

# 초록: 실제 경로, 파랑: 추정 경로.
map_image = render_map(
    grid,
    odometry.pose,
    [(true_path, (0, 200, 0)), (estimated_path, (255, 0, 0))],
    ROBOT_RADIUS,
    view_size=1200,
)
cv2.imwrite(os.path.join(output_dir, "eval_map.png"), map_image)

robot.simulationQuit(0)
