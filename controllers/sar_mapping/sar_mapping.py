import math
import os
import sys

from controller import Robot, Keyboard
import cv2
import numpy as np

from occupancy_grid import OccupancyGrid
from odometry import Odometry
from tb3_params import (
    WHEEL_RADIUS, WHEEL_SEPARATION, ROBOT_RADIUS,
    LIDAR_OFFSET_X, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE,
    compass_to_yaw, lidar_ray_angles,
)
from visualization import render_map


# 제공되는 시작 지점 (position & orientation). controllerArgs로 "x y theta"를 넘기면 그 값을 사용한다.
START_X = -0.3
START_Y = -7.5
START_THETA = math.pi

if len(sys.argv) >= 4:
    START_X, START_Y, START_THETA = (float(value) for value in sys.argv[1:4])

# 지도 설정.
MAP_SIZE = 40.0        # [m] 시작 지점을 중심으로 한 정사각형 지도의 한 변.
MAP_RESOLUTION = 0.05  # [m/cell]

# 바퀴 회전 속도 [rad/s]
SPEED = 3.0


robot = Robot()

timestep = int(robot.getBasicTimeStep())

# 키보드 활성화
keyboard = Keyboard()
keyboard.enable(timestep)

# 모터 가져오기
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

# 속도 제어 모드로 변경
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

# 초기 정지
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Wheel Encoder 활성화
left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
left_encoder.enable(timestep)
right_encoder.enable(timestep)

# Compass 활성화
compass = robot.getDevice("compass")
compass.enable(timestep)

# LiDAR 활성화
lidar = robot.getDevice("LDS-01")
lidar.enable(timestep)

ray_angles = lidar_ray_angles(lidar.getHorizontalResolution(), lidar.getFov())

# 위치 추정 및 지도 작성
odometry = Odometry(START_X, START_Y, START_THETA, WHEEL_RADIUS, WHEEL_SEPARATION)
grid = OccupancyGrid(START_X, START_Y, MAP_SIZE, MAP_RESOLUTION)

trajectory = [(START_X, START_Y)]

output_dir = os.path.dirname(os.path.abspath(__file__))

prev_print_time = 0.0
prev_draw_time = 0.0
prev_save_pressed = False

print("W/A/S/D: 이동, M: 지도 저장 (map.png, map.npz)")

while robot.step(timestep) != -1:
    curr_time = robot.getTime()

    # --- 키보드 조종 ---
    key = keyboard.getKey()

    left_speed = 0.0
    right_speed = 0.0

    if key == ord("w") or key == ord("W"):
        left_speed = SPEED
        right_speed = SPEED

    elif key == ord("s") or key == ord("S"):
        left_speed = -SPEED
        right_speed = -SPEED

    elif key == ord("a") or key == ord("A"):
        left_speed = -SPEED
        right_speed = SPEED

    elif key == ord("d") or key == ord("D"):
        left_speed = SPEED
        right_speed = -SPEED

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # --- 1. 위치 추정 (Wheel Encoder + Compass) ---
    x, y, theta = odometry.update(
        left_encoder.getValue(),
        right_encoder.getValue(),
        compass_to_yaw(compass.getValues()),
    )

    if not odometry.ready:
        continue

    if math.hypot(x - trajectory[-1][0], y - trajectory[-1][1]) >= MAP_RESOLUTION:
        trajectory.append((x, y))

    # --- 2. 지도 작성 (LiDAR + Occupancy Grid) ---
    ranges = lidar.getRangeImage()

    if ranges and len(ranges) == len(ray_angles):
        lidar_x = x + LIDAR_OFFSET_X * math.cos(theta)
        lidar_y = y + LIDAR_OFFSET_X * math.sin(theta)
        grid.update(lidar_x, lidar_y, theta, ranges, ray_angles, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE)

    # --- 결과 확인 ---
    if curr_time - prev_print_time >= 1.0:
        print(f"x: {x:.3f} m, y: {y:.3f} m, theta: {math.degrees(theta):.1f} deg")
        prev_print_time = curr_time

    if curr_time - prev_draw_time >= 0.25:
        map_image = render_map(grid, (x, y, theta), [(trajectory, (255, 0, 0))], ROBOT_RADIUS)
        cv2.imshow("Occupancy Grid Map", map_image)
        cv2.waitKey(1)
        prev_draw_time = curr_time

    save_pressed = key == ord("m") or key == ord("M")

    if save_pressed and not prev_save_pressed:
        map_image = render_map(grid, (x, y, theta), [(trajectory, (255, 0, 0))], ROBOT_RADIUS)
        cv2.imwrite(os.path.join(output_dir, "map.png"), map_image)
        np.savez(
            os.path.join(output_dir, "map.npz"),
            log_odds=grid.log_odds,
            origin=np.array([grid.origin_x, grid.origin_y]),
            resolution=grid.resolution,
            pose=np.array([x, y, theta]),
        )
        print(f"지도를 저장했습니다: {output_dir}")

    prev_save_pressed = save_pressed

cv2.destroyAllWindows()
