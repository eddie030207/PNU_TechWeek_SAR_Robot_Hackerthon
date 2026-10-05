import math

import numpy as np

# TurtleBot3 Burger 파라미터 (강의 노트북 "핵심 파라미터"와 Webots PROTO 기준).
WHEEL_RADIUS = 0.033      # [m]
WHEEL_SEPARATION = 0.160  # [m]
ROBOT_RADIUS = 0.105      # [m]

# LiDAR(LDS-01)는 로봇 중심에서 뒤쪽으로 0.03 m 떨어져 장착되어 있다.
LIDAR_OFFSET_X = -0.03    # [m]
LIDAR_MIN_RANGE = 0.12    # [m]
LIDAR_MAX_RANGE = 3.5     # [m]


def compass_to_yaw(compass_values):
    """Compass 값(로봇 좌표계에서 본 북쪽 벡터)을 월드 기준 yaw [rad]로 변환한다.

    Webots ENU 좌표계에서 북쪽은 +y이므로 yaw = ψ일 때 Compass는 (sin ψ, cos ψ, 0)을 반환한다.
    """
    return math.atan2(compass_values[0], compass_values[1])


def lidar_ray_angles(num_rays, fov):
    """LiDAR 각 측정값의 방향을 로봇 좌표계 각도 [rad]로 반환한다 (정면 0, 반시계 방향 +).

    getRangeImage()는 왼쪽에서 오른쪽(시계 방향) 순서이며 index 0이 후방, num_rays / 2가 정면이다.
    각 측정값은 구간의 중심 방향이다 (+0.5). 평평한 벽을 Ground Truth pose로 직선 맞춤한 결과
    +0.5일 때 기울기 오차가 0.005도였고, +0이나 +1이면 0.5도였다.
    """
    step = fov / num_rays
    return fov / 2.0 - (np.arange(num_rays) + 0.5) * step
