import math

import cv2
import numpy as np


def render_map(grid, pose=None, trajectories=(), robot_radius=0.105, view_size=720, margin_m=1.0):
    """Occupancy Grid Map을 BGR 이미지로 그린다 (위쪽이 +y, 오른쪽이 +x).

    grid: OccupancyGrid.
    pose: 로봇 (x, y, theta). 빨간 원과 방향선으로 표시한다.
    trajectories: [(월드 좌표 (x, y) 리스트, BGR 색상), ...].
    관측된 영역 주변만 잘라서 view_size 픽셀에 맞게 확대한다.
    """
    observed = np.argwhere(grid.log_odds != 0.0)
    margin = int(round(margin_m / grid.resolution))

    if len(observed) > 0:
        iy0, ix0 = observed.min(axis=0) - margin
        iy1, ix1 = observed.max(axis=0) + margin + 1
    else:
        center = grid.size // 2
        iy0 = ix0 = center - margin
        iy1 = ix1 = center + margin

    ix0, iy0 = max(int(ix0), 0), max(int(iy0), 0)
    ix1, iy1 = min(int(ix1), grid.size), min(int(iy1), grid.size)

    # 미관측 127(회색), free 255(흰색), occupied 0(검정).
    probability = grid.probability()[iy0:iy1, ix0:ix1]
    gray = ((1.0 - probability) * 255.0).astype(np.uint8)
    gray = np.flipud(gray)

    scale = max(1, view_size // max(gray.shape))
    image = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def to_pixel(x, y):
        # 픽셀 index는 픽셀 중심을 가리키므로 0.5를 뺀다.
        u = ((x - grid.origin_x) / grid.resolution - ix0) * scale - 0.5
        v = (iy1 - (y - grid.origin_y) / grid.resolution) * scale - 0.5
        return int(round(u)), int(round(v))

    for points, color in trajectories:
        if len(points) >= 2:
            pixels = np.array([to_pixel(x, y) for x, y in points], dtype=np.int32)
            cv2.polylines(image, [pixels], False, color, 1, cv2.LINE_AA)

    if pose is not None:
        x, y, theta = pose
        center = to_pixel(x, y)
        radius = max(2, int(round(robot_radius / grid.resolution * scale)))
        tip = to_pixel(x + 2.0 * robot_radius * math.cos(theta), y + 2.0 * robot_radius * math.sin(theta))
        cv2.circle(image, center, radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, center, tip, (0, 0, 255), 2, cv2.LINE_AA)

    return image
