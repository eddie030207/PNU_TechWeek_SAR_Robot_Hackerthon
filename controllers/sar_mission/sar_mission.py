"""Autonomous Search and Rescue 컨트롤러 (TurtleBot3 Burger, Webots R2025a).

미지의 환경을 탐색하며 지도를 작성하고, 지정된 색의 사과(기본: 빨간 사과 2개)를 모두 찾아
각 사과 근처까지 이동한 뒤 시작 지점으로 복귀한다.

파이프라인: Perception -> Localization -> Mapping -> Planning -> Control
  1. Localization : Wheel Encoder(이동 거리) + Compass(방향) Odometry, Scan-to-Map Matching 보정, 위치 재탐색
  2. Mapping      : LiDAR log-odds Occupancy Grid, 카메라 시야 Coverage, 카메라로 찾은 카펫, 보이지 않는 장애물
  3. Perception   : HSV 색 분할 + 원형도 + 바닥 기하 조건으로 사과 탐지, Scan-to-Scan 비교로 움직이는 사람 추적
  4. Planning     : Inflation Costmap + A* (Global), 시야 기반 Frontier 탐색
  5. Control      : Look-ahead(Pure Pursuit) 경로 추종, LiDAR 안전 속도 제한, 사람 양보와 회피, 막힘 복구
  6. Decision     : FSM (SPIN -> EXPLORE/LOOK -> APPROACH -> CONFIRM -> RETURN -> DONE)
  7. Human-Robot  : 텍스트 명령(키보드, 파일, 실행 인자), 실시간 대시보드, 임무 보고서

사용 센서: Wheel Encoder, Compass, Accelerometer, LiDAR(LDS-01), Camera. GPS와 Supervisor는 사용하지 않는다.
필요 패키지: numpy, opencv-python (강의 노트북과 같은 버전에서 확인).
"""
import heapq
import json
import math
import os
import re
import sys
from collections import deque

from controller import Robot, Keyboard
import cv2
import numpy as np


# =====================================================================================
# 0. 설정
# =====================================================================================
# 제공되는 시작 지점 (position & orientation). 실행 인자 "start=x,y,theta"로 바꿀 수 있다.
START_X = -0.3
START_Y = -7.5
START_THETA = math.pi

# 임무 기본값. 텍스트 명령으로 바꿀 수 있다 (예: "find 2 red apples").
TARGET_COLOR = "red"
TARGET_COUNT = 2
TIME_LIMIT = 0.0          # [s] 0이면 제한 없음. 남은 시간이 복귀에 필요한 시간보다 적어지면 복귀한다.

# TurtleBot3 Burger
WHEEL_RADIUS = 0.033      # [m]
WHEEL_SEPARATION = 0.160  # [m]
TURN_SEPARATION = 0.178   # [m] 회전 명령 변환에 쓰는 유효 바퀴 간격 (제자리 회전 실측값).
ROBOT_RADIUS = 0.105      # [m]
MAX_WHEEL_SPEED = 6.4     # [rad/s] (모터 한계 6.67)

# LiDAR (LDS-01): 로봇 중심에서 0.03 m 뒤에 장착.
LIDAR_OFFSET_X = -0.03
LIDAR_MIN_RANGE = 0.12
LIDAR_MAX_RANGE = 3.5

# Camera: 로봇 중심에서 0.02 m 앞, 바닥에서 0.073 m 높이, 수평 시야각 60도.
CAM_OFFSET_X = 0.02
CAM_HEIGHT = 0.073
CAM_RANGE = 3.3           # [m] 이 거리 안의 바닥은 "카메라로 확인함"으로 기록한다.

# 사과: 반지름 0.05 m, 중심 높이 0.05 m. 꼭지를 포함한 외접원 기준 유효 반지름은 0.054 m (실측).
APPLE_RADIUS = 0.054
APPLE_CENTER_Z = 0.05
DETECT_MAX_RANGE = 4.0    # [m]

# 어두운 바닥(카펫): 밝기 V가 이 값보다 낮은 바닥은 턱이 있는 깔개로 보고 들어가지 않는다 (나무/타일 바닥은 80 이상).
DARK_FLOOR_VALUE = 50
ROUGH_CONFIRM = 3         # 이 횟수 이상 어둡게 관측된 셀을 주행 불가로 본다.

# OpenCV HSV 범위 (H: 0~179).
COLOR_RANGES = {
    "red": [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (179, 255, 255))],
    "orange": [((14, 120, 80), (30, 255, 255))],
    "green": [((35, 80, 60), (85, 255, 255))],
    "purple": [((125, 100, 60), (155, 255, 255))],
}
COLOR_BGR = {"red": (0, 0, 255), "orange": (0, 165, 255), "green": (0, 170, 0), "purple": (200, 0, 160)}
COLOR_WORDS = {
    "red": ("red", "빨간", "빨강", "붉은"),
    "orange": ("orange", "주황"),
    "green": ("green", "초록", "녹색"),
    "purple": ("purple", "보라"),
}

# Occupancy Grid
MAP_SIZE = 40.0           # [m] 시작 지점을 중심으로 한 정사각형 지도의 한 변.
RES = 0.05                # [m/cell]
L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -4.0, 4.0
L_OCC_THRESHOLD = math.log(0.65 / 0.35)
L_FREE_THRESHOLD = math.log(0.35 / 0.65)

# Costmap / Planning
R_LETHAL = ROBOT_RADIUS + 0.03   # [m] 로봇 중심이 정적 장애물에 이보다 가까워지면 안 된다 (로봇 옆면에서 3 cm).
R_SOFT = 0.30                    # [m] 이 거리 안에서는 장애물에 가까울수록 비용이 커진다 (안전거리 확보).
PROXIMITY_WEIGHT = 3.0
MIN_CLUSTER_CELLS = 16           # 카메라로 보지 못한 빈 공간이 이보다 작으면 무시한다 (정밀 탐색에서는 6).
HIDDEN_AREA_CELLS = 12           # 이보다 작은 미관측 영역은 무시한다 (사과 하나가 숨을 수 없는 크기).
LOOK_TRIGGER_CELLS = 120         # 제자리에서 볼 수 있는 미확인 셀이 이보다 많으면 회전해서 확인한다.
VIEW_DISTANCE = 0.6              # [m] 미탐색 영역에서 이 거리 안의 도달 가능한 셀을 탐색 목표로 삼는다.
GAIN_WEIGHT = 0.5                # 넓은 미탐색 영역을 우선하는 가중치.
OUTER_RATIO = 0.45               # 주변 1.5 m의 미관측 비율이 이보다 크면 지도 바깥쪽으로 본다.
OUTER_PENALTY = 6.0              # [m] 지도 바깥쪽의 탐색 대상을 뒤로 미루는 벌점.
TRAIL_RADIUS = 0.25              # [m] 로봇이 지나간 자리로 기록하는 반지름.
TRAIL_COST = 3                   # 탐색 목표를 고를 때 이미 지나간 자리는 3배 먼 것으로 계산한다 (되돌아가는 탐색을 줄인다).
TRAIL_PATH_COST = 1.5            # 탐색 경로를 계획할 때 이미 지나간 자리에 곱하는 비용 (다른 길이 있으면 그 길로 간다).
RIGHT_WEIGHT = 0.8               # [m] 진행 방향의 오른쪽에 있는 탐색 대상을 먼저 고르는 가중치 (오른손 법칙).

# Control
V_MAX = 0.20              # [m/s]
W_MAX = 1.5               # [rad/s]
LOOKAHEAD = 0.30          # [m]
GOAL_TOLERANCE = 0.08     # [m]
APPROACH_RING = (0.35, 0.50)  # [m] 사과 중심에서 이 거리의 지점까지 접근한다.
REACHED_DISTANCE = 0.80       # [m] 이 거리 안에서 사과를 카메라로 확인하면 도착으로 인정한다.

# Scan Matching (Scan-to-Map)
MATCH_STATIC_HITS = 10    # 이 횟수 이상 관측된 장애물 셀만 기준 지도로 쓴다.
MATCH_CAP = 0.15          # [m] 지도의 장애물에서 이 거리 안에 있는 scan 점(inlier)만 정합에 쓴다.
MATCH_PENALTY = 0.05      # 보정량 1 m당 점수 벌점. 복도처럼 구분이 안 되는 방향으로는 움직이지 않게 한다.
MATCH_GAIN = 0.3          # 한 번에 반영하는 보정 비율.

# 사람(동적 장애물)
PERSON_RADIUS = 0.48      # [m] 사람 주변에 두는 계획상의 여유 (R_LETHAL이 더해져 중심 간 0.6 m 이상을 유지한다).
PERSON_SLOW_DISTANCE = 1.8
PERSON_YIELD_DISTANCE = 1.0
PERSON_DANGER_DISTANCE = 0.8   # [m] 사람의 예상 경로가 이 거리 안으로 들어오면 미리 비켜 준다.
PERSON_HORIZON = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)   # [s] 사람 위치를 예측하는 시점.


def wrap_angle(angle):
    """각도를 (-π, π] 범위로 정규화한다."""
    return math.atan2(math.sin(angle), math.cos(angle))


# =====================================================================================
# 1. Localization: Wheel Encoder + Compass Odometry
# =====================================================================================
class Odometry:
    """이동 거리는 Encoder로, 방향은 Compass로 계산한다.

    Encoder만으로 회전각을 구하면 바퀴 미끄러짐 때문에 제자리 회전에서 약 11% 오차가 누적된다.
    Compass는 절대 방향을 주므로 방향 오차가 누적되지 않는다.
    """

    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, wrap_angle(theta)
        self.distance = 0.0
        self.forward = 0.0           # Encoder 기준 누적 전진 거리 (헛바퀴 감지에 쓴다).
        self._prev_left = None
        self._prev_right = None
        self._heading_offset = None  # 시작 방향과 Compass yaw의 차이.

    @property
    def ready(self):
        return self._prev_left is not None

    def update(self, left_position, right_position, compass_values):
        # Webots ENU 좌표계에서 Compass는 (sin ψ, cos ψ, 0)을 반환한다.
        compass_yaw = math.atan2(compass_values[0], compass_values[1])
        if math.isnan(compass_yaw):
            compass_yaw = None
        elif self._heading_offset is None:
            self._heading_offset = wrap_angle(self.theta - compass_yaw)

        # 센서는 첫 샘플링 전까지 NaN을 반환한다.
        if math.isnan(left_position) or math.isnan(right_position):
            return
        if self._prev_left is None:
            self._prev_left, self._prev_right = left_position, right_position
            return

        # 바퀴 이동 거리 d = R * Δφ, 로봇 이동 거리 Δs = (d_r + d_l) / 2
        # 시뮬레이터는 바퀴 회전을 step마다 2·atan(Δφ/2)로 적분하므로 Encoder가 빠를수록 적게 나온다
        # (최고 속도에서 1.24%, 실측). 역함수 2·tan(Δφ/2)로 실제 회전각을 복원한다.
        d_left = WHEEL_RADIUS * 2.0 * math.tan((left_position - self._prev_left) / 2.0)
        d_right = WHEEL_RADIUS * 2.0 * math.tan((right_position - self._prev_right) / 2.0)
        self._prev_left, self._prev_right = left_position, right_position
        delta_s = (d_right + d_left) / 2.0

        if compass_yaw is not None:
            new_theta = wrap_angle(compass_yaw + self._heading_offset)
        else:
            new_theta = wrap_angle(self.theta + (d_right - d_left) / WHEEL_SEPARATION)

        # 이동 구간의 중간 방향으로 위치를 적분한다.
        mid_theta = self.theta + wrap_angle(new_theta - self.theta) / 2.0
        self.x += delta_s * math.cos(mid_theta)
        self.y += delta_s * math.sin(mid_theta)
        self.theta = new_theta
        self.distance += abs(delta_s)
        self.forward += delta_s


# =====================================================================================
# 2. Mapping: log-odds Occupancy Grid + 카메라 시야 Coverage
# =====================================================================================
class OccupancyGrid:
    """log_odds[iy, ix] 하나가 셀 하나이며 0이면 미관측(P = 0.5)이다."""

    def __init__(self, center_x, center_y):
        self.size = int(round(MAP_SIZE / RES))
        self.origin_x = center_x - MAP_SIZE / 2.0   # 셀 (0, 0)의 왼쪽 아래 모서리 월드 좌표.
        self.origin_y = center_y - MAP_SIZE / 2.0
        shape = (self.size, self.size)
        self.log_odds = np.zeros(shape, dtype=np.float32)
        self.hit_count = np.zeros(shape, dtype=np.uint16)   # 장애물로 관측된 횟수 (다시 비면 0으로 돌아간다).
        self.hit_total = np.zeros(shape, dtype=np.uint16)   # 장애물로 관측된 누적 횟수.
        self.free_total = np.zeros(shape, dtype=np.uint16)  # 빈 공간으로 관측된 누적 횟수.
        self.seen = np.zeros(shape, dtype=bool)             # 카메라로 확인한 셀.
        self.ignored_u8 = np.zeros(shape, dtype=np.uint8)   # 관측할 수 없어 탐색 대상에서 뺀 셀.
        self.ignored = self.ignored_u8.view(bool)           # 같은 메모리를 bool로 본 것.
        self.virtual = np.zeros(shape, dtype=np.uint8)      # LiDAR에 보이지 않지만 막혀 있던 곳 (유리 등).
        self.rough = np.zeros(shape, dtype=np.uint8)        # 카메라로 본 어두운 바닥(카펫)의 관측 횟수.
        self.visited = np.zeros(shape, dtype=np.uint8)      # 로봇이 이미 지나간 자리.
        self.bbox = None                                    # 관측된 영역 [iy0, iy1, ix0, ix1).
        self._samples = np.arange(0.0, LIDAR_MAX_RANGE, RES * 0.5)

    def world_to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin_x) / RES).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin_y) / RES).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        return (self.origin_x + (np.asarray(ix) + 0.5) * RES,
                self.origin_y + (np.asarray(iy) + 0.5) * RES)

    def _flat(self, x, y):
        """월드 좌표 배열을 지도 범위 안의 중복 없는 1차원 셀 index로 변환한다."""
        ix, iy = self.world_to_cell(x, y)
        inside = (ix >= 0) & (ix < self.size) & (iy >= 0) & (iy < self.size)
        return np.unique(iy[inside] * self.size + ix[inside])

    def _ray_cells(self, sx, sy, cos_a, sin_a, lengths):
        """각 ray를 센서에서 lengths까지 반 셀 간격으로 샘플링한 셀 index를 반환한다."""
        mask = self._samples[None, :] < lengths[:, None]
        x = sx + cos_a[:, None] * self._samples[None, :]
        y = sy + sin_a[:, None] * self._samples[None, :]
        return self._flat(x[mask], y[mask])

    def _grow_bbox(self, sx, sy):
        ix, iy = self.world_to_cell(sx, sy)
        reach = int(LIDAR_MAX_RANGE / RES) + 3
        box = [max(int(iy) - reach, 0), min(int(iy) + reach, self.size),
               max(int(ix) - reach, 0), min(int(ix) + reach, self.size)]
        if self.bbox is None:
            self.bbox = box
        else:
            self.bbox = [min(self.bbox[0], box[0]), max(self.bbox[1], box[1]),
                         min(self.bbox[2], box[2]), max(self.bbox[3], box[3])]

    def mark_free_disc(self, x, y, radius):
        """로봇이 서 있는 자리는 빈 공간이므로 시작 시 표시한다."""
        ix, iy = self.world_to_cell(x, y)
        r = int(math.ceil(radius / RES))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disc = (xx * xx + yy * yy) * RES * RES <= radius * radius
        region = self.log_odds[iy - r:iy + r + 1, ix - r:ix + r + 1]
        region[disc] = np.minimum(region[disc], -2.0)
        self._grow_bbox(x, y)

    def update(self, sx, sy, sth, ranges, ray_angles):
        """LiDAR scan 한 번으로 지도를 갱신한다."""
        hit = np.isfinite(ranges) & (ranges >= LIDAR_MIN_RANGE) & (ranges <= LIDAR_MAX_RANGE)
        valid = hit | np.isposinf(ranges)
        if not valid.any():
            return
        self._grow_bbox(sx, sy)

        angles = sth + ray_angles
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        hit_x = sx + ranges[hit] * cos_a[hit]
        hit_y = sy + ranges[hit] * sin_a[hit]
        occ_flat = self._flat(hit_x, hit_y)

        # 빈 공간은 끝점 한 셀 앞까지만 갱신한다. 여유가 없으면 거리 노이즈 때문에 벽 셀이 깎인다.
        # 최대 거리 근처의 벽은 hit과 no-return이 섞여 나오므로 no-return ray에도 같은 여유를 둔다.
        free_length = np.where(hit, ranges, LIDAR_MAX_RANGE) - RES
        free_flat = self._ray_cells(sx, sy, cos_a[valid], sin_a[valid], free_length[valid])
        # 같은 scan에서 장애물로 관측된 셀은 빈 공간으로 갱신하지 않는다.
        free_flat = np.setdiff1d(free_flat, occ_flat, assume_unique=True)

        log_odds = self.log_odds.reshape(-1)
        log_odds[free_flat] = np.clip(log_odds[free_flat] + L_FREE, L_MIN, L_MAX)
        log_odds[occ_flat] = np.clip(log_odds[occ_flat] + L_OCC, L_MIN, L_MAX)

        hit_total = self.hit_total.reshape(-1)
        free_total = self.free_total.reshape(-1)
        hit_total[occ_flat] = np.minimum(hit_total[occ_flat], 60000) + 1
        free_total[free_flat] = np.minimum(free_total[free_flat], 60000) + 1
        hit_count = self.hit_count.reshape(-1)
        hit_count[occ_flat] = np.minimum(hit_count[occ_flat], 60000) + 1
        # 다시 확실한 빈 공간이 된 셀은 과거의 장애물 관측 횟수를 지운다 (사람이 지나간 자리).
        hit_count[free_flat[log_odds[free_flat] < -2.0]] = 0

    def mark_seen(self, sx, sy, sth, ranges, ray_angles, half_fov):
        """카메라 시야각 안의 ray가 지나간 셀을 "카메라로 확인함"으로 표시한다."""
        in_fov = np.abs(ray_angles) <= half_fov
        r = ranges[in_fov]
        valid = (np.isfinite(r) & (r >= LIDAR_MIN_RANGE)) | np.isposinf(r)
        lengths = np.minimum(np.where(np.isfinite(r), r, CAM_RANGE), CAM_RANGE)
        angles = sth + ray_angles[in_fov]
        flat = self._ray_cells(sx, sy, np.cos(angles[valid]), np.sin(angles[valid]), lengths[valid])
        self.seen.reshape(-1)[flat] = True


# =====================================================================================
# 3. Perception: 사과 탐지 (Camera), 사람 추적 (LiDAR)
# =====================================================================================
class AppleDetector:
    """HSV 색 분할 -> Contour -> 원형도와 바닥 기하 조건으로 바닥 위의 사과만 남긴다.

    바닥 위 사과는 거리와 무관하게 영상에서 (중심 세로 위치 - 영상 중심) / 반지름이
    (카메라 높이 - 사과 중심 높이) / 사과 반지름 = 0.43으로 일정하다.
    식탁 위 과일, 벽의 붉은 물체, 반사된 상은 이 조건에서 걸러진다.
    거리는 영상에서의 반지름으로 계산한다: Z = f * R / r.
    """

    def __init__(self, width, height, fov):
        self.width, self.height = width, height
        self.focal = (width / 2.0) / math.tan(fov / 2.0)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.floor_ratio = (CAM_HEIGHT - APPLE_CENTER_Z) / APPLE_RADIUS

    def detect(self, frame_bgr):
        """[{color, u, v, r, forward, lateral}, ...]를 반환한다 (forward/lateral은 카메라 기준 [m])."""
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
        detections = []
        for color, ranges in COLOR_RANGES.items():
            mask = None
            for lower, upper in ranges:
                part = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
                mask = part if mask is None else cv2.bitwise_or(mask, part)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 40:
                    continue
                (u, v), r = cv2.minEnclosingCircle(contour)
                x, y, w, h = cv2.boundingRect(contour)
                # 영상 가장자리에 걸린 물체는 크기를 믿을 수 없다.
                if x <= 1 or y <= 1 or x + w >= self.width - 1 or y + h >= self.height - 1:
                    continue
                if area / (math.pi * r * r) < 0.62 or not 0.7 <= w / h <= 1.35:
                    continue
                # 바닥 위의 물체인지 확인한다 (로봇의 앞뒤 흔들림을 고려해 여유를 둔다).
                expected_v = self.height / 2.0 + self.floor_ratio * r
                if abs(v - expected_v) > 0.6 * r + 8.0:
                    continue
                forward = self.focal * APPLE_RADIUS / r
                if forward > DETECT_MAX_RANGE:
                    continue
                lateral = forward * (self.width / 2.0 - u) / self.focal
                detections.append({"color": color, "u": u, "v": v, "r": r,
                                   "forward": forward, "lateral": lateral})
        return detections


class FloorScanner:
    """카메라 영상의 아래쪽(바닥)을 바닥 평면에 투영해 어두운 바닥(카펫)을 찾는다.

    카펫은 5 mm 높이의 턱이 있어 TurtleBot3의 뒤쪽 볼 캐스터(반지름 4 mm)가 걸린다 (실측: 10도까지 기울고 헛바퀴).
    LiDAR에는 보이지 않으므로 카메라로 찾는다. 수평선보다 (v - cy) 픽셀 아래의 바닥까지 거리는 d = f h / (v - cy)다.
    """

    def __init__(self, width, height, fov, ray_count):
        focal = (width / 2.0) / math.tan(fov / 2.0)
        self.rows = np.arange(height // 2 + 30, height, 4)      # 1.35 m 안쪽의 바닥만 쓴다 (먼 곳은 부정확하다).
        self.cols = np.arange(0, width, 4)
        v, u = np.meshgrid(self.rows, self.cols, indexing="ij")
        self.forward = focal * CAM_HEIGHT / (v - height / 2.0) + CAM_OFFSET_X    # 로봇 기준 앞쪽 거리 [m].
        self.lateral = (self.forward - CAM_OFFSET_X) * (width / 2.0 - u) / focal  # 왼쪽이 +.
        self.distance = np.hypot(self.forward, self.lateral)
        bearing = np.arctan2(self.lateral, self.forward)
        self.ray_index = np.round((math.pi - bearing) / (2.0 * math.pi / ray_count) - 0.5).astype(np.int64) % ray_count

    def dark_floor(self, frame_bgr, ranges):
        """어두운 바닥 점들의 로봇 기준 좌표 (앞, 왼쪽)를 반환한다.

        LiDAR가 그 방향으로 더 멀리 보고 있을 때만 인정한다. 그렇지 않으면 바닥이 아니라 가구 같은 물체의 옆면이다.
        """
        value = frame_bgr[self.rows[:, None], self.cols[None, :]].max(axis=2)       # HSV의 V = max(B, G, R)
        dark = (value < DARK_FLOOR_VALUE) & (ranges[self.ray_index] > self.distance + 0.10)
        return self.forward[dark], self.lateral[dark]


class AppleTrack:
    """여러 프레임의 탐지를 하나의 사과로 묶는다. 가까이에서 본 관측일수록 크게 반영한다."""

    def __init__(self, color, x, y, distance):
        self.color = color
        self.x, self.y = x, y
        self.weight = 1.0 / (distance * distance)
        self.count = 1
        self.visited = False
        self.rejected = False
        self.attempts = 0
        self.visit_time = None

    def add(self, x, y, distance):
        w = 1.0 / (distance * distance)
        total = self.weight + w
        self.x = (self.x * self.weight + x * w) / total
        self.y = (self.y * self.weight + y * w) / total
        self.weight = total
        self.count += 1

    @property
    def confirmed(self):
        return self.count >= 3 and not self.rejected


class PersonTracker:
    """LiDAR의 동적 점(조금 전까지 레이저가 지나가던 자리에 나타난 점)으로 사람의 위치와 속도를 추정한다."""

    def __init__(self):
        self.pos = None
        self.vel = np.zeros(2)
        self.last_seen = -1e9
        self.history = deque()

    def update(self, t, points, robot_xy):
        if len(points) >= 2:
            # 추적 중이면 그 근처의 점을, 아니면 로봇에서 가장 가까운 점을 기준으로 묶는다.
            anchor = self.pos if (self.pos is not None and t - self.last_seen < 1.0) else robot_xy
            nearest = points[np.argmin(np.hypot(points[:, 0] - anchor[0], points[:, 1] - anchor[1]))]
            cluster = points[np.hypot(points[:, 0] - nearest[0], points[:, 1] - nearest[1]) < 0.6]
            if len(cluster) >= 2:
                center = cluster.mean(axis=0)
                if self.pos is None or t - self.last_seen > 1.0 or np.hypot(*(center - self.pos)) > 0.6:
                    self.history.clear()
                    self.vel = np.zeros(2)
                    self.pos = center
                else:
                    self.pos = 0.5 * self.pos + 0.5 * center
                self.history.append((t, self.pos.copy()))
                while t - self.history[0][0] > 1.5:
                    self.history.popleft()
                t0, p0 = self.history[0]
                if t - t0 >= 0.6:
                    velocity = (self.pos - p0) / (t - t0)
                    speed = np.hypot(*velocity)
                    if speed > 0.6:
                        velocity *= 0.6 / speed
                    self.vel = 0.7 * self.vel + 0.3 * velocity
                self.last_seen = t

    def active(self, t):
        return self.pos is not None and t - self.last_seen < 1.0 and len(self.history) >= 3

    def moving(self):
        """추적 대상이 실제로 이동하고 있는지 (제자리에서 깜빡이는 점은 사람이 아니다)."""
        if len(self.history) < 3:
            return False
        return float(np.hypot(*(self.history[-1][1] - self.history[0][1]))) > 0.12 or float(np.hypot(*self.vel)) > 0.1

    def predicted(self, horizon=(0.0, 1.0, 2.0, 3.0)):
        """현재와 몇 초 뒤의 예상 위치. 속도가 불확실하면 현재 위치만 쓴다."""
        if np.hypot(*self.vel) < 0.08:
            return [self.pos]
        return [self.pos + self.vel * dt for dt in horizon]

    def closest_approach(self, x, y):
        """앞으로 몇 초 동안 사람이 (x, y)에 가장 가까워지는 거리."""
        return min(math.hypot(p[0] - x, p[1] - y) for p in self.predicted(PERSON_HORIZON))


# =====================================================================================
# 4. Planning: Inflation Costmap, 도달 가능 영역, A*, 시야 기반 Frontier 탐색
# =====================================================================================
NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]
KERNEL3 = np.ones((3, 3), np.uint8)
KERNEL_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


class Planner:
    """관측된 영역(grid.bbox)만 잘라서 Costmap을 만들고 그 위에서 계획한다."""

    def __init__(self, grid):
        self.grid = grid
        self.ready = False
        self.thorough = False   # 정밀 탐색 여부.
        self.goal_cluster = np.empty((0, 2))

    # ----- 좌표 변환 (crop 기준 셀 (cy, cx)) -----
    def to_cell(self, x, y):
        ix, iy = self.grid.world_to_cell(x, y)
        return int(iy) - self.oy, int(ix) - self.ox

    def to_world(self, cy, cx):
        x, y = self.grid.cell_to_world(np.asarray(cx) + self.ox, np.asarray(cy) + self.oy)
        return x, y

    def inside(self, cy, cx):
        return 0 <= cy < self.shape[0] and 0 <= cx < self.shape[1]

    # ----- Costmap -----
    def rebuild(self, objects, person_points, relaxed=False):
        """objects: [(x, y, 반지름)] 사과 등 LiDAR에 보이지 않는 물체. person_points: 사람 예상 위치.

        relaxed: 복귀 경로를 찾지 못할 때의 마지막 수단. LiDAR로 본 장애물만 남기고 여유를 1 cm로 줄인다.
        """
        iy0, iy1, ix0, ix1 = self.grid.bbox
        self.oy, self.ox = iy0, ix0
        self.crop = (slice(iy0, iy1), slice(ix0, ix1))
        log_odds = self.grid.log_odds[self.crop]
        self.shape = log_odds.shape

        occupied = log_odds > L_OCC_THRESHOLD
        self.occupied = occupied
        self.free = log_odds < L_FREE_THRESHOLD
        self.unknown = ~occupied & ~self.free
        self.trail = self.grid.visited[self.crop] > 0
        self.free_float = (log_odds < -2.0).astype(np.float32)   # 여러 번 빈 공간으로 확인된 셀.

        obstacle = occupied.astype(np.uint8)
        if not relaxed:
            obstacle |= self.grid.virtual[self.crop] | (self.grid.rough[self.crop] >= ROUGH_CONFIRM).astype(np.uint8)
        for x, y, radius in ([] if relaxed else objects):
            cy, cx = self.to_cell(x, y)
            cv2.circle(obstacle, (cx, cy), max(1, int(round(radius / RES))), 1, -1)

        hits = self.grid.hit_count[self.crop]
        # 구조물 영역: 여러 번, 관측의 상당 부분에서 장애물이었던 셀과 그 주변 0.1 m.
        # 계단처럼 LiDAR 높이에 걸쳐 값이 깜빡이는 곳의 점을 사람으로 착각하지 않기 위해 쓴다.
        # 사람이 자주 지나가는 길은 대부분의 관측에서 비어 있으므로 여기에 들어가지 않는다.
        total_hits = self.grid.hit_total[self.crop].astype(np.float32)
        total_free = self.grid.free_total[self.crop].astype(np.float32)
        structure = (total_hits >= 30) & (total_hits > 0.15 * (total_hits + total_free))
        self.structure = cv2.dilate(structure.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

        # Scan Matching 기준: 안정적으로 관측된 장애물까지의 거리 [m] (MATCH_CAP에서 자른다).
        stable = (occupied & (hits >= MATCH_STATIC_HITS)).astype(np.uint8)
        stable_distance = cv2.distanceTransform(1 - stable, cv2.DIST_L2, 5) * RES
        self.match_field = np.minimum(stable_distance, MATCH_CAP)
        self.wide_field = np.minimum(stable_distance, 0.5)      # 위치를 크게 잃었을 때의 재탐색용.

        if len(person_points) > 0:
            for index, point in enumerate(person_points):
                cy, cx = self.to_cell(point[0], point[1])
                radius = PERSON_RADIUS if index == 0 else PERSON_RADIUS - 0.1
                cv2.circle(obstacle, (cx, cy), int(round(radius / RES)), 1, -1)

        # 가장 가까운 장애물까지의 거리 [m].
        self.dist = cv2.distanceTransform(1 - obstacle, cv2.DIST_L2, 5) * RES
        self.lethal = ROBOT_RADIUS + 0.01 if relaxed else R_LETHAL
        self.trav = self.free & (self.dist > self.lethal)
        proximity = np.clip((R_SOFT - self.dist) / (R_SOFT - self.lethal), 0.0, 1.0)
        self.cost = 1.0 + PROXIMITY_WEIGHT * proximity * proximity
        self._lists = None
        self.ready = True

    def nearest_traversable(self, cy, cx, max_radius=0.5):
        """(cy, cx)에서 가장 가까운 주행 가능 셀. 로봇이 장애물 여유 영역 안에 있을 때 쓴다."""
        if self.inside(cy, cx) and self.trav[cy, cx]:
            return cy, cx
        r = int(max_radius / RES)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, self.shape[0])
        x0, x1 = max(cx - r, 0), min(cx + r + 1, self.shape[1])
        if y0 >= y1 or x0 >= x1:
            return None
        ys, xs = np.nonzero(self.trav[y0:y1, x0:x1])
        if len(ys) == 0:
            return None
        best = np.argmin((ys + y0 - cy) ** 2 + (xs + x0 - cx) ** 2)
        return int(ys[best] + y0), int(xs[best] + x0)

    def reach(self, x, y, start_radius=0.5, trail_cost=1):
        """로봇 위치에서 도달 가능한 셀까지의 거리(셀 단위 wavefront)를 self.D에 계산한다.

        trail_cost > 1이면 이미 지나간 자리의 셀은 들어가는 데 그만큼 더 걸리는 것으로 계산한다.
        탐색 목표를 고를 때 쓰며, 되돌아가야 닿는 곳보다 새 길로 닿는 곳이 가깝게 나온다.
        """
        self.D = np.full(self.shape, -1, dtype=np.int32)
        self.start = self.nearest_traversable(*self.to_cell(x, y), max_radius=start_radius)
        if self.start is None:
            return False
        trav = self.trav.astype(np.uint8)
        reached = np.zeros(self.shape, dtype=np.uint8)
        reached[self.start] = 1
        self.D[self.start] = 0
        if trail_cost <= 1:
            wave = reached.copy()
            step = 0
            while True:
                step += 1
                # 상하좌우로만 확장한다 (A*는 양옆이 막힌 대각선 이동을 허용하지 않는다).
                wave = cv2.dilate(wave, KERNEL_CROSS) & trav & (1 - reached)
                if not wave.any():
                    break
                self.D[wave > 0] = step
                reached |= wave
            return True
        # 셀마다 들어가는 비용이 다른 wavefront: 시각 k에 도착한 셀들에서 이웃으로 퍼지고,
        # 이웃은 k + (그 셀의 비용) 시각에 도착하는 것으로 예약한다 (시각 순서로 처리하므로 처음 예약된 값이 최소다).
        trail = self.trail.astype(np.uint8)
        pending = {0: reached.copy()}
        step = 0
        while pending:
            wave = pending.pop(step, None)
            if wave is not None:
                new = cv2.dilate(wave, KERNEL_CROSS) & trav & (1 - reached)
                reached |= new
                for cells, arrival in ((new & (1 - trail), step + 1), (new & trail, step + trail_cost)):
                    if cells.any():
                        self.D[cells > 0] = arrival
                        pending[arrival] = pending[arrival] | cells if arrival in pending else cells
            step += 1
        return True

    # ----- A* -----
    def astar(self, start, goal, avoid_trail=False):
        """Costmap 위의 8방향 A*. 비용 = 이동 거리 x (1 + 장애물 근접 비용), 휴리스틱 = octile 거리.

        avoid_trail이면 이미 지나간 자리의 비용을 TRAIL_PATH_COST배로 해서 새 길을 먼저 찾는다 (탐색 중에만 쓴다).
        """
        height, width = self.shape
        if self._lists is None:
            self._lists = {}
        if avoid_trail not in self._lists:
            cost = self.cost * np.where(self.trail, TRAIL_PATH_COST, 1.0) if avoid_trail else self.cost
            self._lists[avoid_trail] = (self.trav.reshape(-1).tolist(), cost.reshape(-1).tolist())
        trav, cost = self._lists[avoid_trail]
        start_index = start[0] * width + start[1]
        goal_index = goal[0] * width + goal[1]
        goal_y, goal_x = goal
        g_score = {start_index: 0.0}
        came_from = {}
        closed = bytearray(height * width)
        open_heap = [(0.0, 0.0, start_index)]
        diagonal_extra = math.sqrt(2.0) - 1.0

        while open_heap:
            _, g_current, current = heapq.heappop(open_heap)
            if closed[current]:
                continue
            closed[current] = 1
            if current == goal_index:
                path = [goal]
                while current in came_from:
                    current = came_from[current]
                    path.append(divmod(current, width))
                return path[::-1]
            cy, cx = divmod(current, width)
            for dy, dx, step in NEIGHBORS:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                neighbor = ny * width + nx
                if closed[neighbor] or not trav[neighbor]:
                    continue
                # 대각선 이동은 양옆 셀이 모두 비어 있을 때만 허용한다.
                if dy and dx and not (trav[cy * width + nx] and trav[ny * width + cx]):
                    continue
                g_new = g_current + step * cost[neighbor]
                if g_new < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = g_new
                    came_from[neighbor] = current
                    ay, ax = abs(ny - goal_y), abs(nx - goal_x)
                    heuristic = max(ay, ax) + diagonal_extra * min(ay, ax)
                    heapq.heappush(open_heap, (g_new + heuristic, g_new, neighbor))
        return None

    def _line_clear(self, a, b, min_dist):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        ys = np.round(np.linspace(a[0], b[0], n)).astype(int)
        xs = np.round(np.linspace(a[1], b[1], n)).astype(int)
        if not (np.all(self.trav[ys, xs]) and np.all(self.dist[ys, xs] >= min_dist)):
            return False
        # 대각선으로 넘어가는 곳은 양옆 셀도 비어 있어야 한다 (A*의 규칙과 같다).
        diagonal = (np.diff(ys) != 0) & (np.diff(xs) != 0)
        return bool(np.all(self.trav[ys[:-1][diagonal], xs[1:][diagonal]])
                    and np.all(self.trav[ys[1:][diagonal], xs[:-1][diagonal]]))

    def smooth(self, cells):
        """장애물 여유를 줄이지 않는 범위에서 경로의 꺾임을 직선으로 줄인다."""
        clearance = np.array([self.dist[c] for c in cells])
        result = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = min(i + 60, len(cells) - 1)
            while j > i + 1:
                needed = min(0.25, float(clearance[i:j + 1].min()) - 0.01)
                if self._line_clear(cells[i], cells[j], needed):
                    break
                j -= 2 if j - i > 3 else 1
            result.append(cells[j])
            i = j
        return result

    def plan(self, goal, avoid_trail=False):
        """reach()로 계산한 시작 셀에서 goal 셀까지의 월드 좌표 경로 (N, 2)를 반환한다."""
        if self.start is None or not self.inside(*goal) or self.D[goal] < 0:
            return None
        cells = self.astar(self.start, goal, avoid_trail)
        if cells is None:
            return None
        cells = self.smooth(cells) if len(cells) > 2 else cells
        ys, xs = zip(*cells)
        x, y = self.to_world(np.array(ys), np.array(xs))
        return np.column_stack([x, y])

    def blocked(self, path, from_index, distance=2.0):
        """경로의 앞부분이 새로 발견된 장애물에 막혔는지 확인한다."""
        end = min(len(path), from_index + int(distance / RES))
        for x, y in path[from_index:end]:
            cy, cx = self.to_cell(x, y)
            if not self.inside(cy, cx) or self.dist[cy, cx] < self.lethal - 0.03:
                return True
        return False

    # ----- 시야 기반 Frontier 탐색 -----
    @staticmethod
    def _large_components(mask, min_cells):
        """연결된 덩어리 중 min_cells 이상인 것만 남긴다."""
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_cells
        keep[0] = False
        return keep[labels]

    def exploration_targets(self):
        """탐색 대상 셀을 반환한다.

        unseen  : 지도에는 빈 공간이지만 카메라로 아직 보지 못한 곳.
        frontier: LiDAR로도 아직 보지 못한 영역과 맞닿은 빈 공간. 가구 뒤의 가려진 구석처럼 사과가 숨을 수 있는
                  크기(HIDDEN_AREA_CELLS 이상)의 미관측 영역만 센다. 벽 근처의 한두 셀짜리 미관측 점은 무시한다.
        정밀 탐색(self.thorough)에서는 더 작은 영역까지 대상으로 삼는다.
        """
        seen = cv2.dilate(self.grid.seen[self.crop].astype(np.uint8), KERNEL3) > 0
        ignored = self.grid.ignored[self.crop]
        unseen = self._large_components(self.free & ~seen & ~ignored, 6 if self.thorough else MIN_CLUSTER_CELLS)
        hidden = self._large_components(self.unknown, HIDDEN_AREA_CELLS)
        frontier = self.free & (cv2.dilate(hidden.astype(np.uint8), KERNEL3) > 0) & ~ignored
        return unseen, self._large_components(frontier, 3)

    def exploration_goal(self, allow_ignore=True, attract=(), pose=None):
        """도달 가능한 셀 중 점수 = 경로 길이 + 바깥쪽 벌점 + 오른쪽 우선 벌점 - 영역 크기 이득 이 최소인 셀을 고른다.

        경로 길이는 reach()가 계산한 값이며, 이미 지나간 자리를 다시 지나는 길은 더 길게 계산되어 있다.

        1차 탐색: 카메라로 보지 못한 빈 공간(unseen)을 먼저 모두 보고, 그 다음에 미관측 경계(frontier)로 간다.
        2차 정밀 탐색(thorough): 둘을 함께 놓고 가까운 곳부터 간다.
        바깥쪽 벌점: 주변 1.5 m의 절반 가까이가 미관측인 대상(지도의 가장자리, 유리창 너머로만 보이는 곳)은 뒤로 미룬다.
        attract: 이미 발견했지만 아직 접근 지점에 갈 수 없는 사과의 위치들 (유리창 너머로 먼저 본 경우 등).
        탐색 대상이 남아 있는지는 self.targets_left에 기록한다.
        """
        unseen, frontier = self.exploration_targets()
        self.targets_left = bool(unseen.any() or frontier.any())
        unknown_ratio = cv2.blur(self.unknown.astype(np.float32), (31, 31))
        for targets in ((unseen | frontier,) if self.thorough else (unseen, frontier)):
            if not targets.any():
                continue
            # 각 셀에서 가장 가까운 탐색 대상까지의 거리와 그 대상 덩어리의 label.
            dist, nearest = cv2.distanceTransformWithLabels(np.where(targets, 0, 255).astype(np.uint8),
                                                            cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
            candidates = (self.D >= 0) & (dist * RES <= VIEW_DISTANCE)
            if not candidates.any():
                # 도달할 수 없는 영역은 다시 고르지 않도록 제외한다 (사람이 길을 막고 있을 때는 제외하지 않는다).
                if allow_ignore:
                    self.grid.ignored[self.crop] |= targets
                continue
            labels = nearest[targets]
            count = np.bincount(labels, minlength=int(nearest.max()) + 1)
            outer = (np.bincount(labels, weights=unknown_ratio[targets], minlength=len(count))
                     / np.maximum(count, 1)) >= OUTER_RATIO
            ys, xs = np.nonzero(candidates)
            label = nearest[ys, xs]
            score = (self.D[ys, xs] * RES + OUTER_PENALTY * outer[label]
                     - GAIN_WEIGHT * np.minimum(count[label] * RES * RES, 4.0))
            wx, wy = self.to_world(ys, xs)
            if pose is not None:
                # 오른손 법칙: 진행 방향의 오른쪽에 있는 곳부터 고른다 (오른쪽, 정면, 왼쪽, 뒤 순서로 벌점이 커진다).
                bearing = np.arctan2(wy - pose[1], wx - pose[0]) - pose[2]
                score = score + RIGHT_WEIGHT * np.mod(bearing + 0.75 * math.pi, 2.0 * math.pi) / math.pi
            if len(attract):
                # 이미 발견했지만 아직 가는 길을 모르는 사과가 있으면 그 사과에 가까운 곳부터 탐색한다.
                score = score + 2.0 * np.min([np.hypot(wx - ax, wy - ay) for ax, ay in attract], axis=0)
            best = int(np.argmin(score))
            # 이 목표가 향하는 탐색 대상 덩어리 (도착해서 둘러본 뒤 다시 찾아오지 않도록 기억한다).
            cy, cx = np.nonzero(targets & (nearest == label[best]))
            self.goal_cluster = np.column_stack(self.to_world(cy, cx))
            return int(ys[best]), int(xs[best])
        if allow_ignore:
            self.targets_left = False
        return None

    def targets_near(self, x, y, radius):
        """(x, y) 주변 radius 안에 남아 있는 탐색 대상 셀의 월드 좌표 (미확인, 미관측 경계)."""
        unseen, frontier = self.exploration_targets()
        cy, cx = self.to_cell(x, y)
        r = int(radius / RES)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, self.shape[0])
        x0, x1 = max(cx - r, 0), min(cx + r + 1, self.shape[1])
        if y0 >= y1 or x0 >= x1:
            return np.empty((0, 2)), np.empty((0, 2))
        result = []
        for mask in (unseen, frontier):
            ys, xs = np.nonzero(mask[y0:y1, x0:x1])
            wx, wy = self.to_world(ys + y0, xs + x0)
            points = np.column_stack([wx, wy])
            result.append(points[np.hypot(wx - x, wy - y) <= radius] if len(points) else points)
        return result[0], result[1]

    def ignore_points(self, points):
        if len(points):
            ix, iy = self.grid.world_to_cell(points[:, 0], points[:, 1])
            self.grid.ignored[iy, ix] = True

    # ----- 목표 지점 선택 -----
    def standoff(self, x, y):
        """사과 (x, y)에 접근할 지점을 고른다.

        사과에서 APPROACH_RING 거리에 있고 사과까지 시야가 트인 도달 가능한 셀 중 경로가 가장 짧은 셀.
        없으면 도착으로 인정되는 거리(REACHED_DISTANCE) 안에서 사과에 가장 가까운 셀로 간다.
        """
        ys, xs = np.nonzero(self.D >= 0)
        if len(ys) == 0:
            return None
        wx, wy = self.to_world(ys, xs)
        distance = np.hypot(wx - x, wy - y)
        apple = self.to_cell(x, y)
        ring = np.nonzero((distance >= APPROACH_RING[0]) & (distance <= APPROACH_RING[1]))[0]
        for index in ring[np.argsort(self.D[ys[ring], xs[ring]])][:60]:
            if self.line_of_sight((int(ys[index]), int(xs[index])), apple):
                return int(ys[index]), int(xs[index])
        near = np.nonzero((distance > APPROACH_RING[0]) & (distance <= REACHED_DISTANCE - 0.05))[0]
        for index in near[np.argsort(distance[near])][:60]:
            if self.line_of_sight((int(ys[index]), int(xs[index])), apple):
                return int(ys[index]), int(xs[index])
        return None

    def line_of_sight(self, a, b):
        """두 셀 사이에 LiDAR로 관측된 장애물이 없는지 확인한다."""
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        ys = np.clip(np.round(np.linspace(a[0], b[0], n)).astype(int), 0, self.shape[0] - 1)
        xs = np.clip(np.round(np.linspace(a[1], b[1], n)).astype(int), 0, self.shape[1] - 1)
        return not bool(self.occupied[ys, xs].any())

    def escape(self, person_points, max_path=3.5):
        """사람의 예상 경로에서 가장 멀어지는 가까운 셀을 고른다."""
        ys, xs = np.nonzero((self.D >= 0) & (self.D * RES <= max_path))
        if len(ys) == 0:
            return None, 0.0
        wx, wy = self.to_world(ys, xs)
        clearance = np.full(len(ys), np.inf)
        for point in person_points:
            clearance = np.minimum(clearance, np.hypot(wx - point[0], wy - point[1]))
        score = np.minimum(clearance, 1.5) - 0.1 * self.D[ys, xs] * RES
        best = int(np.argmax(score))
        return (int(ys[best]), int(xs[best])), float(clearance[best])


# =====================================================================================
# 5. Control: Look-ahead(Pure Pursuit) 경로 추종
# =====================================================================================
class PathFollower:
    def __init__(self):
        self.path = None

    def set_path(self, path, x, y):
        """경로를 5 cm 간격으로 다시 나누고 현재 위치를 시작점으로 붙인다."""
        points = np.vstack([[x, y], path])
        segment = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
        keep = np.concatenate([[True], segment > 1e-6])
        points = points[keep]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])))])
        if s[-1] < 1e-6:
            self.path = points[-1:].copy()
            self.s = np.zeros(1)
        else:
            samples = np.append(np.arange(0.0, s[-1], RES), s[-1])
            self.path = np.column_stack([np.interp(samples, s, points[:, 0]), np.interp(samples, s, points[:, 1])])
            self.s = samples
        self.index = 0
        self.rotating = False

    def clear(self):
        self.path = None

    def remaining(self, x, y):
        return float(self.s[-1] - self.s[self.index] + np.hypot(*(self.path[self.index] - (x, y))))

    def control(self, x, y, theta):
        """(v, w, arrived)를 반환한다."""
        goal_distance = math.hypot(self.path[-1, 0] - x, self.path[-1, 1] - y)
        if goal_distance < GOAL_TOLERANCE:
            return 0.0, 0.0, True

        # 1. 경로 위에서 로봇과 가장 가까운 점 (뒤로 돌아가지 않도록 앞쪽 구간만 찾는다).
        window = self.path[self.index:self.index + 40]
        self.index += int(np.argmin(np.hypot(window[:, 0] - x, window[:, 1] - y)))

        # 2. Look-ahead Point: 경로를 따라 LOOKAHEAD만큼 앞선 점.
        target = min(int(np.searchsorted(self.s, self.s[self.index] + LOOKAHEAD)), len(self.path) - 1)
        dx, dy = self.path[target, 0] - x, self.path[target, 1] - y

        # 3. 로봇 좌표계로 변환 (x: 앞뒤, y: 좌우).
        x_la = math.cos(theta) * dx + math.sin(theta) * dy
        y_la = -math.sin(theta) * dx + math.cos(theta) * dy
        alpha = math.atan2(y_la, x_la)

        # 방향이 크게 어긋나면 제자리에서 먼저 회전한다.
        if abs(alpha) > (0.35 if self.rotating else 0.9):
            self.rotating = True
            return 0.0, float(np.clip(2.5 * alpha, -W_MAX, W_MAX)), False
        self.rotating = False

        # 4. 곡률 κ = 2 y_LA / (x_LA² + y_LA²), 5. ω = v κ
        curvature = 2.0 * y_la / max(x_la * x_la + y_la * y_la, 1e-6)
        v = V_MAX * (1.0 - 0.6 * min(1.0, abs(alpha) / 0.9))
        v = min(v, max(0.04, 1.2 * goal_distance))
        return v, float(np.clip(v * curvature, -W_MAX, W_MAX)), False


# =====================================================================================
# 6. Mission: 센서 처리, 상태 기계(FSM), 사람 대응, Human-Robot Interface
# =====================================================================================
class Mission:
    def __init__(self, robot, show_gui):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.show_gui = show_gui
        self.target_color = TARGET_COLOR
        self.target_count = TARGET_COUNT
        self.time_limit = TIME_LIMIT

        # ----- 장치 -----
        self.left_motor = robot.getDevice("left wheel motor")
        self.right_motor = robot.getDevice("right wheel motor")
        for motor in (self.left_motor, self.right_motor):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.left_encoder = self.left_motor.getPositionSensor()
        self.right_encoder = self.right_motor.getPositionSensor()
        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)
        self.compass = robot.getDevice("compass")
        self.compass.enable(self.timestep)
        self.accelerometer = robot.getDevice("accelerometer")
        self.accelerometer.enable(self.timestep)
        self.lidar = robot.getDevice("LDS-01")
        self.lidar.enable(self.timestep)
        self.camera = robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # LiDAR 각 측정값의 방향 (정면 0, 반시계 +). index 0이 후방이며 값은 구간의 중심 방향이다.
        rays = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()
        self.ray_angles = fov / 2.0 - (np.arange(rays) + 0.5) * (fov / rays)
        self.ray_cos = np.cos(self.ray_angles)
        self.ray_sin = np.sin(self.ray_angles)

        self.cam_width = self.camera.getWidth()
        self.cam_height = self.camera.getHeight()
        self.cam_half_fov = self.camera.getFov() / 2.0 - math.radians(2.0)

        # ----- 모듈 -----
        self.odom = Odometry(START_X, START_Y, START_THETA)
        self.grid = OccupancyGrid(START_X, START_Y)
        self.grid.mark_free_disc(START_X, START_Y, ROBOT_RADIUS + 0.12)
        self.planner = Planner(self.grid)
        self.detector = AppleDetector(self.cam_width, self.cam_height, self.camera.getFov())
        self.floor = FloorScanner(self.cam_width, self.cam_height, self.camera.getFov(), rays)
        self.person = PersonTracker()
        self.follower = PathFollower()

        # ----- 상태 -----
        self.state = "SPIN"
        self.paused = False
        self.tracks = []
        self.trajectory = [(START_X, START_Y)]
        self.ranges = None
        self.frame = None
        self.detections = []
        self.step_count = 0
        self.time = 0.0
        self.spin_angle = 0.0
        self.spin_prev = None
        self.goal_cell_world = None     # 현재 탐색 목표 (월드 좌표).
        self.goal_since = 0.0           # 지금의 탐색 목표(와 가까운 목표)를 향하기 시작한 시각.
        self.current_track = None       # 접근 중인 사과.
        self.track_goal = None          # 접근 지점을 계산할 때 쓴 사과 위치.
        self.look_start = -1e9
        self.confirm_start = None
        self.confirm_seen = None
        self.recover_until = -1e9
        self.recover_speed = 0.0
        self.person_mode = None         # None / "YIELD" / "EVADE"
        self.evade_time = -1e9
        self.costmap_time = -1e9
        self.replan_time = -1e9
        self.blocked_since = None
        self.no_path_count = 0
        self.return_plan_time = -1e9
        self.home_close_enough = False
        self.relaxed = False            # 복귀 경로가 없어 Costmap의 여유를 줄였는지 여부.
        self.return_reason = ""
        self.finish_time = None
        self.command_buffer = ""
        self.pressed_keys = set()
        self.command_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "command.txt")
        self.messages = deque(maxlen=6)
        self.draw_time = -1e9
        self.status_time = -1e9
        self.command_v = 0.0
        self.turn_watch = None
        self.match_shift = np.zeros(2)  # Scan-to-Map Matching으로 옮긴 누적 보정량.
        self.scans = deque()            # 최근 1.5초의 (시간, scan, LiDAR x, y, 방향, 누적 보정량)
        self.motion = deque()           # 최근 1.5초의 (시간, Encoder 누적 전진 거리, x, y)
        self.tilt = 0.0
        self.tilt_history = deque(maxlen=6)
        self.lost_steps = 0
        self.watch = (0.0, START_X, START_Y, START_THETA)    # 움직임 감시용 (시간, x, y, 방향).
        self.home_best = (float("inf"), 0.0)                 # 복귀 중 시작 지점까지의 최소 거리와 그 시각.
        self.look_travel = 0.0          # 마지막으로 주변을 둘러본 시점의 주행 거리.
        self.look_direction = 1.0
        self.look_turned = 0.0
        self.look_prev = None
        self.look_arrival = False
        self.danger_time = -1e9
        offsets = np.arange(-4, 5) * 0.5                     # [cell] ±0.10 m, 2.5 cm 간격.
        self.match_dx, self.match_dy = (a.reshape(-1) for a in np.meshgrid(offsets, offsets))
        self.match_norm = np.hypot(self.match_dx, self.match_dy) * RES
        self.match_zero = int(np.argmin(self.match_norm))
        offsets = np.arange(-12, 13).astype(float)           # [cell] ±0.60 m, 5 cm 간격.
        self.wide_dx, self.wide_dy = (a.reshape(-1) for a in np.meshgrid(offsets, offsets))
        self.wide_norm = np.hypot(self.wide_dx, self.wide_dy) * RES
        self.wide_zero = int(np.argmin(self.wide_norm))

    # ---------------------------------------------------------------------------------
    # 출력
    # ---------------------------------------------------------------------------------
    def say(self, korean, english=None):
        """콘솔에는 한국어로, 대시보드에는 영어로(OpenCV가 한글을 그리지 못한다) 알린다."""
        print(f"[{self.time:6.1f}s] {korean}", flush=True)
        self.messages.append(f"{self.time:6.1f}s  {english or korean}")

    def drive(self, v, w):
        """선속도 v [m/s]와 각속도 w [rad/s]를 좌우 바퀴 속도 [rad/s]로 변환한다."""
        left = (v - w * TURN_SEPARATION / 2.0) / WHEEL_RADIUS
        right = (v + w * TURN_SEPARATION / 2.0) / WHEEL_RADIUS
        peak = max(abs(left), abs(right))
        if peak > MAX_WHEEL_SPEED:
            left, right = left * MAX_WHEEL_SPEED / peak, right * MAX_WHEEL_SPEED / peak
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)
        self.command_v = v

        # 제자리 회전을 명령했는데 방향이 바뀌지 않으면 바퀴가 헛돌고 있는 것이다 (카펫 턱 등).
        if abs(w) > 0.3 and abs(v) < 0.02:
            if self.turn_watch is None or abs(wrap_angle(self.odom.theta - self.turn_watch[1])) > 0.2:
                self.turn_watch = (self.time, self.odom.theta)
            elif self.time - self.turn_watch[0] > 1.5:
                self.turn_watch = None
                self.say("회전이 되지 않습니다. 조금 이동해서 빠져나옵니다.", "rotation stalled, moving off")
                self.start_recovery(0.08 if self.front_clearance() > ROBOT_RADIUS + 0.25 else -0.06, 1.0)
        else:
            self.turn_watch = None

    def scan_points(self):
        """LiDAR 점들의 로봇 좌표계 위치 (x: 앞, y: 왼쪽)."""
        finite = np.isfinite(self.ranges)
        return (LIDAR_OFFSET_X + self.ranges[finite] * self.ray_cos[finite],
                self.ranges[finite] * self.ray_sin[finite])

    def front_clearance(self, half_width=ROBOT_RADIUS + 0.02):
        """진행 방향 통로 안에서 가장 가까운 LiDAR 점까지의 거리 (로봇 중심 기준)."""
        px, py = self.scan_points()
        ahead = (px > 0.0) & (np.abs(py) < half_width)
        return float(px[ahead].min()) if ahead.any() else LIDAR_MAX_RANGE

    def rear_clearance(self):
        px, py = self.scan_points()
        behind = (px < 0.0) & (np.abs(py) < ROBOT_RADIUS + 0.03)
        return float(-px[behind].max()) if behind.any() else LIDAR_MAX_RANGE

    def start_recovery(self, speed, duration):
        self.follower.clear()
        self.recover_speed = speed
        self.recover_until = self.time + duration

    # ---------------------------------------------------------------------------------
    # 센서 처리: Localization -> Mapping -> Perception
    # ---------------------------------------------------------------------------------
    def sense(self):
        self.time = self.robot.getTime()
        self.step_count += 1
        self.odom.update(self.left_encoder.getValue(), self.right_encoder.getValue(), self.compass.getValues())
        if not self.odom.ready:
            return False
        ranges = self.lidar.getRangeImage()
        if not ranges or len(ranges) != len(self.ray_angles):
            return False
        ranges = np.asarray(ranges, dtype=np.float64)
        if np.isnan(ranges).any():
            return False
        self.ranges = ranges
        theta = self.odom.theta

        # 로봇이 기울어지면 (낮은 물건이나 턱에 올라탄 경우) LiDAR 평면이 바닥을 비춰 헛것이 보인다.
        # 가속도계의 수평 성분으로 기울기를 판단하고, 기울어진 동안의 scan은 지도와 위치 보정에 쓰지 않는다.
        # 기울어지면 중력이 수평 성분으로 계속 같은 방향에 나타난다. 출발과 정지의 가속은 방향이 바뀌며
        # 서로 상쇄되므로, 최근 6개 가속도 벡터의 평균 크기로 판단한다 (0.6 m/s² = 약 3.5도).
        acceleration = self.accelerometer.getValues()
        if not math.isnan(acceleration[0]):
            self.tilt_history.append((acceleration[0], acceleration[1]))
        self.tilt = float(np.hypot(*np.mean(self.tilt_history, axis=0))) if self.tilt_history else 0.0
        level = self.tilt < 0.6

        # ----- Localization 보정: Scan-to-Map Matching (누적 오차, 바퀴 미끄러짐) -----
        # scan이 지도와 모순되면 (미끄러져 위치를 잃은 경우) 넓은 범위에서 위치를 다시 찾고,
        # 찾을 때까지는 지도를 갱신하지 않는다. 틀린 위치로 지도를 그리면 틀린 위치가 굳어지기 때문이다.
        # 5초가 지나도 못 찾으면 환경이 바뀐 것으로 보고 갱신을 다시 시작한다.
        consistent = level and self.match_scan(ranges)
        if level and not consistent:
            consistent = self.relocalize(ranges)
        self.lost_steps = 0 if consistent else self.lost_steps + 1
        update_map = level and (consistent or self.lost_steps > 75)
        x, y = self.odom.x, self.odom.y
        self.motion.append((self.time, self.odom.forward, x, y))
        while self.time - self.motion[0][0] > 1.5:
            self.motion.popleft()
        if math.hypot(x - self.trajectory[-1][0], y - self.trajectory[-1][1]) >= RES:
            self.trajectory.append((x, y))
            cell = tuple(int(v) for v in self.grid.world_to_cell(x, y))
            cv2.circle(self.grid.visited, cell, int(TRAIL_RADIUS / RES), 1, -1)
        lidar_x = x + LIDAR_OFFSET_X * math.cos(theta)
        lidar_y = y + LIDAR_OFFSET_X * math.sin(theta)

        # ----- 사람 탐지 (Scan-to-Scan), Mapping -----
        if level:
            self.person.update(self.time, self.find_moving_points(ranges, lidar_x, lidar_y, theta), (x, y))
        else:
            self.scans.clear()
        if update_map:
            self.grid.update(lidar_x, lidar_y, theta, ranges, self.ray_angles)

        # ----- 사과 탐지 (2 step마다) -----
        if self.step_count % 2 == 0:
            image = self.camera.getImage()
            if image:
                frame = np.frombuffer(image, np.uint8).reshape((self.cam_height, self.cam_width, 4))
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.detections = self.detector.detect(self.frame)
                self.register_detections(x, y, theta)
                if update_map:
                    self.grid.mark_seen(lidar_x, lidar_y, theta, ranges, self.ray_angles, self.cam_half_fov)
                    # 어두운 바닥(카펫)을 지도에 기록한다.
                    forward, lateral = self.floor.dark_floor(self.frame, ranges)
                    if len(forward):
                        ix, iy = self.grid.world_to_cell(x + forward * math.cos(theta) - lateral * math.sin(theta),
                                                         y + forward * math.sin(theta) + lateral * math.cos(theta))
                        inside = (ix >= 0) & (ix < self.grid.size) & (iy >= 0) & (iy < self.grid.size)
                        flat = np.unique(iy[inside] * self.grid.size + ix[inside])
                        rough = self.grid.rough.reshape(-1)
                        rough[flat] = np.minimum(rough[flat], 200) + 1
        return True

    def find_moving_points(self, ranges, lidar_x, lidar_y, theta):
        """약 1.4초 전의 scan과 현재 scan을 비교해 움직이는 물체(사람)의 점을 찾는다 (Scan-to-Scan).

        현재 scan의 각 점을 이전 scan의 좌표계로 옮겨, 이전 scan이 그 방향으로 얼마나 멀리 보았는지와 비교한다.
        이전에는 주변 5개 ray가 모두 그 자리를 지나 더 멀리 갔는데 지금 점이 있으면 움직이는 물체다.
        짧은 시간의 Odometry만 쓰므로 전역 위치 오차의 영향을 받지 않는다.
        """
        self.scans.append((self.time, ranges, lidar_x, lidar_y, theta, self.match_shift.copy()))
        while self.time - self.scans[0][0] > 1.5:
            self.scans.popleft()
        t0, old, old_x, old_y, old_theta, shift0 = self.scans[0]
        empty = np.empty((0, 2))
        if self.time - t0 < 1.2 or not self.planner.ready:
            return empty
        # 그 사이 Scan-to-Map Matching이 옮긴 만큼 이전 위치도 옮겨 같은 좌표계에서 비교한다.
        old_x += self.match_shift[0] - shift0[0]
        old_y += self.match_shift[1] - shift0[1]

        index = np.nonzero(np.isfinite(ranges) & (ranges >= 0.15) & (ranges <= LIDAR_MAX_RANGE - 0.3))[0]
        angles = theta + self.ray_angles[index]
        px = lidar_x + ranges[index] * np.cos(angles)
        py = lidar_y + ranges[index] * np.sin(angles)
        dx, dy = px - old_x, py - old_y
        rho = np.hypot(dx, dy)
        beta = np.arctan2(dy, dx) - old_theta
        count = len(ranges)
        old_index = np.round((math.pi - np.arctan2(np.sin(beta), np.cos(beta))) / (2.0 * math.pi / count) - 0.5).astype(np.int64) % count
        old_far = np.where(np.isfinite(old), old, LIDAR_MAX_RANGE + 1.0)
        nearest_old = np.min([old_far[(old_index + k) % count] for k in (-2, -1, 0, 1, 2)], axis=0)
        moved = (nearest_old > rho + 0.18) & (rho > 0.25) & (rho < LIDAR_MAX_RANGE - 0.3)

        # 구조물(벽, 가구, 계단) 위의 점은 제외한다.
        planner = self.planner
        ix, iy = self.grid.world_to_cell(px, py)
        cy, cx = iy - planner.oy, ix - planner.ox
        inside = (cy >= 0) & (cy < planner.shape[0]) & (cx >= 0) & (cx < planner.shape[1])
        on_structure = np.zeros(len(px), dtype=bool)
        on_structure[inside] = planner.structure[cy[inside], cx[inside]]
        moved &= ~on_structure

        moved_index = index[moved]
        if len(moved_index) < 2:
            return empty
        points = np.column_stack([px[moved], py[moved]])
        # 이웃한 ray끼리 묶고, 점이 너무 적거나 너무 넓은 묶음은 버린다.
        keep = []
        for group in np.split(np.arange(len(moved_index)), np.nonzero(np.diff(moved_index) > 3)[0] + 1):
            extent = np.hypot(*(points[group[-1]] - points[group[0]]))
            if len(group) >= (3 if ranges[moved_index[group[0]]] < 2.2 else 2) and extent < 0.8:
                keep.extend(group)
        return points[keep] if keep else empty

    def scan_cells(self, ranges):
        """scan 끝점의 Costmap 기준 셀 좌표 (실수)를 반환한다."""
        hit = np.isfinite(ranges) & (ranges >= 0.15) & (ranges <= 3.3)
        theta = self.odom.theta
        lidar_x = self.odom.x + LIDAR_OFFSET_X * math.cos(theta)
        lidar_y = self.odom.y + LIDAR_OFFSET_X * math.sin(theta)
        angles = theta + self.ray_angles[hit]
        fx = (lidar_x + ranges[hit] * np.cos(angles) - self.grid.origin_x) / RES - self.planner.ox
        fy = (lidar_y + ranges[hit] * np.sin(angles) - self.grid.origin_y) / RES - self.planner.oy
        return fx, fy

    def field_lookup(self, field, fx, fy, cap):
        """셀 좌표 (실수 배열)에서 거리장 값을 읽는다. 범위 밖은 cap으로 둔다."""
        ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        inside = (ix >= 0) & (ix < field.shape[1]) & (iy >= 0) & (iy < field.shape[0])
        values = np.full(ix.shape, cap, dtype=np.float32)
        values[inside] = field[iy[inside], ix[inside]]
        return values

    def match_scan(self, ranges):
        """현재 scan이 지도와 가장 잘 겹치는 위치 보정량 (dx, dy)을 찾아 Odometry에 반영한다 (Scan-to-Map).

        방향은 Compass로 정확히 알기 때문에 위치만 탐색한다 (±0.10 m, 2.5 cm 간격의 81개 후보).
        점수 = scan 끝점에서 가장 가까운 정적 장애물까지의 평균 거리.
        이미 지도에 있는 장애물 근처의 점(inlier)만 쓴다. 아직 지도에 없는 벽의 점까지 쓰면
        지도 끝 쪽으로 끌려가는 편향이 생긴다.

        scan이 지도와 모순되면 False를 반환한다: 지도에서 빈 공간이라고 확인된 곳에 scan 점이 많이 찍히는 경우로,
        위치가 크게 어긋났다는 뜻이다 (처음 보는 곳은 모순이 아니다).
        """
        planner = self.planner
        if not planner.ready or len(ranges) == 0:
            return True
        fx, fy = self.scan_cells(ranges)
        if len(fx) < 30:
            return True
        distance = self.field_lookup(planner.match_field, fx, fy, MATCH_CAP)
        in_free = self.field_lookup(planner.free_float, fx, fy, 0.0) > 0
        if np.mean(in_free & (distance >= MATCH_CAP)) > 0.25:
            return False
        inlier = distance < MATCH_CAP
        if np.count_nonzero(inlier) < 60 or np.mean(inlier) < 0.4:
            return True     # 처음 보는 곳: 보정하지 않는다.
        fx, fy = fx[inlier], fy[inlier]
        values = self.field_lookup(planner.match_field, fx[None, :] + self.match_dx[:, None],
                                   fy[None, :] + self.match_dy[:, None], MATCH_CAP)
        score = values.mean(axis=1) + MATCH_PENALTY * self.match_norm
        best = int(np.argmin(score))
        if best != self.match_zero and score[self.match_zero] - score[best] >= 0.002:
            shift = MATCH_GAIN * RES * np.array([self.match_dx[best], self.match_dy[best]])
            self.odom.x += shift[0]
            self.odom.y += shift[1]
            self.match_shift += shift
        return True

    def relocalize(self, ranges):
        """위치를 크게 잃었을 때 ±0.6 m 범위(5 cm 간격)에서 지도와 가장 잘 맞는 위치를 다시 찾는다.

        낮은 물건에 올라타 미끄러진 뒤처럼 오차가 Scan Matching의 탐색 범위를 넘은 경우에 쓴다.
        이미 지도에 있는 곳을 보고 있고, 옮긴 위치에서 scan의 대부분이 설명될 때만 받아들인다.
        """
        planner = self.planner
        if not planner.ready:
            return False
        fx, fy = self.scan_cells(ranges)
        if len(fx) < 80:
            return False
        values = self.field_lookup(planner.wide_field, fx[None, :] + self.wide_dx[:, None],
                                   fy[None, :] + self.wide_dy[:, None], 0.5)
        explained = np.mean(values < 0.10, axis=1)
        score = values.mean(axis=1) + 0.03 * self.wide_norm
        best = int(np.argmin(score))
        if best == self.wide_zero or explained[best] < 0.6 or explained[best] < explained[self.wide_zero] + 0.15:
            return False
        shift = RES * np.array([self.wide_dx[best], self.wide_dy[best]])
        self.odom.x += shift[0]
        self.odom.y += shift[1]
        self.match_shift += shift
        self.say(f"위치를 다시 맞췄습니다 (보정 {np.hypot(*shift):.2f} m).", f"relocalized ({np.hypot(*shift):.2f} m)")
        return True

    def register_detections(self, x, y, theta):
        cam_x = x + CAM_OFFSET_X * math.cos(theta)
        cam_y = y + CAM_OFFSET_X * math.sin(theta)
        for detection in self.detections:
            forward, lateral = detection["forward"], detection["lateral"]
            wx = cam_x + forward * math.cos(theta) - lateral * math.sin(theta)
            wy = cam_y + forward * math.sin(theta) + lateral * math.cos(theta)
            distance = math.hypot(forward, lateral)
            detection["world"] = (wx, wy)
            detection["distance"] = distance
            gate = max(0.35, 0.25 * distance)
            best = None
            for track in self.tracks:
                if track.color != detection["color"]:
                    continue
                d = math.hypot(track.x - wx, track.y - wy)
                if d < gate and (best is None or d < best[0]):
                    best = (d, track)
            if best is not None:
                best[1].add(wx, wy, distance)
                detection["track"] = best[1]
            else:
                track = AppleTrack(detection["color"], wx, wy, distance)
                self.tracks.append(track)
                detection["track"] = track
            track = detection["track"]
            if track.count == 3 and not track.rejected:
                role = "구조 대상" if track.color == self.target_color else "대상 아님, 장애물로 기록"
                self.say(f"{track.color} 사과 발견 ({track.x:.2f}, {track.y:.2f}) [{role}]",
                         f"{track.color} apple found at ({track.x:.2f}, {track.y:.2f})")

    # ---------------------------------------------------------------------------------
    # 계획 보조
    # ---------------------------------------------------------------------------------
    def targets(self, visited=None):
        result = [t for t in self.tracks if t.color == self.target_color and t.confirmed]
        if visited is not None:
            result = [t for t in result if t.visited == visited]
        return result

    def refresh_costmap(self, force=False):
        if not force and self.time - self.costmap_time < 0.5:
            return False
        # 사과는 LiDAR보다 낮아 지도에 나타나지 않으므로 가상 장애물로 넣는다.
        objects = [(t.x, t.y, 0.06) for t in self.tracks if t.count >= 2 and not t.rejected]
        person_points = self.person.predicted() if self.person.active(self.time) else []
        self.planner.rebuild(objects, person_points, relaxed=self.relaxed)
        self.costmap_time = self.time
        return True

    def navigate_to_cell(self, cell, avoid_trail=False):
        path = self.planner.plan(cell, avoid_trail)
        if path is None:
            return False
        self.follower.set_path(path, self.odom.x, self.odom.y)
        self.replan_time = self.time
        self.blocked_since = None
        return True

    def plan_to_world(self, x, y, allow_near=True):
        """(x, y)로 가는 경로를 계획한다. 갈 수 없으면 도달 가능한 가장 가까운 셀로 간다."""
        self.refresh_costmap(force=True)
        if not self.planner.reach(self.odom.x, self.odom.y):
            return False
        cell = self.planner.to_cell(x, y)
        if self.planner.inside(*cell) and self.planner.D[cell] >= 0:
            return self.navigate_to_cell(cell)
        if not allow_near:
            return False
        ys, xs = np.nonzero(self.planner.D >= 0)
        if len(ys) == 0:
            return False
        wx, wy = self.planner.to_world(ys, xs)
        best = int(np.argmin(np.hypot(wx - x, wy - y)))
        return self.navigate_to_cell((int(ys[best]), int(xs[best])))

    def follow_path(self):
        """경로를 따라간다. "moving" / "arrived" / "blocked"를 반환한다."""
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        v, w, arrived = self.follower.control(x, y, theta)
        if arrived:
            self.drive(0.0, 0.0)
            self.follower.clear()
            return "arrived"

        # LiDAR 안전 속도 제한: 진행 방향 통로 안의 가장 가까운 점까지의 거리에 비례해 속도를 줄인다.
        v = min(v, max(0.0, 0.8 * (self.front_clearance() - ROBOT_RADIUS - 0.03)))

        # 사람이 가까우면 속도를 줄인다 (길을 비켜 주는 중에는 줄이지 않는다).
        if self.person.active(self.time) and self.person_mode != "EVADE":
            distance = math.hypot(self.person.pos[0] - x, self.person.pos[1] - y)
            if distance < PERSON_SLOW_DISTANCE:
                v = min(v, V_MAX * max(0.3, (distance - 0.6) / (PERSON_SLOW_DISTANCE - 0.6)))

        # 장애물 가까이에서는 천천히 간다 (안전거리 확보).
        cy, cx = self.planner.to_cell(x, y)
        if self.planner.inside(cy, cx):
            clearance = float(self.planner.dist[cy, cx])
            v = min(v, V_MAX * float(np.clip((clearance - 0.08) / 0.15, 0.5, 1.0)))

        if v < 0.01 and abs(w) < 0.05 and not self.follower.rotating:
            if self.blocked_since is None:
                self.blocked_since = self.time
            elif self.time - self.blocked_since > 2.5:
                self.blocked_since = None
                self.drive(0.0, 0.0)
                return "blocked"
        else:
            self.blocked_since = None
        self.drive(v, w)
        return "moving"

    def stalled(self):
        """최근 약 1.3초 동안 Encoder는 전진했는데 지도 정합으로 보정한 위치는 거의 그대로인지 판단한다."""
        if not self.motion or self.time - self.motion[0][0] < 1.3:
            return False
        _, forward0, x0, y0 = self.motion[0]
        encoder_distance = self.odom.forward - forward0
        return encoder_distance > 0.12 and math.hypot(self.odom.x - x0, self.odom.y - y0) < 0.4 * encoder_distance

    def watchdog(self):
        """일시 정지나 양보 중이 아닌데 12초 동안 제자리라면 계획을 버리고 다시 시작한다."""
        _, x0, y0, theta0 = self.watch
        moved = (math.hypot(self.odom.x - x0, self.odom.y - y0) > 0.05
                 or abs(wrap_angle(self.odom.theta - theta0)) > 0.3)
        if moved or self.paused or self.person_mode is not None or self.state in ("DONE", "SPIN"):
            self.watch = (self.time, self.odom.x, self.odom.y, self.odom.theta)
        elif self.time - self.watch[0] > 12.0:
            self.watch = (self.time, self.odom.x, self.odom.y, self.odom.theta)
            self.say("움직이지 못하고 있습니다. 경로를 다시 계획합니다.", "no progress: replanning")
            if self.state == "EXPLORE" and self.goal_cell_world is not None:
                self.planner.ignore_points(np.vstack(self.planner.targets_near(*self.goal_cell_world, VIEW_DISTANCE + 0.1)))
            if self.state == "RETURN":
                self.no_path_count += 4
            self.start_recovery(-0.06 if self.rear_clearance() > ROBOT_RADIUS + 0.15 else 0.0, 1.2)

    def mark_invisible_obstacle(self, korean="보이지 않는 장애물(유리 등)에 막혔습니다. 지도에 기록하고 돌아갑니다.",
                                english="invisible obstacle: marked, replanning"):
        """로봇 앞을 막은, LiDAR에 보이지 않는 장애물(유리창, 낮은 물건, 턱)을 지도에 기록하고 물러난다.

        닿은 지점에 반지름 0.1 m의 원만 표시한다. 넓게 막으면 실제로는 지나갈 수 있는 길까지 막을 수 있기 때문이다.
        """
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        front = (x + (ROBOT_RADIUS + 0.05) * math.cos(theta), y + (ROBOT_RADIUS + 0.05) * math.sin(theta))
        center = tuple(int(v) for v in self.grid.world_to_cell(front[0], front[1]))
        cv2.circle(self.grid.virtual, center, 2, 1, -1)
        self.say(korean, english)
        self.start_recovery(-0.06 if self.rear_clearance() > ROBOT_RADIUS + 0.15 else 0.0, 1.5)

    def rotate_toward(self, angle, tolerance=math.radians(4.0)):
        error = wrap_angle(angle - self.odom.theta)
        if abs(error) < tolerance:
            self.drive(0.0, 0.0)
            return True
        w = float(np.clip(2.5 * error, -W_MAX, W_MAX))
        if abs(w) < 0.3:
            w = math.copysign(0.3, w)
        self.drive(0.0, w)
        return False

    # ---------------------------------------------------------------------------------
    # 사람 대응: 예상 경로를 피해 계획하고, 가까워지면 양보(정지)하거나 회피한다
    # ---------------------------------------------------------------------------------
    def handle_person(self):
        """사람 대응으로 이번 step의 행동을 결정했으면 True를 반환한다.

        EVADE: 사람의 예상 경로가 로봇 가까이 지나가면, 그 경로에서 가장 멀어지는 곳으로 미리 비켜 준다.
        YIELD: 위험하지는 않지만 사람이 바로 앞에 있으면 멈춰서 기다린다.
        """
        if not self.person.active(self.time):
            if self.person_mode is not None:
                self.say("사람이 지나갔습니다. 임무를 계속합니다.", "person passed, resuming")
                self.person_mode = None
                self.follower.clear()
            return False
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        rel = self.person.pos - (x, y)
        distance = float(np.hypot(*rel))
        approach = self.person.closest_approach(x, y)
        if approach < PERSON_DANGER_DISTANCE and distance < 3.0 and (self.person.moving() or distance < 0.6):
            self.danger_time = self.time

        if self.person_mode != "EVADE" and self.danger_time == self.time:
            self.person_mode = "EVADE"
            self.evade_time = -1e9
            self.follower.clear()
            self.say(f"사람이 다가옵니다 (거리 {distance:.2f} m, 예상 최근접 {approach:.2f} m). 길을 비켜 줍니다.",
                     f"person approaching ({distance:.2f} m): moving aside")
        elif self.person_mode == "EVADE" and self.time - self.danger_time > 1.5:
            self.say("사람과 충분히 멀어졌습니다. 임무를 계속합니다.", "person clear, resuming")
            self.person_mode = None
            self.follower.clear()

        if self.person_mode == "EVADE":
            if self.time - self.evade_time > 0.7:
                self.evade_time = self.time
                self.refresh_costmap(force=True)
                points = [self.person.pos + self.person.vel * dt for dt in PERSON_HORIZON]
                escaping = False
                if self.planner.reach(x, y, start_radius=1.2):
                    cell, clearance = self.planner.escape(points)
                    escaping = cell is not None and clearance > approach + 0.15 and self.navigate_to_cell(cell)
                if not escaping:
                    self.follower.clear()
            if self.follower.path is not None:
                if self.follow_path() != "moving":
                    self.follower.clear()
            else:
                self.drive(0.0, 0.0)
            return True

        bearing = abs(wrap_angle(math.atan2(rel[1], rel[0]) - theta))
        if distance < PERSON_YIELD_DISTANCE and bearing < math.radians(70.0):
            if self.person_mode != "YIELD":
                self.say(f"앞에 사람이 있습니다 (거리 {distance:.2f} m). 멈춰서 양보합니다.",
                         f"person ahead ({distance:.2f} m): yielding")
            self.person_mode = "YIELD"
            self.drive(0.0, 0.0)
            return True
        if self.person_mode == "YIELD":
            self.person_mode = None
            self.follower.clear()
        return False

    # ---------------------------------------------------------------------------------
    # 상태 기계 (FSM)
    # ---------------------------------------------------------------------------------
    def set_state(self, state):
        if state != self.state:
            self.state = state
            self.follower.clear()
            self.blocked_since = None

    def start_return(self, reason):
        self.home_best = (float("inf"), self.time)
        self.home_close_enough = False
        self.relaxed = False            # 복귀 경로가 없어 Costmap의 여유를 줄였는지 여부.
        self.return_reason = reason
        self.say(f"{reason} 시작 지점으로 복귀합니다.", "returning to start")
        self.set_state("RETURN")

    def decide(self):
        if self.state == "DONE":
            self.drive(0.0, 0.0)
            return
        if self.paused:
            self.drive(0.0, 0.0)
            return
        costmap_updated = self.refresh_costmap()
        # 크게 기울어졌다면 LiDAR에 보이지 않는 낮은 물건(책 등)에 올라탄 것이다. 기록하고 물러난다.
        if self.tilt > 0.9 and self.time >= self.recover_until:
            self.mark_invisible_obstacle("낮은 장애물에 올라탔습니다. 지도에 기록하고 물러납니다.", "climbed a low obstacle: backing off")
            self.recover_until = self.time + 2.0
        self.watchdog()
        # 바퀴는 앞으로 돌았는데 실제로는 나아가지 못했다면 보이지 않는 장애물(유리창 등)에 막힌 것이다.
        if self.command_v > 0.05 and self.time >= self.recover_until and self.stalled():
            self.motion.clear()
            self.mark_invisible_obstacle()
        if self.state != "SPIN" and self.handle_person():
            return
        if self.time < self.recover_until:
            # 물러나거나 빠져나오는 중에도 진행 방향이 막히면 멈춘다.
            clearance = self.front_clearance() if self.recover_speed > 0 else self.rear_clearance()
            self.drive(self.recover_speed if clearance > ROBOT_RADIUS + 0.06 else 0.0, 0.0)
            return

        # 제한 시간이 있으면 복귀에 필요한 시간을 남긴다.
        if (self.time_limit > 0 and self.state in ("EXPLORE", "LOOK", "APPROACH", "CONFIRM")
                and self.time - self.status_time > 2.0):
            self.status_time = self.time
            home = math.hypot(self.odom.x - START_X, self.odom.y - START_Y)
            if self.time + 1.6 * home / (0.7 * V_MAX) + 15.0 > self.time_limit:
                self.start_return("제한 시간이 얼마 남지 않았습니다.")

        getattr(self, "state_" + self.state.lower())(costmap_updated)

    def state_spin(self, _):
        """시작 지점에서 한 바퀴 돌며 주변 지도와 카메라 시야를 확보한다."""
        if self.spin_prev is not None:
            self.spin_angle += abs(wrap_angle(self.odom.theta - self.spin_prev))
        self.spin_prev = self.odom.theta
        if self.spin_angle >= 2.0 * math.pi + 0.15:
            self.drive(0.0, 0.0)
            self.say("주변 확인을 마쳤습니다. 탐색을 시작합니다.", "initial scan done, exploring")
            self.set_state("EXPLORE")
        else:
            self.drive(0.0, 1.3)

    def next_target(self):
        """아직 가지 않은 구조 대상 중 접근 지점까지의 경로가 가장 짧은 것을 고른다."""
        best = None
        for track in self.targets(visited=False):
            if track.attempts >= 5:
                continue        # 여러 번 시도해도 확인하지 못한 사과는 포기한다.
            cell = self.planner.standoff(track.x, track.y)
            if cell is not None and (best is None or self.planner.D[cell] < best[0]):
                best = (self.planner.D[cell], track, cell)
        return best

    def state_explore(self, costmap_updated):
        if len(self.targets(visited=True)) >= self.target_count:
            self.start_return(f"{self.target_color} 사과 {self.target_count}개를 모두 확인했습니다.")
            return

        need_goal = self.follower.path is None
        if not need_goal and costmap_updated:
            # 지금 자리에서 돌기만 해도 볼 수 있는 미확인 영역이 넓으면 먼저 둘러본다 (방 입구 등).
            if (self.odom.distance - self.look_travel >= 1.0
                    and len(self.visible_unseen()) >= LOOK_TRIGGER_CELLS):
                self.start_look(arrival=False)
                return
            if self.planner.blocked(self.follower.path, self.follower.index):
                need_goal = True
            elif self.goal_cell_world is not None:
                unseen, frontier = self.planner.targets_near(*self.goal_cell_world, VIEW_DISTANCE + 0.1)
                if len(unseen) + len(frontier) == 0:
                    need_goal = True        # 가는 도중에 이미 확인된 영역이다.
            if not need_goal and self.targets(visited=False) and self.time - self.replan_time > 1.0:
                need_goal = True            # 새로 발견한 구조 대상이 있다.

        if need_goal:
            self.refresh_costmap(force=True)
            if not self.planner.reach(self.odom.x, self.odom.y):
                self.recover()
                return
            target = self.next_target()
            if target is not None:
                _, track, cell = target
                if self.navigate_to_cell(cell):
                    self.current_track = track
                    self.track_goal = (track.x, track.y)
                    self.say(f"{track.color} 사과 ({track.x:.2f}, {track.y:.2f})로 이동합니다.",
                             f"approaching {track.color} apple")
                    self.state = "APPROACH"
                    return
            person_near = self.person.active(self.time)
            pending = [(t.x, t.y) for t in self.targets(visited=False) if t.attempts < 5]
            # 탐색 목표는 이미 지나간 자리를 더 먼 것으로 계산한 거리와 오른쪽 우선 규칙으로 고른다.
            self.planner.reach(self.odom.x, self.odom.y, trail_cost=TRAIL_COST)
            cell = self.planner.exploration_goal(allow_ignore=not person_near, attract=pending,
                                                 pose=(self.odom.x, self.odom.y, self.travel_heading()))
            if cell is None or not self.navigate_to_cell(cell, avoid_trail=True):
                if cell is None:
                    # 사람이 길을 막고 있어 갈 곳이 없으면 지나갈 때까지 기다린다.
                    if not (person_near and self.planner.targets_left):
                        self.exploration_finished()
                else:
                    wx, wy = self.planner.to_world(*cell)
                    self.planner.ignore_points(np.vstack(self.planner.targets_near(float(wx), float(wy), VIEW_DISTANCE + 0.1)))
                self.drive(0.0, 0.0)
                return
            wx, wy = self.planner.to_world(*cell)
            goal = (float(wx), float(wy))
            # 같은 곳을 목표로 40초 넘게 도착하지 못하고 있으면 (의자 다리 사이처럼 다가갈 수 없는 곳) 포기한다.
            if self.goal_cell_world is None or math.hypot(goal[0] - self.goal_cell_world[0], goal[1] - self.goal_cell_world[1]) > 0.5:
                self.goal_since = self.time
            elif self.time - self.goal_since > 40.0:
                self.give_up_goal(goal)
                self.drive(0.0, 0.0)
                return
            self.goal_cell_world = goal

        status = self.follow_path()
        if status == "arrived":
            self.start_look(arrival=True)
        elif status == "blocked":
            self.recover()

    def travel_heading(self):
        """최근 0.5 m를 이동한 방향 [rad]. 제자리에서 둘러본 뒤에도 진행 방향의 기준이 바뀌지 않게 한다."""
        if len(self.trajectory) > 10:
            (x0, y0), (x1, y1) = self.trajectory[-11], self.trajectory[-1]
            return math.atan2(y1 - y0, x1 - x0)
        return self.odom.theta

    def give_up_goal(self, goal):
        """도착하지 못하는 탐색 목표와 그 주변, 목표가 향하던 덩어리를 탐색 대상에서 뺀다."""
        ix, iy = (int(v) for v in self.grid.world_to_cell(goal[0], goal[1]))
        cv2.circle(self.grid.ignored_u8, (ix, iy), int(1.0 / RES), 1, -1)
        cluster = self.planner.goal_cluster
        if len(cluster):
            self.planner.ignore_points(cluster[np.hypot(cluster[:, 0] - goal[0], cluster[:, 1] - goal[1]) < 2.0])
        self.follower.clear()
        self.goal_cell_world = None

    def visible_unseen(self):
        """지금 자리에서 회전만 하면 카메라로 볼 수 있는 미확인 셀들의 방향 (로봇 기준 각도 [rad])."""
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        unseen, _ = self.planner.targets_near(x, y, CAM_RANGE - 0.2)
        if len(unseen) == 0:
            return np.empty(0)
        dx, dy = unseen[:, 0] - x, unseen[:, 1] - y
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - theta
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        count = len(self.ranges)
        index = np.round((math.pi - bearing) / (2.0 * math.pi / count) - 0.5).astype(np.int64) % count
        # LiDAR가 그 방향으로 셀보다 멀리 보고 있으면 가려지지 않은 셀이다.
        return bearing[(self.ranges[index] > distance - 0.05) & (distance > 0.25)]

    def finish_arrival(self):
        """탐색 목표에 도착해 둘러본 뒤에도 남은 영역은 여기서는 관측할 수 없는 곳이므로 탐색 대상에서 뺀다.

        로봇 주변 1 m와, 이번 목표가 향하던 덩어리 중 2 m 안의 부분을 뺀다.
        식탁 밑처럼 의자 다리에 가려 끝내 보이지 않는 곳을 계속 다시 찾아가지 않기 위해서다.
        정밀 탐색에서는 0.5 m 안만 빼서 작은 구석도 여러 방향에서 확인한다.
        """
        x, y = self.odom.x, self.odom.y
        # 주변 원 안은 통째로 제외한다. 계단처럼 LiDAR 값이 깜빡여 탐색 대상이 계속 새로 생기는 곳에서 맴돌지 않기 위해서다.
        ix, iy = (int(v) for v in self.grid.world_to_cell(x, y))
        cv2.circle(self.grid.ignored_u8, (ix, iy), int((0.5 if self.planner.thorough else 1.0) / RES), 1, -1)
        cluster = self.planner.goal_cluster
        if len(cluster) and not self.planner.thorough:
            self.planner.ignore_points(cluster[np.hypot(cluster[:, 0] - x, cluster[:, 1] - y) < 2.0])

    def start_look(self, arrival):
        self.follower.clear()
        self.drive(0.0, 0.0)
        if arrival and np.count_nonzero(np.abs(self.visible_unseen()) > self.cam_half_fov) < 20:
            self.finish_arrival()       # 돌아볼 것이 없다.
            return
        self.look_arrival = arrival
        self.look_turned = 0.0
        self.look_prev = None
        self.look_start = self.time
        self.state = "LOOK"

    def exploration_finished(self):
        found = len(self.targets())
        if found < self.target_count and not self.planner.thorough:
            # 1차 탐색으로 다 찾지 못했다. 제외했던 영역과 작은 구석까지 다시 확인한다.
            self.planner.thorough = True
            self.grid.ignored[:] = False
            self.say(f"1차 탐색을 마쳤습니다 (발견 {found}/{self.target_count}). 가려진 구석까지 정밀 탐색합니다.",
                     "first pass done, searching hidden corners")
            return
        unvisited = [t for t in self.targets(visited=False) if t.attempts < 5]
        if unvisited and any(t.attempts < 3 for t in unvisited):
            for track in unvisited:
                track.attempts += 1
            return
        self.start_return(f"탐색을 마쳤습니다 (발견 {found}/{self.target_count}).")

    def state_look(self, _):
        """제자리에서 회전하며, 가려지지 않았지만 아직 카메라로 보지 못한 방향을 확인한다."""
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        bearings = self.visible_unseen()
        remaining = bearings[np.abs(bearings) > self.cam_half_fov]   # 아직 시야 밖에 있는 방향.
        if self.look_prev is None:
            if len(remaining):
                # 조금만 돌면 되는 쪽으로 돌기 시작한다.
                self.look_direction = 1.0 if remaining[np.argmin(np.abs(remaining))] > 0 else -1.0
        else:
            self.look_turned += abs(wrap_angle(theta - self.look_prev))
        self.look_prev = theta

        if len(remaining) < 20 or self.look_turned > 2.0 * math.pi + 0.2 or self.time - self.look_start > 9.0:
            self.drive(0.0, 0.0)
            if self.look_arrival:
                self.finish_arrival()
            self.look_travel = self.odom.distance
            self.state = "EXPLORE"
            return
        self.drive(0.0, self.look_direction * 1.3)

    def state_approach(self, costmap_updated):
        track = self.current_track
        if track is None or track.visited or track.rejected:
            self.set_state("EXPLORE")
            return
        if self.follower.path is None:
            if not self.replan_approach():
                return
        elif costmap_updated:
            moved = math.hypot(track.x - self.track_goal[0], track.y - self.track_goal[1])
            if moved > 0.15 or self.planner.blocked(self.follower.path, self.follower.index):
                if not self.replan_approach():
                    return
        status = self.follow_path()
        if status == "arrived":
            self.confirm_start = None
            self.state = "CONFIRM"
        elif status == "blocked":
            self.recover()

    def replan_approach(self):
        track = self.current_track
        self.refresh_costmap(force=True)
        cell = None
        if self.planner.reach(self.odom.x, self.odom.y):
            cell = self.planner.standoff(track.x, track.y)
        if cell is None or not self.navigate_to_cell(cell):
            track.attempts += 1
            self.say(f"{track.color} 사과로 가는 길을 찾지 못했습니다. 탐색을 계속합니다.", "no path to apple, exploring")
            self.drive(0.0, 0.0)
            self.current_track = None
            self.set_state("EXPLORE")
            return False
        self.track_goal = (track.x, track.y)
        return True

    def state_confirm(self, _):
        """사과를 바라보고 카메라로 다시 확인한다 (오탐 방지)."""
        track = self.current_track
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        if self.confirm_start is None:
            self.confirm_start = self.time
            self.confirm_seen = None
        bearing = math.atan2(track.y - y, track.x - x)
        facing = self.rotate_toward(bearing)
        for detection in self.detections:
            if detection.get("track") is track:
                self.confirm_seen = detection["distance"]
        distance = math.hypot(track.x - x, track.y - y)
        if facing and self.confirm_seen is not None and distance <= REACHED_DISTANCE:
            self.mark_reached(track, distance)
        elif self.time - self.confirm_start > 4.0:
            track.attempts += 1
            if self.confirm_seen is None:
                count = len(self.ranges)
                index = int(round((math.pi - wrap_angle(bearing - theta)) / (2.0 * math.pi / count) - 0.5)) % count
                in_sight = self.ranges[index] > distance - 0.1
                if track.count >= 12 and distance <= REACHED_DISTANCE:
                    # 여러 번 일관되게 관측한 사과는 지금 가려져 보이지 않아도 도착으로 인정한다.
                    self.mark_reached(track, distance)
                    return
                if in_sight and track.attempts >= 3:
                    track.rejected = True
                    self.say(f"({track.x:.2f}, {track.y:.2f})에서 사과를 다시 확인하지 못해 후보에서 제외합니다.", "apple not confirmed, rejected")
            self.finish_confirm()

    def mark_reached(self, track, distance):
        # 위치 오차 때문에 같은 사과가 두 개로 기록된 경우 다시 세지 않는다.
        for other in self.targets(visited=True):
            if other is not track and math.hypot(other.x - track.x, other.y - track.y) < 0.5:
                track.rejected = True
                self.say(f"({track.x:.2f}, {track.y:.2f})의 사과는 이미 확인한 사과와 같은 것으로 판단합니다.", "duplicate of a reached apple")
                self.finish_confirm()
                return
        track.visited = True
        track.visit_time = self.time
        done = len(self.targets(visited=True))
        self.say(f"구조 대상 {done}/{self.target_count} 도착: {track.color} 사과 ({track.x:.2f}, {track.y:.2f}), 거리 {distance:.2f} m",
                 f"TARGET {done}/{self.target_count} REACHED at ({track.x:.2f}, {track.y:.2f})")
        self.finish_confirm()

    def finish_confirm(self):
        self.confirm_start = None
        self.current_track = None
        self.set_state("EXPLORE")

    def state_return(self, costmap_updated):
        home_distance = math.hypot(self.odom.x - START_X, self.odom.y - START_Y)
        if home_distance < self.home_best[0] - 0.03:
            self.home_best = (home_distance, self.time)
        elif home_distance < 0.5 and self.time - self.home_best[1] > 10.0:
            # 시작 지점 바로 옆에서 10초 동안 더 가까워지지 못했다 (위치 오차로 벽에 막힌 경우). 여기서 마친다.
            self.home_close_enough = True
            self.follower.clear()
        if self.follower.path is None:
            if home_distance < GOAL_TOLERANCE + 0.04 or self.home_close_enough:
                if self.rotate_toward(START_THETA):
                    self.finish()
                return
            if self.time - self.return_plan_time < 1.0:
                self.drive(0.0, 0.0)
                return
            self.return_plan_time = self.time
            if home_distance < 0.5 and self.no_path_count >= 8:
                # 시작 지점 바로 옆인데 더 다가갈 수 없으면 (위치 오차로 벽에 막힌 경우) 여기서 마친다.
                self.home_close_enough = True
                return
            if not self.plan_to_world(START_X, START_Y) or self.follower.remaining(self.odom.x, self.odom.y) < 0.02:
                self.no_path_count += 1
                if self.no_path_count >= 6 and not self.relaxed:
                    self.relaxed = True
                    self.say("복귀 경로가 없습니다. 기록해 둔 보이지 않는 장애물과 카펫을 무시하고 여유를 줄여 다시 찾습니다.",
                             "no path home: relaxing costmap")
                if self.no_path_count % 5 == 1:
                    self.say("복귀 경로를 찾는 중입니다.", "searching path home")
                self.recover()
                return
        elif costmap_updated and self.planner.blocked(self.follower.path, self.follower.index):
            self.follower.clear()
            self.drive(0.0, 0.0)
            return
        elif costmap_updated and self.time - self.replan_time > 4.0:
            self.follower.clear()   # 지도가 바뀌었을 수 있으니 주기적으로 다시 계획한다 (다음 step에 바로 이어 간다).
            return
        status = self.follow_path()
        if status == "blocked":
            self.no_path_count += 1
            self.recover()

    def recover(self):
        """막혔을 때: 뒤가 비어 있으면 조금 물러난 뒤 다시 계획한다."""
        if self.time >= self.recover_until:
            self.start_recovery(-0.06 if self.rear_clearance() > ROBOT_RADIUS + 0.15 else 0.0, 1.2)
        self.drive(self.recover_speed, 0.0)

    def finish(self):
        self.drive(0.0, 0.0)
        self.finish_time = self.time
        self.state = "DONE"
        visited = self.targets(visited=True)
        error = math.hypot(self.odom.x - START_X, self.odom.y - START_Y)
        success = len(visited) >= self.target_count
        self.say(f"임무 {'완료' if success else '종료'}: {self.target_color} 사과 {len(visited)}/{self.target_count}개 확인, "
                 f"소요 시간 {self.time:.1f} s, 주행 거리 {self.odom.distance:.1f} m, 시작 지점과의 거리 {error:.2f} m",
                 f"MISSION {'COMPLETE' if success else 'ENDED'}: {len(visited)}/{self.target_count} in {self.time:.0f}s")
        self.save_report(success)

    # ---------------------------------------------------------------------------------
    # Human-Robot Interface: 텍스트 명령
    # ---------------------------------------------------------------------------------
    HELP = ("명령: find <색> <개수> | pause | resume | home | status | time <초> | help\n"
            "  예) find 2 red apples / 초록 사과 1개 찾아줘 / pause / home\n"
            "  입력 방법: Webots 3D 화면 또는 대시보드 창에서 입력 후 Enter, 또는 command.txt 파일에 한 줄 쓰기")

    def handle_command(self, text):
        raw = text.strip()
        low = raw.lower()
        if not low:
            return
        self.say(f'명령 수신: "{raw}"', f'command: "{raw}"')
        if "help" in low or "도움" in low or low == "?":
            print(self.HELP, flush=True)
        elif any(word in low for word in ("status", "report", "상태")):
            self.print_status()
        elif any(word in low for word in ("pause", "stop", "wait", "정지", "멈춰", "멈춤")):
            self.paused = True
            self.say("일시 정지합니다. resume을 입력하면 계속합니다.", "paused")
        elif any(word in low for word in ("resume", "continue", "계속", "재개")) or low in ("go", "start"):
            self.paused = False
            self.say("임무를 계속합니다.", "resumed")
        elif any(word in low for word in ("home", "return", "back", "복귀", "돌아")):
            self.paused = False
            if self.state != "DONE":
                self.start_return("사람의 요청으로")
        else:
            limit = re.search(r"(?:time|limit|시간|제한)\s*=?\s*(\d+(?:\.\d+)?)", low)
            if limit:
                self.time_limit = float(limit.group(1))
                low = low.replace(limit.group(0), " ")
            color = next((name for name, words in COLOR_WORDS.items() if any(word in low for word in words)), None)
            count = re.search(r"(?<!\d)([1-9])(?!\d)", low)
            if color is None and count is None and limit is None:
                self.say("이해하지 못한 명령입니다. help를 입력해 보세요.", "unknown command (try: help)")
                return
            if color is not None:
                self.target_color = color
            if count is not None:
                self.target_count = int(count.group(1))
            self.paused = False
            limit_text = f", 제한 시간 {self.time_limit:.0f} s" if self.time_limit > 0 else ""
            self.say(f"임무 설정: {self.target_color} 사과 {self.target_count}개 찾기{limit_text}",
                     f"mission: find {self.target_count} {self.target_color} apple(s)")
            if self.state in ("RETURN", "DONE", "APPROACH", "CONFIRM", "LOOK") and (color or count):
                self.current_track = None
                self.confirm_start = None
                self.planner.thorough = False
                self.set_state("EXPLORE")

    def print_status(self):
        found, visited = self.targets(), self.targets(visited=True)
        print(f"  상태: {self.state}{' (일시 정지)' if self.paused else ''}, 시간 {self.time:.1f} s, 주행 거리 {self.odom.distance:.1f} m\n"
              f"  위치: ({self.odom.x:.2f}, {self.odom.y:.2f}), 방향 {math.degrees(self.odom.theta):.0f} deg\n"
              f"  임무: {self.target_color} 사과 {self.target_count}개 (발견 {len(found)}, 도착 {len(visited)})", flush=True)
        for track in self.tracks:
            if track.confirmed:
                print(f"    - {track.color} 사과 ({track.x:.2f}, {track.y:.2f}){' [도착]' if track.visited else ''}", flush=True)

    def read_commands(self, cv_key):
        """Webots 키보드, 대시보드 창(OpenCV), command.txt 세 경로로 텍스트 명령을 받는다."""
        typed = []
        pressed = set()
        key = self.keyboard.getKey()
        while key != -1:
            code = key & 0xFFFF
            if code not in pressed and code not in self.pressed_keys:
                typed.append(code)      # 새로 눌린 키만, 읽은 순서대로.
            pressed.add(code)
            key = self.keyboard.getKey()
        self.pressed_keys = pressed
        if cv_key not in (-1, 255):
            typed.append(cv_key & 0xFF)
        for code in typed:
            if code in (4, 10, 13):                 # Enter
                command, self.command_buffer = self.command_buffer, ""
                self.handle_command(command)
            elif code in (3, 8, 127):               # Backspace
                self.command_buffer = self.command_buffer[:-1]
            elif 32 <= code < 127 and len(self.command_buffer) < 60:
                self.command_buffer += chr(code).lower()

        if self.step_count % 16 == 0 and os.path.exists(self.command_file):
            try:
                with open(self.command_file, encoding="utf-8") as f:
                    lines = [line for line in f.read().splitlines() if line.strip()]
                if lines:
                    open(self.command_file, "w").close()
                    for line in lines:
                        self.handle_command(line)
            except OSError:
                pass

    # ---------------------------------------------------------------------------------
    # Human-Robot Interface: 대시보드, 상태 공유, 임무 보고서
    # ---------------------------------------------------------------------------------
    def render_map(self, size):
        iy0, iy1, ix0, ix1 = self.grid.bbox
        log_odds = self.grid.log_odds[iy0:iy1, ix0:ix1]
        seen = self.grid.seen[iy0:iy1, ix0:ix1]
        image = np.full(log_odds.shape + (3,), 128, np.uint8)
        free = log_odds < L_FREE_THRESHOLD
        image[free & seen] = (255, 255, 255)       # 카메라로 확인한 빈 공간.
        image[free & ~seen] = (235, 215, 190)      # 지도에는 있지만 카메라로 아직 보지 못한 곳.
        image[log_odds > L_OCC_THRESHOLD] = (0, 0, 0)
        image[self.grid.rough[iy0:iy1, ix0:ix1] >= ROUGH_CONFIRM] = (90, 90, 90)   # 어두운 바닥 (카펫).
        image[self.grid.virtual[iy0:iy1, ix0:ix1] > 0] = (160, 120, 0)   # 보이지 않는 장애물 (유리 등).
        image = np.flipud(image)
        scale = max(1, min(size // image.shape[0], size // image.shape[1]))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        def pixel(x, y):
            u = ((x - self.grid.origin_x) / RES - ix0) * scale - 0.5
            v = (iy1 - (y - self.grid.origin_y) / RES) * scale - 0.5
            return int(round(u)), int(round(v))

        if len(self.trajectory) >= 2:
            cv2.polylines(image, [np.array([pixel(x, y) for x, y in self.trajectory], np.int32)], False, (170, 170, 170), 1)
        if self.follower.path is not None and len(self.follower.path) >= 2:
            points = np.array([pixel(x, y) for x, y in self.follower.path[self.follower.index:]], np.int32)
            cv2.polylines(image, [points], False, (255, 120, 0), 2, cv2.LINE_AA)
        home = pixel(START_X, START_Y)
        cv2.rectangle(image, (home[0] - 6, home[1] - 6), (home[0] + 6, home[1] + 6), (0, 160, 0), 2)
        for track in self.tracks:
            if not track.confirmed:
                continue
            center = pixel(track.x, track.y)
            cv2.circle(image, center, max(4, scale * 2), COLOR_BGR[track.color], -1, cv2.LINE_AA)
            if track.color == self.target_color:
                cv2.circle(image, center, max(8, scale * 4), (0, 200, 0) if track.visited else (0, 0, 255), 2, cv2.LINE_AA)
        if self.person.active(self.time):
            center = pixel(*self.person.pos)
            cv2.circle(image, center, int(PERSON_RADIUS / RES * scale), (255, 0, 255), 2, cv2.LINE_AA)
            tip = pixel(*(self.person.pos + self.person.vel * 3.0))
            cv2.arrowedLine(image, center, tip, (255, 0, 255), 2, cv2.LINE_AA)
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        center = pixel(x, y)
        cv2.circle(image, center, max(3, int(ROBOT_RADIUS / RES * scale)), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, center, pixel(x + 0.25 * math.cos(theta), y + 0.25 * math.sin(theta)), (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def render_dashboard(self):
        canvas = np.full((720, 1180, 3), 40, np.uint8)
        map_image = self.render_map(700)
        h, w = min(map_image.shape[0], 700), min(map_image.shape[1], 700)
        canvas[10:10 + h, 10:10 + w] = map_image[:h, :w]
        if self.frame is not None:
            view = self.frame.copy()
            for detection in self.detections:
                color = COLOR_BGR[detection["color"]]
                center = (int(detection["u"]), int(detection["v"]))
                cv2.circle(view, center, int(detection["r"]) + 3, color, 2)
                cv2.putText(view, f'{detection["color"]} {detection["distance"]:.2f}m', (center[0] - 40, center[1] - int(detection["r"]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            canvas[10:340, 730:1170] = cv2.resize(view, (440, 330))
        mode = "PAUSED" if self.paused else (self.person_mode or self.state)
        found, visited = len(self.targets()), len(self.targets(visited=True))
        lines = [
            (f"STATE: {mode}", (0, 255, 255)),
            (f"MISSION: find {self.target_count} {self.target_color} apple(s)   found {found}  reached {visited}", (255, 255, 255)),
            (f"time {self.time:6.1f} s   distance {self.odom.distance:5.1f} m", (255, 255, 255)),
            (f"pose ({self.odom.x:.2f}, {self.odom.y:.2f}) m, {math.degrees(self.odom.theta):.0f} deg", (255, 255, 255)),
            ("PERSON: tracked" if self.person.active(self.time) else "PERSON: none", (255, 0, 255) if self.person.active(self.time) else (150, 150, 150)),
        ]
        lines += [(message, (200, 200, 200)) for message in self.messages]
        for index, (text, color) in enumerate(lines):
            cv2.putText(canvas, text[:62], (730, 375 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, "> " + self.command_buffer + "_", (730, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "type: find red 2 | pause | resume | home | status  + Enter", (730, 710),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        return canvas

    def share(self):
        """대시보드를 그리고, 로봇의 customData에 현재 상태를 JSON으로 공유한다."""
        key = -1
        if self.time - self.draw_time >= 0.25 and self.grid.bbox is not None:
            self.draw_time = self.time
            status = {"t": round(self.time, 2), "state": self.state, "paused": self.paused, "person": self.person_mode,
                      "x": round(self.odom.x, 3), "y": round(self.odom.y, 3), "theta": round(self.odom.theta, 3),
                      "found": len(self.targets()), "reached": len(self.targets(visited=True))}
            self.robot.setCustomData(json.dumps(status))
            if self.show_gui:
                try:
                    cv2.imshow("Search and Rescue Dashboard", self.render_dashboard())
                    key = cv2.waitKey(1)
                except cv2.error:
                    self.show_gui = False
                    print("화면을 열 수 없어 대시보드를 끕니다.", flush=True)
        return key

    def save_report(self, success):
        folder = os.path.dirname(os.path.abspath(__file__))
        report = {
            "success": bool(success),
            "target": {"color": self.target_color, "count": self.target_count},
            "reached": [{"x": round(t.x, 3), "y": round(t.y, 3), "time": round(t.visit_time, 1)} for t in self.targets(visited=True)],
            "objects": [{"color": t.color, "x": round(t.x, 3), "y": round(t.y, 3), "observations": t.count}
                        for t in self.tracks if t.confirmed],
            "elapsed_time": round(self.time, 1),
            "travel_distance": round(self.odom.distance, 2),
            "start_pose": [START_X, START_Y, START_THETA],
            "final_pose": [round(self.odom.x, 3), round(self.odom.y, 3), round(self.odom.theta, 3)],
            "return_reason": self.return_reason,
        }
        try:
            with open(os.path.join(folder, "mission_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            cv2.imwrite(os.path.join(folder, "mission_map.png"), self.render_map(900))
            self.say("임무 보고서를 저장했습니다: mission_report.json, mission_map.png", "report saved")
        except OSError as error:
            print(f"보고서 저장 실패: {error}", flush=True)


# =====================================================================================
# main
# =====================================================================================
def main():
    global START_X, START_Y, START_THETA
    show_gui = True
    command_words = []
    for argument in sys.argv[1:]:
        if argument.lower() == "nogui":
            show_gui = False
        elif argument.lower().startswith("start="):
            START_X, START_Y, START_THETA = (float(value) for value in argument[6:].split(","))
        else:
            command_words.append(argument)

    robot = Robot()
    mission = Mission(robot, show_gui)
    print("=== Autonomous Search and Rescue ===", flush=True)
    print(mission.HELP, flush=True)
    if command_words:
        mission.handle_command(" ".join(command_words))
    else:
        mission.say(f"임무 설정: {mission.target_color} 사과 {mission.target_count}개 찾기",
                    f"mission: find {mission.target_count} {mission.target_color} apple(s)")

    cv_key = -1
    while robot.step(mission.timestep) != -1:
        if not mission.sense():
            mission.drive(0.0, 0.0)     # 센서 값이 유효하지 않으면 움직이지 않는다.
            continue
        mission.read_commands(cv_key)
        mission.decide()
        cv_key = mission.share()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
