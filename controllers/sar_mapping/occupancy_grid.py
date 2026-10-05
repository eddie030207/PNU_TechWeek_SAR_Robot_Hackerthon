import math

import numpy as np

# Trinary 표현 값.
UNKNOWN = -1
FREE = 0
OCCUPIED = 100


class OccupancyGrid:
    """LiDAR scan으로 갱신하는 log-odds 기반 Occupancy Grid Map.

    log_odds[iy, ix]가 셀 하나의 값이며 0이면 미관측(P = 0.5)이다.
    """

    def __init__(self, center_x, center_y, size_m=40.0, resolution=0.05,
                 l_occ=0.85, l_free=-0.4, l_min=-4.0, l_max=4.0,
                 p_occ_threshold=0.65, p_free_threshold=0.35):
        self.resolution = resolution
        self.size = int(round(size_m / resolution))

        # 셀 (0, 0)의 왼쪽 아래 모서리 월드 좌표.
        self.origin_x = center_x - size_m / 2.0
        self.origin_y = center_y - size_m / 2.0

        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)

        self.l_occ = l_occ
        self.l_free = l_free
        self.l_min = l_min
        self.l_max = l_max
        self.l_occ_threshold = math.log(p_occ_threshold / (1.0 - p_occ_threshold))
        self.l_free_threshold = math.log(p_free_threshold / (1.0 - p_free_threshold))

    def world_to_cell(self, x, y):
        """월드 좌표 [m]를 셀 index (ix, iy)로 변환한다. 배열 입력도 가능하다."""
        ix = np.floor((np.asarray(x) - self.origin_x) / self.resolution).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin_y) / self.resolution).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        """셀 index의 중심점 월드 좌표 [m]를 반환한다."""
        x = self.origin_x + (np.asarray(ix) + 0.5) * self.resolution
        y = self.origin_y + (np.asarray(iy) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, ix, iy):
        return (ix >= 0) & (ix < self.size) & (iy >= 0) & (iy < self.size)

    def update(self, sensor_x, sensor_y, sensor_theta, ranges, ray_angles, min_range, max_range):
        """LiDAR scan 한 번으로 지도를 갱신한다.

        sensor_x, sensor_y, sensor_theta: 월드 기준 LiDAR pose.
        ranges: 측정 거리 [m]. 최대 거리 안에 장애물이 없으면 inf.
        ray_angles: 각 측정값의 센서 기준 방향 [rad].
        """
        ranges = np.asarray(ranges, dtype=np.float64)
        ray_angles = np.asarray(ray_angles, dtype=np.float64)
        if ranges.shape != ray_angles.shape:
            raise ValueError("ranges와 ray_angles의 길이가 다릅니다.")

        hit = np.isfinite(ranges) & (ranges >= min_range) & (ranges <= max_range)
        no_return = np.isposinf(ranges)
        valid = hit | no_return
        if not valid.any():
            return

        angles = sensor_theta + ray_angles
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # 장애물 셀 (ray의 끝점).
        hit_x = sensor_x + ranges[hit] * cos_a[hit]
        hit_y = sensor_y + ranges[hit] * sin_a[hit]
        occ_flat = self._flat_indices(hit_x, hit_y)

        # 빈 공간 셀 (센서에서 끝점 한 셀 앞까지 ray를 따라 샘플링).
        # 한 셀의 여유를 두지 않으면 거리 노이즈 때문에 벽 셀이 빈 공간으로 깎인다.
        # 최대 거리 근처의 벽은 hit과 no-return이 섞여 나오므로 no-return ray에도 같은 여유를 둔다.
        free_length = np.where(hit, ranges, max_range) - self.resolution
        t = np.arange(0.0, max_range, self.resolution * 0.5)
        mask = valid[:, None] & (t[None, :] < free_length[:, None])
        free_x = sensor_x + cos_a[:, None] * t[None, :]
        free_y = sensor_y + sin_a[:, None] * t[None, :]
        free_flat = self._flat_indices(free_x[mask], free_y[mask])

        # 같은 scan에서 장애물로 관측된 셀은 빈 공간으로 갱신하지 않는다.
        free_flat = np.setdiff1d(free_flat, occ_flat, assume_unique=True)

        flat = self.log_odds.reshape(-1)
        flat[free_flat] = np.clip(flat[free_flat] + self.l_free, self.l_min, self.l_max)
        flat[occ_flat] = np.clip(flat[occ_flat] + self.l_occ, self.l_min, self.l_max)

    def _flat_indices(self, x, y):
        """월드 좌표 배열을 지도 범위 안의 중복 없는 1차원 셀 index로 변환한다."""
        ix, iy = self.world_to_cell(x, y)
        inside = self.in_bounds(ix, iy)
        return np.unique(iy[inside] * self.size + ix[inside])

    def probability(self):
        """각 셀의 occupancy 확률 (0 = free, 1 = occupied)을 반환한다."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def trinary(self):
        """UNKNOWN(-1) / FREE(0) / OCCUPIED(100)으로 분류한 지도를 반환한다."""
        grid = np.full(self.log_odds.shape, UNKNOWN, dtype=np.int8)
        grid[self.log_odds < self.l_free_threshold] = FREE
        grid[self.log_odds > self.l_occ_threshold] = OCCUPIED
        return grid
