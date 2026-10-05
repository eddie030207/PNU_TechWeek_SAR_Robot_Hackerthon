import math


def wrap_angle(angle):
    """각도를 (-π, π] 범위로 정규화한다."""
    return math.atan2(math.sin(angle), math.cos(angle))


class Odometry:
    """Wheel Encoder와 Compass를 이용한 Differential Drive 로봇의 위치 추정.

    이동 거리는 Encoder로 계산하고, 방향은 Compass가 있으면 Compass를 사용한다.
    Compass가 없으면 Encoder의 좌우 바퀴 이동 거리 차이로 방향을 계산한다.
    """

    def __init__(self, x, y, theta, wheel_radius, wheel_separation):
        self.x = x
        self.y = y
        self.theta = wrap_angle(theta)

        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation

        self._prev_left = None
        self._prev_right = None

        # 시작 방향(theta)과 Compass yaw의 차이. 첫 Compass 값에서 한 번만 계산한다.
        self._heading_offset = None

    @property
    def pose(self):
        return self.x, self.y, self.theta

    @property
    def ready(self):
        """Encoder 기준값이 설정되어 위치 추정이 시작되었는지 여부."""
        return self._prev_left is not None

    def update(self, left_position, right_position, compass_yaw=None):
        """Encoder 누적 회전각 [rad]과 Compass yaw [rad]로 pose를 갱신한다."""
        # 센서는 첫 샘플링 전까지 NaN을 반환한다.
        if compass_yaw is not None and math.isnan(compass_yaw):
            compass_yaw = None

        # 로봇이 회전하기 전에 기준을 잡도록 첫 유효 Compass 값에서 바로 계산한다.
        if compass_yaw is not None and self._heading_offset is None:
            self._heading_offset = wrap_angle(self.theta - compass_yaw)

        if math.isnan(left_position) or math.isnan(right_position):
            return self.pose

        if self._prev_left is None:
            self._prev_left = left_position
            self._prev_right = right_position
            return self.pose

        # 바퀴 이동 거리: d = R * Δφ
        d_left = self.wheel_radius * (left_position - self._prev_left)
        d_right = self.wheel_radius * (right_position - self._prev_right)
        self._prev_left = left_position
        self._prev_right = right_position

        # 로봇 이동 거리: Δs = (d_r + d_l) / 2
        delta_s = (d_right + d_left) / 2.0

        if compass_yaw is not None:
            new_theta = wrap_angle(compass_yaw + self._heading_offset)
        else:
            # 로봇 회전 각도: Δθ = (d_r - d_l) / L
            new_theta = wrap_angle(self.theta + (d_right - d_left) / self.wheel_separation)

        # 이동 구간의 중간 방향으로 위치를 적분한다.
        delta_theta = wrap_angle(new_theta - self.theta)
        mid_theta = self.theta + delta_theta / 2.0

        self.x += delta_s * math.cos(mid_theta)
        self.y += delta_s * math.sin(mid_theta)
        self.theta = new_theta

        return self.pose
