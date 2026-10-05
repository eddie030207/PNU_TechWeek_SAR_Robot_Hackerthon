"""sar_mission 검증용 심판 컨트롤러 (검증 전용, 제출물 아님).

별도의 Supervisor 로봇으로 실행되어 임무 컨트롤러는 실제 조건 그대로 두고 밖에서 측정한다.
측정: 위치 추정 오차, 충돌(바닥 외 접촉), 사람과의 최소 거리, 사과 접근 거리, 복귀 오차, 소요 시간.
"""
import json
import math
import os
import sys

from controller import Supervisor
import numpy as np

TIMEOUT = float(sys.argv[1]) if len(sys.argv) >= 2 else 1200.0
TAG = sys.argv[2] if len(sys.argv) >= 3 else "run"

STATES = {"SPIN": 0, "EXPLORE": 1, "LOOK": 2, "APPROACH": 3, "CONFIRM": 4, "RETURN": 5, "DONE": 6}

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
tb3 = robot.getFromDef("TB3")
ped = robot.getFromDef("PED")
tb3.enableContactPointsTracking(timestep, True)
custom = tb3.getField("customData")

apples = {}
children = robot.getRoot().getField("children")
for i in range(children.getCount()):
    node = children.getMFNode(i)
    if node.getTypeName().endswith("Apple") and node.getTypeName() != "Apple":
        apples[f"{node.getTypeName()}@{i}"] = node

robot.step(timestep)
start = tb3.getPosition()[:2]
start_apples = {name: node.getPosition()[:2] for name, node in apples.items()}
min_apple = {name: float("inf") for name in apples}
trace = []
collisions = []
min_ped = float("inf")
loc_errors = []
done_time = None
last = {}
truth_at = {}
last_est_time = None
sync_error = float("nan")

while robot.step(timestep) != -1:
    t = robot.getTime()
    p = tb3.getPosition()
    R = tb3.getOrientation()
    yaw = math.atan2(R[3], R[0])
    pp = ped.getPosition() if ped is not None else [float("nan")] * 3
    d_ped = math.hypot(p[0] - pp[0], p[1] - pp[1])
    min_ped = min(min_ped, d_ped)
    for name, node in apples.items():
        a = node.getPosition()
        min_apple[name] = min(min_apple[name], math.hypot(p[0] - a[0], p[1] - a[1]))

    body = [c.point for c in tb3.getContactPoints(True) if c.point[2] > 0.012]
    if body and (not collisions or t - collisions[-1]["t_end"] > 1.0):
        collisions.append({"t": round(t, 2), "t_end": t, "x": round(body[0][0], 2), "y": round(body[0][1], 2), "z": round(body[0][2], 3)})
    elif body:
        collisions[-1]["t_end"] = t

    truth_at[round(t, 2)] = (p[0], p[1])
    est = [float("nan")] * 3
    try:
        last = json.loads(custom.getSFString())
        est = [last["x"], last["y"], last["theta"]]
        # 추정값이 기록된 시각의 실제 위치와 비교한다 (customData는 0.25초마다 갱신된다).
        if last["t"] != last_est_time and last["t"] in truth_at:
            last_est_time = last["t"]
            tx, ty = truth_at[last["t"]]
            loc_errors.append(math.hypot(est[0] - tx, est[1] - ty))
            sync_error = loc_errors[-1]
    except (ValueError, KeyError):
        pass
    vel = tb3.getVelocity()
    trace.append([t, p[0], p[1], yaw, est[0], est[1], est[2], pp[0], pp[1], d_ped, len(body), p[2], R[8], vel[0], sync_error,
                  last.get('pd', float('nan')), STATES.get(last.get('state'), -1), {None: 0, 'YIELD': 1, 'EVADE': 2}.get(last.get('person'), -1)])

    if last.get("state") == "DONE" and done_time is None:
        done_time = t
    if (done_time is not None and t - done_time > 2.0) or t >= TIMEOUT:
        break

p = tb3.getPosition()
moved = {name: round(math.hypot(node.getPosition()[0] - start_apples[name][0], node.getPosition()[1] - start_apples[name][1]), 3)
         for name, node in apples.items()}
result = {
    "finished": done_time is not None,
    "mission_time": round(done_time, 1) if done_time is not None else None,
    "sim_time": round(robot.getTime(), 1),
    "final_state": last,
    "return_error_true": round(math.hypot(p[0] - start[0], p[1] - start[1]), 3),
    "localization_error": {"mean": round(float(np.mean(loc_errors)), 3), "max": round(float(np.max(loc_errors)), 3)} if loc_errors else None,
    "min_distance_to_person": round(min_ped, 3),
    "collisions": [{k: (round(v, 2) if k == "t_end" else v) for k, v in c.items()} for c in collisions],
    "min_distance_to_apples": {name: round(d, 3) for name, d in min_apple.items()},
    "apple_displacement": moved,
}
folder = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(folder, f"referee_{TAG}.json"), "w") as f:
    json.dump(result, f, indent=2)
np.save(os.path.join(folder, f"trace_{TAG}.npy"), np.array(trace))
print(json.dumps(result, indent=2), flush=True)
robot.simulationQuit(0)
