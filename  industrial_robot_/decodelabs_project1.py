"""
DecodeLabs Project 1: Robotic Arm Kinematics & Path Planning
Single-file version

IPO Pipeline:
  INPUT    -> Target poses (Point A, Point B)
  PROCESS  -> Forward/Inverse Kinematics, Quintic Spline, Collision Check
  OUTPUT   -> PASS/FAIL verdict + smooth joint trajectory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# =============================================================================
# CONFIGURATION
# =============================================================================

POINT_A = (0.35, 0.15, 0.55)   # start XYZ (meters)
POINT_B = (0.35, -0.15, 0.55)  # end XYZ (meters)
TRAJECTORY_DURATION = 4.0      # seconds
TRAJECTORY_DT = 0.05           # seconds


@dataclass
class DHLink:
    a: float
    alpha: float
    d: float
    theta: float


# PUMA-560 style 6-DOF arm (meters / radians)
DEFAULT_DH: List[DHLink] = [
    DHLink(0.0, np.pi / 2, 0.6718, 0.0),
    DHLink(0.4318, 0.0, 0.0, 0.0),
    DHLink(0.0203, -np.pi / 2, 0.1500, 0.0),
    DHLink(0.0, np.pi / 2, 0.4318, 0.0),
    DHLink(0.0, -np.pi / 2, 0.0, 0.0),
    DHLink(0.0, 0.0, 0.0, 0.0),
]

JOINT_LIMITS = np.array([
    [-np.pi, np.pi],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
])


@dataclass
class BoxObstacle:
    center: np.ndarray
    half_size: np.ndarray


OBSTACLES = [
    BoxObstacle(center=np.array([0.38, 0.0, 0.50]), half_size=np.array([0.05, 0.05, 0.08])),
]


# =============================================================================
# KINEMATICS
# =============================================================================

def dh_transform(link: DHLink, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(link.alpha), np.sin(link.alpha)
    return np.array([
        [ct, -st * ca, st * sa, link.a * ct],
        [st, ct * ca, -ct * sa, link.a * st],
        [0.0, sa, ca, link.d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def forward_kinematics(
    joints: np.ndarray,
    dh_table: List[DHLink] = DEFAULT_DH,
) -> np.ndarray:
    t = np.eye(4)
    for i, link in enumerate(dh_table):
        t = t @ dh_transform(link, joints[i] + link.theta)
    return t


def get_link_positions(
    joints: np.ndarray,
    dh_table: List[DHLink] = DEFAULT_DH,
) -> np.ndarray:
    positions = [np.array([0.0, 0.0, 0.0])]
    t = np.eye(4)
    for i, link in enumerate(dh_table):
        t = t @ dh_transform(link, joints[i] + link.theta)
        positions.append(t[:3, 3].copy())
    return np.array(positions)


def make_target_pose(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = [x, y, z]
    return pose


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    pos_err = target[:3, 3] - current[:3, 3]
    r_err = current[:3, :3].T @ target[:3, :3]
    angle = np.arccos(np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        rot_err = np.zeros(3)
    else:
        axis = np.array([
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ])
        axis = axis / (2.0 * np.sin(angle))
        rot_err = axis * angle
    return np.concatenate([pos_err, rot_err])


def numerical_jacobian(
    joints: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    j = np.zeros((6, 6))
    base_err = pose_error(forward_kinematics(joints), target)
    for i in range(6):
        q_plus = joints.copy()
        q_plus[i] += eps
        err_plus = pose_error(forward_kinematics(q_plus), target)
        j[:, i] = (err_plus - base_err) / eps
    return j


def clamp_joints(joints: np.ndarray) -> np.ndarray:
    return np.clip(joints, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def inverse_kinematics(
    target_pose: np.ndarray,
    seed: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    damping: float = 0.05,
) -> Tuple[np.ndarray, bool, int]:
    q = np.zeros(6) if seed is None else seed.copy()
    q = clamp_joints(q)

    for iteration in range(1, max_iter + 1):
        current = forward_kinematics(q)
        err = pose_error(current, target_pose)
        if np.linalg.norm(err) < tol:
            return q, True, iteration

        j = numerical_jacobian(q, target_pose)
        jj_t = j @ j.T
        dq = j.T @ np.linalg.solve(jj_t + damping**2 * np.eye(6), err)
        q = clamp_joints(q + dq)

    return q, False, max_iter


# =============================================================================
# TRAJECTORY (QUINTIC SPLINE)
# =============================================================================

def quintic_coefficients(q0: float, qf: float, duration: float) -> np.ndarray:
    return np.array([
        q0,
        0.0,
        0.0,
        10.0 * (qf - q0) / duration**3,
        -15.0 * (qf - q0) / duration**4,
        6.0 * (qf - q0) / duration**5,
    ])


def evaluate_quintic(coeffs: np.ndarray, t: float) -> Tuple[float, float, float]:
    a0, a1, a2, a3, a4, a5 = coeffs
    pos = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    vel = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
    acc = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
    return pos, vel, acc


def generate_joint_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float = TRAJECTORY_DURATION,
    dt: float = TRAJECTORY_DT,
) -> dict[str, np.ndarray]:
    n_joints = len(q_start)
    times = np.arange(0.0, duration + dt, dt)
    coeffs = [quintic_coefficients(q_start[j], q_end[j], duration) for j in range(n_joints)]

    positions = np.zeros((len(times), n_joints))
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)

    for i, t in enumerate(times):
        t_clamped = min(t, duration)
        for j in range(n_joints):
            p, v, a = evaluate_quintic(coeffs[j], t_clamped)
            positions[i, j] = p
            velocities[i, j] = v
            accelerations[i, j] = a

    return {
        "time": times,
        "position": positions,
        "velocity": velocities,
        "acceleration": accelerations,
    }


# =============================================================================
# COLLISION CHECK
# =============================================================================

def point_in_box(point: np.ndarray, box: BoxObstacle) -> bool:
    d = np.abs(point - box.center)
    return np.all(d <= box.half_size + 1e-6)


def segment_box_collision(
    p1: np.ndarray,
    p2: np.ndarray,
    box: BoxObstacle,
    samples: int = 20,
) -> bool:
    for alpha in np.linspace(0.0, 1.0, samples):
        p = (1.0 - alpha) * p1 + alpha * p2
        if point_in_box(p, box):
            return True
    return False


def trajectory_collision(
    link_positions_over_time: np.ndarray,
    obstacles: List[BoxObstacle],
) -> Tuple[bool, int]:
    for t_idx in range(link_positions_over_time.shape[0]):
        links = link_positions_over_time[t_idx]
        for i in range(len(links) - 1):
            p1, p2 = links[i], links[i + 1]
            for obs in obstacles:
                if segment_box_collision(p1, p2, obs):
                    return True, t_idx
    return False, -1


# =============================================================================
# PATH PLANNER (POINT A -> POINT B)
# =============================================================================

def interpolate_poses(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    steps: int,
) -> List[np.ndarray]:
    poses = []
    for alpha in np.linspace(0.0, 1.0, steps):
        p = (1.0 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = pose_b[:3, :3]
        pose[:3, 3] = p
        poses.append(pose)
    return poses


def plan_path_a_to_b(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
    obstacles: List[BoxObstacle],
    duration: float = TRAJECTORY_DURATION,
    waypoint_steps: int = 8,
) -> dict:
    pose_a = make_target_pose(*point_a, roll=0.0, pitch=np.pi / 2, yaw=0.0)
    pose_b = make_target_pose(*point_b, roll=0.0, pitch=np.pi / 2, yaw=0.0)

    q_seed, ok_a, _ = inverse_kinematics(pose_a)
    if not ok_a:
        return {"success": False, "reason": "IK failed for Point A"}

    q_goal, ok_b, _ = inverse_kinematics(pose_b, seed=q_seed)
    if not ok_b:
        return {"success": False, "reason": "IK failed for Point B"}

    waypoints = interpolate_poses(pose_a, pose_b, waypoint_steps)
    for wp in waypoints[1:-1]:
        _, ok, _ = inverse_kinematics(wp, seed=q_seed)
        if not ok:
            return {"success": False, "reason": "IK failed for intermediate waypoint"}

    traj = generate_joint_trajectory(q_seed, q_goal, duration=duration, dt=TRAJECTORY_DT)

    link_history = np.array([get_link_positions(q) for q in traj["position"]])
    collided, t_hit = trajectory_collision(link_history, obstacles)

    if collided:
        return {
            "success": False,
            "reason": f"Collision detected at t={traj['time'][t_hit]:.2f}s",
            "trajectory": traj,
            "q_start": q_seed,
            "q_end": q_goal,
            "link_history": link_history,
            "pose_a": pose_a,
            "pose_b": pose_b,
        }

    return {
        "success": True,
        "reason": "PASS — collision-free trajectory",
        "trajectory": traj,
        "q_start": q_seed,
        "q_end": q_goal,
        "link_history": link_history,
        "pose_a": pose_a,
        "pose_b": pose_b,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_box(ax, box: BoxObstacle, color: str = "red", alpha: float = 0.3) -> None:
    c = box.center
    h = box.half_size
    x = [c[0] - h[0], c[0] + h[0]]
    y = [c[1] - h[1], c[1] + h[1]]
    z = [c[2] - h[2], c[2] + h[2]]

    for xi in x:
        for yi in y:
            ax.plot([xi, xi], [yi, yi], [z[0], z[1]], color=color, alpha=alpha)
    for xi in x:
        for zi in z:
            ax.plot([xi, xi], [y[0], y[1]], [zi, zi], color=color, alpha=alpha)
    for yi in y:
        for zi in z:
            ax.plot([x[0], x[1]], [yi, yi], [zi, zi], color=color, alpha=alpha)


def animate_arm(
    link_history: np.ndarray,
    obstacles: List[BoxObstacle],
    title: str = "DecodeLabs Project 1 — Point A to Point B",
) -> FuncAnimation:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    for obs in obstacles:
        draw_box(ax, obs)

    line, = ax.plot([], [], [], "o-", lw=3, color="#00b4d8", markersize=6)
    path_line, = ax.plot([], [], [], "--", color="#ffd600", alpha=0.6)

    ee_path = link_history[:, -1, :]
    path_line.set_data(ee_path[:, 0], ee_path[:, 1])
    path_line.set_3d_properties(ee_path[:, 2])

    all_pts = link_history.reshape(-1, 3)
    margin = 0.2
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    ax.set_zlim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    def update(frame_idx: int):
        pts = link_history[frame_idx]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        return line, path_line

    anim = FuncAnimation(fig, update, frames=len(link_history), interval=50, blit=False)
    plt.tight_layout()
    plt.show()
    return anim


def print_report(result: dict) -> None:
    print("=" * 60)
    print("  DecodeLabs Project 1 — Kinematics & Path Planning")
    print("=" * 60)
    print(f"Point A: {POINT_A}")
    print(f"Point B: {POINT_B}")
    print(f"Obstacles: {len(OBSTACLES)}")

    if "q_start" in result:
        print(f"\nStart joints (rad): {result['q_start'].round(3)}")
        print(f"End joints (rad):   {result['q_end'].round(3)}")

    if "trajectory" in result:
        print(f"Trajectory samples: {len(result['trajectory']['time'])}")

    verdict = "PASS (0)" if result["success"] else "FAIL (1)"
    print(f"PLC Signal:         {verdict}")
    print(f"Status:             {result['reason']}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    result = plan_path_a_to_b(POINT_A, POINT_B, OBSTACLES, duration=TRAJECTORY_DURATION)
    print_report(result)

    if "link_history" not in result:
        return

    animate_arm(result["link_history"], OBSTACLES)


if __name__ == "__main__":
    main()