#!/usr/bin/env python3

# ============================================================
# IMPORTS
# ============================================================
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

import numpy as np
import math
import csv
import threading
import time
import os
import copy
import traceback

# ROS2 messages
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg

# TF2
import tf2_ros
from tf2_ros import TransformListener, Buffer

# MoveIt2
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.msg import PositionIKRequest, MoveItErrorCodes, RobotState

# Action
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance

# Matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.spatial.transform import Rotation as R
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# CONSTANTS
# ============================================================
JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

PLANNING_GROUP = "arm"
EEF_LINK       = "end_effector_link"
BASE_FRAME     = "base_link"

ACTION_SERVER = "/joint_trajectory_controller/follow_joint_trajectory"
FK_SERVICE    = "/compute_fk"
IK_SERVICE    = "/compute_ik"
CSV_FILENAME  = "trajectory_log_weld_zigzag.csv"
PNG_FILENAME  = "trajectory_result_weld_zigzag.png"

UNIFORM_AMP = 0.025   # 25 mm half-amplitude → 50 mm peak-to-peak


# ============================================================
# UTILITY
# ============================================================
def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg) -> Quaternion:
    roll  = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw   = math.radians(yaw_deg)

    if HAS_SCIPY:
        r = R.from_euler("xyz", [roll, pitch, yaw])
        qx, qy, qz, qw = r.as_quat()
    else:
        cr, cp, cy = math.cos(roll/2), math.cos(pitch/2), math.cos(yaw/2)
        sr, sp, sy = math.sin(roll/2), math.sin(pitch/2), math.sin(yaw/2)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

    q = Quaternion()
    q.x, q.y, q.z, q.w = float(qx), float(qy), float(qz), float(qw)
    return q


def _triangle_wave(u: float) -> float:
    return (2.0 / math.pi) * math.asin(math.sin(2.0 * math.pi * u))


# ============================================================
# WeldZigzagTrajectoryGenerator
# ============================================================
class WeldZigzagTrajectoryGenerator:
    def __init__(
        self,
        x_start=0.25,    y_center=0.00,   z_height=0.25,
        travel_length=0.20,
        amp_start=UNIFORM_AMP, amp_end=UNIFORM_AMP,  
        num_cycles=10,
        num_points=240,
        dwell_peaks=False,
        roll_deg=180.0, pitch_deg=0.0, yaw_deg=90.0,
    ):
        self.x_start       = x_start
        self.y_center      = y_center
        self.z_height      = z_height
        self.travel_length = travel_length
        self.amp_start     = amp_start
        self.amp_end       = amp_end
        self.num_cycles    = num_cycles
        self.num_points    = num_points
        self.dwell_peaks   = dwell_peaks
        self.quaternion    = euler_to_quaternion(roll_deg, pitch_deg, yaw_deg)

    # ── internal ─────────────────────────────────────────────
    def _amplitude(self, s: float) -> float:
        return self.amp_start * (1.0 - s) + self.amp_end * s

    def _sample_xyz(self, n: int) -> list:
        points = []
        for i in range(n):
            s  = i / max(n - 1, 1)
            px = self.x_start + s * self.travel_length
            py = self.y_center + self._amplitude(s) * _triangle_wave(s * self.num_cycles)
            pz = self.z_height
            points.append((px, py, pz))
        return points

    def _sample_with_dwells(self) -> list:
        base = self._sample_xyz(self.num_points)
        if not self.dwell_peaks:
            return base
        augmented = []
        for i, pt in enumerate(base):
            augmented.append(pt)
            s = i / max(self.num_points - 1, 1)
            if abs(_triangle_wave(s * self.num_cycles)) > 0.95:
                augmented.append(pt)
        return augmented

    # ── public ───────────────────────────────────────────────
    def generate(self) -> list:
        samples = self._sample_with_dwells()
        waypoints = []
        for px, py, pz in samples:
            ps = PoseStamped()
            ps.header.frame_id = BASE_FRAME
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.position.z = pz
            ps.pose.orientation = copy.deepcopy(self.quaternion)
            waypoints.append(ps)
        return waypoints

    @property
    def desired_xyz(self) -> np.ndarray:
        return np.array(self._sample_xyz(self.num_points))

    @property
    def plot_limits(self) -> dict:
        pad_x = 0.05
        pad_y = max(0.03, self.amp_start * 0.8)
        pad_z = 0.06
        return {
            "xlim": (self.x_start - pad_x,
                     self.x_start + self.travel_length + pad_x),
            "ylim": (self.y_center - self.amp_start - pad_y,
                     self.y_center + self.amp_start + pad_y),
            "zlim": (self.z_height - pad_z, self.z_height + pad_z),
        }

    @property
    def curve_label(self) -> str:
        mode = "uniform" if abs(self.amp_start - self.amp_end) < 1e-4 else "chirp"
        return (
            f"Weld zigzag ({mode})  "
            f"L={self.travel_length*100:.0f} cm  "
            f"A={self.amp_start*1000:.1f} mm (const)  "
            f"n={self.num_cycles} teeth"
        )


# ============================================================
# WorkspaceChecker
# ============================================================
class WorkspaceChecker:
    def __init__(
        self,
        x_min=0.15, x_max=0.55,
        y_min=-0.30, y_max=0.30,
        z_min=0.10, z_max=0.45,
    ):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        self.rejected_count = 0

    def is_inside_workspace(self, x, y, z) -> bool:
        inside = (
            self.x_min < x < self.x_max and
            self.y_min < y < self.y_max and
            self.z_min < z < self.z_max
        )
        if not inside:
            self.rejected_count += 1
        return inside


# ============================================================
# SingularityChecker
# ============================================================
class SingularityChecker:
    def __init__(self, threshold=0.02):
        self.threshold      = threshold
        self.rejected_count = 0

    def is_near_singularity_from_joints(self, joint_positions: list) -> bool:
        for idx in [1, 3]:
            val = joint_positions[idx]
            if abs(val) < 0.03 or abs(abs(val) - math.pi) < 0.03:
                self.rejected_count += 1
                return True
        return False

    def check_jacobian(self, J: np.ndarray) -> bool:
        JJT = J @ J.T
        m   = math.sqrt(max(np.linalg.det(JJT), 0.0))
        if m < self.threshold:
            self.rejected_count += 1
            return True
        return False


# ============================================================
# IKClient
# ============================================================
class IKClient:
    def __init__(self, node: Node, timeout_sec=2.0):
        self.node           = node
        self.timeout_sec    = timeout_sec
        self.rejected_count = 0

        self._cb_group = MutuallyExclusiveCallbackGroup()
        self._client   = node.create_client(
            GetPositionIK, IK_SERVICE, callback_group=self._cb_group
        )
        node.get_logger().info("Waiting for IK service…")
        if not self._client.wait_for_service(timeout_sec=10.0):
            node.get_logger().warn("IK service not available.")

    def compute_ik(self, pose_stamped: PoseStamped, seed_state=None) -> list:
        req    = GetPositionIK.Request()
        ik_req = PositionIKRequest()

        ik_req.group_name       = PLANNING_GROUP
        ik_req.ik_link_name     = EEF_LINK
        ik_req.pose_stamped     = pose_stamped
        ik_req.timeout          = DurationMsg(sec=int(self.timeout_sec), nanosec=0)
        ik_req.avoid_collisions = False

        rs = RobotState()
        rs.joint_state.name     = JOINT_NAMES
        rs.joint_state.position = seed_state if seed_state else [0.0] * len(JOINT_NAMES)
        ik_req.robot_state      = rs
        req.ik_request          = ik_req

        future   = self._client.call_async(req)
        deadline = time.time() + self.timeout_sec + 0.5
        while not future.done():
            time.sleep(0.02)
            if time.time() > deadline:
                self.node.get_logger().warn("IK timed out.")
                self.rejected_count += 1
                return None

        resp = future.result()
        if resp is None or resp.error_code.val != MoveItErrorCodes.SUCCESS:
            self.rejected_count += 1
            return None

        return list(resp.solution.joint_state.position[: len(JOINT_NAMES)])


# ============================================================
# TrajectoryExecutor
# ============================================================
class TrajectoryExecutor:
    def __init__(self, node: Node, execution_time=20.0):
        self.node           = node
        self.execution_time = execution_time
        self._lock          = threading.Lock()
        self._cb_group      = ReentrantCallbackGroup()

        self._action_client = ActionClient(
            node,
            FollowJointTrajectory,
            ACTION_SERVER,
            callback_group=self._cb_group,
        )
        node.get_logger().info("Waiting for FollowJointTrajectory action server…")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            node.get_logger().warn("Action server not available.")

        self._result      = None
        self._goal_handle = None
        self._done_event  = threading.Event()

    def build_trajectory(self, joint_positions_list: list, execution_time=None) -> JointTrajectory:
        traj             = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.header.frame_id = BASE_FRAME

        n         = len(joint_positions_list)
        exec_time = execution_time if execution_time else self.execution_time

        for i, positions in enumerate(joint_positions_list):
            pt           = JointTrajectoryPoint()
            pt.positions = [float(p) for p in positions]
            t_sec        = exec_time * (i + 1) / n
            pt.time_from_start = DurationMsg(
                sec=int(t_sec),
                nanosec=int((t_sec - int(t_sec)) * 1e9),
            )
            traj.points.append(pt)

        return traj

    def execute(self, trajectory: JointTrajectory) -> bool:
        with self._lock:
            self._done_event.clear()
            self._result = None

            goal            = FollowJointTrajectory.Goal()
            goal.trajectory = trajectory

            for name in JOINT_NAMES:
                tol          = JointTolerance()
                tol.name     = name
                tol.position = 0.05
                tol.velocity = 0.1
                goal.goal_tolerance.append(tol)

            goal.goal_time_tolerance = DurationMsg(sec=5, nanosec=0)

            send_future = self._action_client.send_goal_async(
                goal,
                feedback_callback=self._feedback_callback,
            )
            send_future.add_done_callback(self._goal_response_callback)

            last_pt   = trajectory.points[-1] if trajectory.points else None
            wait_time = (last_pt.time_from_start.sec + 10.0) if last_pt else 30.0
            self._done_event.wait(timeout=wait_time)

            if self._result is None:
                self.node.get_logger().warn("Trajectory execution timed out.")
                return False

            ok = (self._result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
            if not ok:
                self.node.get_logger().warn(
                    f"Trajectory failed, code={self._result.result.error_code}"
                )
            return ok

    def _goal_response_callback(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.node.get_logger().warn("Goal rejected.")
            self._done_event.set()
            return
        self._goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        self._result = future.result()
        self._done_event.set()

    def _feedback_callback(self, _):
        pass


# ============================================================
# Live3DPlotter  — THREE panels:
#   top-left  : XY top-down view
#   top-right : 3-D view
#   bottom    : live tracking error vs time  
# ============================================================
class Live3DPlotter:
    def __init__(self, desired_xyz, lock, actual_xyz_ref, current_pos_ref,
                 error_log_ref,          # ← NEW: list of (t, err_m) tuples
                 plot_limits=None, curve_label="Desired"):
        self.desired_xyz     = desired_xyz
        self.lock            = lock
        self.actual_xyz_ref  = actual_xyz_ref
        self.current_pos_ref = current_pos_ref
        self.error_log_ref   = error_log_ref   # shared reference
        self.plot_limits     = plot_limits or {}
        self.curve_label     = curve_label
        self._thread         = threading.Thread(target=self._run, daemon=True)
        self._stop           = threading.Event()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            plt.ion()
            fig = plt.figure(figsize=(14, 8))
            gs  = gridspec.GridSpec(
                2, 2,
                figure=fig,
                height_ratios=[1.4, 1.0],
                hspace=0.40,
                wspace=0.35,
            )

            lim = self.plot_limits

            # ── Panel 1: XY top-down ─────────────────────────────────
            ax_xy = fig.add_subplot(gs[0, 0])
            ax_xy.plot(
                self.desired_xyz[:, 0], self.desired_xyz[:, 1],
                "b--", linewidth=1.2, label=self.curve_label,
            )
            (actual_xy,)  = ax_xy.plot([], [], "g-",  linewidth=2.0, label="Actual")
            (current_xy,) = ax_xy.plot([], [], "ro",  markersize=7,  label="Current EE")
            ax_xy.set_xlim(*lim.get("xlim", (0.20, 0.50)))
            ax_xy.set_ylim(*lim.get("ylim", (-0.10, 0.10)))
            ax_xy.set_xlabel("X — travel (m)", fontsize=8)
            ax_xy.set_ylabel("Y — weave (m)",  fontsize=8)
            ax_xy.set_title("Top-down view (XY)", fontsize=9)
            ax_xy.legend(loc="upper right", fontsize=7)
            ax_xy.set_aspect("equal", adjustable="box")

            # ── Panel 2: 3-D view ────────────────────────────────────
            ax3d = fig.add_subplot(gs[0, 1], projection="3d")
            ax3d.plot(
                self.desired_xyz[:, 0], self.desired_xyz[:, 1],
                self.desired_xyz[:, 2],
                "b--", linewidth=1.2, label=self.curve_label,
            )
            (actual_3d,)  = ax3d.plot([], [], [], "g-",  linewidth=2.0, label="Actual")
            (current_3d,) = ax3d.plot([], [], [], "ro",  markersize=7,  label="Current EE")
            ax3d.set_xlim(*lim.get("xlim", (0.20, 0.50)))
            ax3d.set_ylim(*lim.get("ylim", (-0.10, 0.10)))
            ax3d.set_zlim(*lim.get("zlim", (0.18, 0.32)))
            ax3d.set_xlabel("X (m)", fontsize=7)
            ax3d.set_ylabel("Y (m)", fontsize=7)
            ax3d.set_zlabel("Z (m)", fontsize=7)
            ax3d.set_title("3-D view", fontsize=9)
            ax3d.legend(loc="upper right", fontsize=7)

            # ── Panel 3: live error vs time ──────────────────────────
            ax_err = fig.add_subplot(gs[1, :])   # spans both columns
            ax_err.set_xlabel("Elapsed time (s)", fontsize=9)
            ax_err.set_ylabel("Tracking error (mm)", fontsize=9)
            ax_err.set_title("Live end-effector tracking error vs time", fontsize=9)
            ax_err.set_xlim(0, 5)        # will auto-expand below
            ax_err.set_ylim(0, 5)        # will auto-expand below
            ax_err.grid(True, linestyle="--", alpha=0.5)
            (err_line,) = ax_err.plot([], [], "r-", linewidth=1.5, label="||pos_desired − pos_actual||")

            # horizontal dashed lines for mean / max annotation (updated live)
            h_mean = ax_err.axhline(y=0, color="orange", linestyle="--",
                                    linewidth=1.0, label="Running mean")
            h_max  = ax_err.axhline(y=0, color="purple",  linestyle=":",
                                    linewidth=1.0, label="Running max")
            ax_err.legend(loc="upper right", fontsize=7)

            fig.suptitle("Gen3 Lite – Constant-Width Weld Zigzag Tracking (Live)", fontsize=11)

            # ── Animation loop ───────────────────────────────────────
            while not self._stop.is_set():
                with self.lock:
                    actual    = list(self.actual_xyz_ref)
                    current   = list(self.current_pos_ref)
                    err_pairs = list(self.error_log_ref)     # [(t, err_m), …]

                # XY update
                if len(actual) > 1:
                    arr = np.array(actual)
                    actual_xy.set_data(arr[:, 0], arr[:, 1])
                    actual_3d.set_data(arr[:, 0], arr[:, 1])
                    actual_3d.set_3d_properties(arr[:, 2])

                if len(current) == 3:
                    current_xy.set_data([current[0]], [current[1]])
                    current_3d.set_data([current[0]], [current[1]])
                    current_3d.set_3d_properties([current[2]])

                # Error vs time update
                if len(err_pairs) > 1:
                    ts  = np.array([p[0] for p in err_pairs])
                    ers = np.array([p[1] * 1000.0 for p in err_pairs])  # → mm
                    err_line.set_data(ts, ers)

                    # Auto-scale axes
                    t_max   = float(ts[-1]) if ts[-1] > 0 else 5.0
                    err_max = float(ers.max()) * 1.15 if ers.max() > 0 else 5.0
                    ax_err.set_xlim(0, max(t_max, 5.0))
                    ax_err.set_ylim(0, max(err_max, 1.0))

                    # Running statistics lines
                    mean_mm = float(ers.mean())
                    max_mm  = float(ers.max())
                    h_mean.set_ydata([mean_mm, mean_mm])
                    h_max.set_ydata([max_mm, max_mm])
                    ax_err.set_title(
                        f"Live tracking error vs time   "
                        f"mean={mean_mm:.2f} mm   max={max_mm:.2f} mm",
                        fontsize=9,
                    )

                fig.canvas.draw_idle()
                plt.pause(0.1)

            plt.savefig(PNG_FILENAME, dpi=150, bbox_inches="tight")
            plt.ioff()
            plt.close(fig)

        except Exception as exc:
            print(f"[Live3DPlotter] {exc}")
            traceback.print_exc()

    def save_final(self):
        """Fallback static save if the live loop exited before saving."""
        if not os.path.exists(PNG_FILENAME):
            try:
                fig = plt.figure(figsize=(14, 8))
                gs  = gridspec.GridSpec(2, 2, figure=fig,
                                        height_ratios=[1.4, 1.0],
                                        hspace=0.40, wspace=0.35)
                lim = self.plot_limits

                ax_xy = fig.add_subplot(gs[0, 0])
                ax_xy.plot(self.desired_xyz[:, 0], self.desired_xyz[:, 1],
                           "b--", label=self.curve_label)
                with self.lock:
                    actual    = list(self.actual_xyz_ref)
                    err_pairs = list(self.error_log_ref)
                if len(actual) > 1:
                    arr = np.array(actual)
                    ax_xy.plot(arr[:, 0], arr[:, 1], "g-", label="Actual")
                ax_xy.set_xlim(*lim.get("xlim", (0.20, 0.50)))
                ax_xy.set_ylim(*lim.get("ylim", (-0.10, 0.10)))
                ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
                ax_xy.set_title("Top-down (XY)")
                ax_xy.set_aspect("equal", adjustable="box")
                ax_xy.legend()

                ax3d = fig.add_subplot(gs[0, 1], projection="3d")
                ax3d.plot(self.desired_xyz[:, 0], self.desired_xyz[:, 1],
                          self.desired_xyz[:, 2], "b--", label=self.curve_label)
                if len(actual) > 1:
                    arr = np.array(actual)
                    ax3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], "g-", label="Actual")
                ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
                ax3d.set_title("3-D view"); ax3d.legend()

                ax_err = fig.add_subplot(gs[1, :])
                if len(err_pairs) > 1:
                    ts  = np.array([p[0] for p in err_pairs])
                    ers = np.array([p[1] * 1000.0 for p in err_pairs])
                    ax_err.plot(ts, ers, "r-", linewidth=1.5,
                                label="Tracking error")
                    ax_err.axhline(ers.mean(), color="orange",
                                   linestyle="--", label=f"Mean {ers.mean():.2f} mm")
                    ax_err.axhline(ers.max(), color="purple",
                                   linestyle=":", label=f"Max {ers.max():.2f} mm")
                ax_err.set_xlabel("Elapsed time (s)")
                ax_err.set_ylabel("Error (mm)")
                ax_err.set_title("End-effector tracking error vs time")
                ax_err.legend(fontsize=7)
                ax_err.grid(True, linestyle="--", alpha=0.5)

                fig.suptitle("Gen3 Lite – Constant-Width Weld Zigzag Tracking Result")
                plt.savefig(PNG_FILENAME, dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                print(f"[Live3DPlotter] Could not save PNG: {exc}")


# ============================================================
# MetricsCalculator
# ============================================================
class MetricsCalculator:
    def __init__(self):
        self.errors:          list  = []
        self.execution_time:  float = 0.0
        self.valid_waypoints: int   = 0
        self.rejected_ws:     int   = 0
        self.rejected_sing:   int   = 0
        self.rejected_ik:     int   = 0

    def add_error(self, desired, actual):
        if len(desired) == 3 and len(actual) == 3:
            self.errors.append(math.sqrt(sum((d-a)**2 for d, a in zip(desired, actual))))

    def mean_error(self): return float(np.mean(self.errors))  if self.errors else 0.0
    def max_error(self):  return float(np.max(self.errors))   if self.errors else 0.0
    def rms_error(self):  return float(np.sqrt(np.mean(np.square(self.errors)))) if self.errors else 0.0

    def print_summary(self):
        print("\n" + "=" * 60)
        print("          TRAJECTORY TRACKING METRICS SUMMARY")
        print("=" * 60)
        print(f"  Valid Waypoints          : {self.valid_waypoints}")
        print(f"  Rejected (Workspace)     : {self.rejected_ws}")
        print(f"  Rejected (Singularity)   : {self.rejected_sing}")
        print(f"  Rejected (IK Failure)    : {self.rejected_ik}")
        print(f"  Execution Time           : {self.execution_time:.2f} s")
        print("-" * 60)
        if self.errors:
            print(f"  Mean Tracking Error      : {self.mean_error()*1000:.3f} mm")
            print(f"  Max  Tracking Error      : {self.max_error()*1000:.3f} mm")
            print(f"  RMS  Tracking Error      : {self.rms_error()*1000:.3f} mm")
        else:
            print("  No tracking error data (FK/TF unavailable).")
        print("=" * 60 + "\n")


# ============================================================
# Gen3WeldZigzagTrackingNode
# ============================================================
class Gen3WeldZigzagTrackingNode(Node):

    def __init__(self):
        super().__init__("gen3_weld_zigzag_tracking_node")

        # ── ROS parameters ───────────────────────────────────
        self.declare_parameter("x_start",                  0.25)
        self.declare_parameter("y_center",                  0.00)
        self.declare_parameter("z_height",                  0.25)
        self.declare_parameter("travel_length",             0.20)
        self.declare_parameter("amp_start",                 UNIFORM_AMP)   # 0.025 m
        self.declare_parameter("amp_end",                   UNIFORM_AMP)   # 0.025 m ← same
        self.declare_parameter("num_cycles",                10)
        self.declare_parameter("num_points",                240)
        self.declare_parameter("dwell_peaks",               False)
        self.declare_parameter("execution_time",            25.0)
        self.declare_parameter("manipulability_threshold",  0.02)

        x_start       = self.get_parameter("x_start").value
        y_center      = self.get_parameter("y_center").value
        z_height      = self.get_parameter("z_height").value
        travel_length = self.get_parameter("travel_length").value
        amp_start     = self.get_parameter("amp_start").value
        amp_end       = self.get_parameter("amp_end").value
        num_cycles    = self.get_parameter("num_cycles").value
        num_points    = self.get_parameter("num_points").value
        dwell_peaks   = self.get_parameter("dwell_peaks").value
        self._exec_time  = self.get_parameter("execution_time").value
        self._manip_thr  = self.get_parameter("manipulability_threshold").value

        self.get_logger().info(
            f"Weld zigzag parameters: "
            f"x=[{x_start:.3f}, {x_start+travel_length:.3f}] m  "
            f"y_center={y_center:.3f} m  z={z_height:.3f} m  "
            f"amp={amp_start*1000:.1f} mm (constant)  "
            f"teeth={num_cycles}  n={num_points}  "
            f"dwell_peaks={dwell_peaks}  t_exec={self._exec_time}s"
        )

        # ── Shared state ─────────────────────────────────────
        self._lock           = threading.Lock()
        self._actual_traj:   list = []
        self._current_pos:   list = []
        # NEW: live error log — list of (elapsed_time_s, error_m) tuples
        self._error_log:     list = []
        self._exec_start_time: float = 0.0   # set when trajectory starts

        # ── Components ───────────────────────────────────────
        self._gen = WeldZigzagTrajectoryGenerator(
            x_start=x_start,
            y_center=y_center,
            z_height=z_height,
            travel_length=travel_length,
            amp_start=amp_start,
            amp_end=amp_end,
            num_cycles=num_cycles,
            num_points=num_points,
            dwell_peaks=dwell_peaks,
        )
        self._ws_checker   = WorkspaceChecker()
        self._sing_checker = SingularityChecker(threshold=self._manip_thr)
        self._metrics      = MetricsCalculator()

        # TF2
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Joint states
        self._joint_positions: list = [0.0] * len(JOINT_NAMES)
        self._js_lock = threading.Lock()
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)

        # FK client
        self._fk_client = self.create_client(GetPositionFK, FK_SERVICE)

        # ── Plotter (now receives error_log_ref) ─────────────
        self._plotter = Live3DPlotter(
            desired_xyz=self._gen.desired_xyz,
            lock=self._lock,
            actual_xyz_ref=self._actual_traj,
            current_pos_ref=self._current_pos,
            error_log_ref=self._error_log,      # ← pass live error list
            plot_limits=self._gen.plot_limits,
            curve_label=self._gen.curve_label,
        )

        # CSV
        self._csv_file   = None
        self._csv_writer = None
        self._init_csv()

        self._ik_client    = None
        self._executor_obj = None

        # desired_xyz for nearest-neighbour error lookup during TF callback
        self._desired_xyz_arr = self._gen.desired_xyz   # shape (N, 3)
        self._tracking_active = False                    # gate for error log

        self.get_logger().info("Gen3WeldZigzagTrackingNode initialised (waiting for late_init).")

    def late_init(self):
        self._ik_client    = IKClient(self, timeout_sec=2.0)
        self._executor_obj = TrajectoryExecutor(self, execution_time=self._exec_time)
        self.get_logger().info("late_init complete — IK client and action client ready.")

    # ── CSV ─────────────────────────────────────────────────
    def _init_csv(self):
        self._csv_file   = open(CSV_FILENAME, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "time", "desired_x", "desired_y", "desired_z",
            "actual_x", "actual_y", "actual_z", "error_m",
        ])

    def _log_csv(self, t, desired, actual, error):
        if self._csv_writer:
            self._csv_writer.writerow([
                f"{t:.4f}",
                f"{desired[0]:.6f}", f"{desired[1]:.6f}", f"{desired[2]:.6f}",
                f"{actual[0]:.6f}",  f"{actual[1]:.6f}",  f"{actual[2]:.6f}",
                f"{error:.6f}",
            ])

    def _close_csv(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
            self.get_logger().info(f"CSV saved to {CSV_FILENAME}")

    # ── Joint states & TF ───────────────────────────────────
    def _joint_state_callback(self, msg: JointState):
        positions = {name: pos for name, pos in zip(msg.name, msg.position)}
        with self._js_lock:
            for i, jname in enumerate(JOINT_NAMES):
                if jname in positions:
                    self._joint_positions[i] = positions[jname]
        self._update_ee_from_tf()

    def _update_ee_from_tf(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EEF_LINK, Time(), timeout=Duration(seconds=0.05)
            )
            t = tf.transform.translation
            ee = [t.x, t.y, t.z]

            with self._lock:
                self._current_pos.clear()
                self._current_pos.extend(ee)

                # Live error computation during trajectory execution
                if self._tracking_active and self._exec_start_time > 0.0:
                    elapsed = time.time() - self._exec_start_time

                    # Find nearest desired waypoint in X (fast proxy)
                    dx = self._desired_xyz_arr[:, 0] - ee[0]
                    idx = int(np.argmin(np.abs(dx)))
                    des = self._desired_xyz_arr[idx]
                    err = math.sqrt(
                        (des[0]-ee[0])**2 +
                        (des[1]-ee[1])**2 +
                        (des[2]-ee[2])**2
                    )
                    self._error_log.append((elapsed, err))

        except Exception:
            pass

    def _get_ee_from_fk(self, joint_positions: list) -> list:
        if not self._fk_client.service_is_ready():
            return []

        req = GetPositionFK.Request()
        req.header.frame_id                  = BASE_FRAME
        req.fk_link_names                    = [EEF_LINK]
        req.robot_state.joint_state.name     = JOINT_NAMES
        req.robot_state.joint_state.position = [float(p) for p in joint_positions]

        future   = self._fk_client.call_async(req)
        deadline = time.time() + 1.0
        while not future.done():
            time.sleep(0.02)
            if time.time() > deadline:
                return []

        resp = future.result()
        if resp and resp.error_code.val == MoveItErrorCodes.SUCCESS and resp.pose_stamped:
            p = resp.pose_stamped[0].pose.position
            return [p.x, p.y, p.z]
        return []

    # ── Main pipeline ────────────────────────────────────────
    def run(self):
        if self._ik_client is None or self._executor_obj is None:
            self.get_logger().error("late_init() was not called. Aborting.")
            return

        self.get_logger().info(f"Generating waypoints — {self._gen.curve_label}…")
        waypoints = self._gen.generate()

        self.get_logger().info(f"Solving IK for {len(waypoints)} waypoints…")
        valid_joint_positions = []
        valid_desired_xyz     = []

        with self._js_lock:
            seed = list(self._joint_positions)

        for i, ps in enumerate(waypoints):
            x, y, z = ps.pose.position.x, ps.pose.position.y, ps.pose.position.z

            if not self._ws_checker.is_inside_workspace(x, y, z):
                self.get_logger().debug(f"Waypoint {i} rejected (workspace).")
                continue

            joint_pos = self._ik_client.compute_ik(ps, seed_state=seed)
            if joint_pos is None:
                self.get_logger().debug(f"Waypoint {i} rejected (IK failed).")
                continue

            if self._sing_checker.is_near_singularity_from_joints(joint_pos):
                self.get_logger().debug(f"Waypoint {i} rejected (singularity).")
                continue

            valid_joint_positions.append(joint_pos)
            valid_desired_xyz.append([x, y, z])
            seed = joint_pos

        n_valid = len(valid_joint_positions)
        self.get_logger().info(
            f"Valid waypoints: {n_valid}/{len(waypoints)} "
            f"(ws={self._ws_checker.rejected_count}, "
            f"sing={self._sing_checker.rejected_count}, "
            f"ik={self._ik_client.rejected_count})"
        )

        if n_valid == 0:
            self.get_logger().error("No valid waypoints. Aborting.")
            return

        with self._lock:
            self._actual_traj.clear()
            self._error_log.clear()

        # Phase 2: Move to start
        self.get_logger().info("Moving to weld start position (5 s)…")
        start_traj = self._executor_obj.build_trajectory(
            [valid_joint_positions[0]], execution_time=5.0
        )
        ok = self._executor_obj.execute(start_traj)
        if ok:
            self.get_logger().info("Reached weld start position.")
        else:
            self.get_logger().warn("Move-to-start incomplete; proceeding anyway.")
        time.sleep(1.0)

        # Phase 3: Execute weld zigzag (enable live error logging)
        traj = self._executor_obj.build_trajectory(valid_joint_positions)
        self.get_logger().info("Executing constant-width weld zigzag trajectory…")
        with self._lock:
            self._exec_start_time = time.time()
            self._tracking_active = True

        t_start = time.time()
        success = self._executor_obj.execute(traj)
        t_end   = time.time()

        with self._lock:
            self._tracking_active = False

        self._metrics.execution_time  = t_end - t_start
        self._metrics.valid_waypoints = n_valid
        self._metrics.rejected_ws     = self._ws_checker.rejected_count
        self._metrics.rejected_sing   = self._sing_checker.rejected_count
        self._metrics.rejected_ik     = self._ik_client.rejected_count

        self.get_logger().info(
            f"Execution {'succeeded' if success else 'failed/timed out'} "
            f"in {self._metrics.execution_time:.2f}s"
        )

        # Phase 4: FK for post-hoc error metrics and CSV
        self.get_logger().info("Computing FK for actual positions…")
        for i, joint_pos in enumerate(valid_joint_positions):
            t_rel   = self._metrics.execution_time * (i + 1) / n_valid
            desired = valid_desired_xyz[i]

            actual = self._get_ee_from_fk(joint_pos)
            if not actual:
                with self._lock:
                    actual = list(self._current_pos) if self._current_pos else desired

            err = math.sqrt(sum((d-a)**2 for d, a in zip(desired, actual)))
            self._metrics.add_error(desired, actual)
            self._log_csv(t_rel, desired, actual, err)

            with self._lock:
                self._actual_traj.append(actual)

        # Phase 5: Finalize
        self._close_csv()
        self._metrics.print_summary()
        self._plotter.stop()
        time.sleep(0.5)
        self._plotter.save_final()
        self.get_logger().info(f"Plot saved to {PNG_FILENAME}")


# ============================================================
# main()
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = Gen3WeldZigzagTrackingNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(0.5)
    node.late_init()

    node._plotter.start()

    try:
        time.sleep(2.0)
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    except Exception as exc:
        node.get_logger().error(f"Unhandled exception: {exc}")
        traceback.print_exc()
    finally:
        node._close_csv()
        node._plotter.stop()
        executor.shutdown()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)
        print("Node shut down cleanly.")


if __name__ == "__main__":
    main()