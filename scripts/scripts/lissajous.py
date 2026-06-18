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
CSV_FILENAME  = "trajectory_log_lissajous.csv"
PNG_FILENAME  = "trajectory_result_lissajous.png"


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


# ============================================================
# Lissajous3DTrajectoryGenerator
# ============================================================
class Lissajous3DTrajectoryGenerator:
    def __init__(
        self,
        center_x=0.35, center_y=0.00, center_z=0.25,
        amp_x=0.06,    amp_y=0.06,    amp_z=0.04,
        freq_x=3,      freq_y=2,      freq_z=1,
        phase_x=0.0,   phase_y=math.pi/2, phase_z=math.pi/4,
        num_points=200,
        roll_deg=180.0, pitch_deg=0.0, yaw_deg=90.0,
    ):
        self.center_x  = center_x
        self.center_y  = center_y
        self.center_z  = center_z
        self.amp_x     = amp_x
        self.amp_y     = amp_y
        self.amp_z     = amp_z
        self.freq_x    = freq_x
        self.freq_y    = freq_y
        self.freq_z    = freq_z
        self.phase_x   = phase_x
        self.phase_y   = phase_y
        self.phase_z   = phase_z
        self.num_points = num_points
        self.quaternion = euler_to_quaternion(roll_deg, pitch_deg, yaw_deg)

    def _sample_xyz(self, n: int) -> list:
        points = []
        for i in range(n):
            t  = 2.0 * math.pi * i / n
            px = self.center_x + self.amp_x * math.sin(self.freq_x * t + self.phase_x)
            py = self.center_y + self.amp_y * math.sin(self.freq_y * t + self.phase_y)
            pz = self.center_z + self.amp_z * math.sin(self.freq_z * t + self.phase_z)
            points.append((px, py, pz))
        return points

    def generate(self) -> list:
        waypoints = []
        for px, py, pz in self._sample_xyz(self.num_points):
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
        pad_x = max(0.05, self.amp_x * 0.5)
        pad_y = max(0.05, self.amp_y * 0.5)
        pad_z = max(0.05, self.amp_z * 1.0)
        return {
            "xlim": (self.center_x - self.amp_x - pad_x,
                     self.center_x + self.amp_x + pad_x),
            "ylim": (self.center_y - self.amp_y - pad_y,
                     self.center_y + self.amp_y + pad_y),
            "zlim": (self.center_z - self.amp_z - pad_z,
                     self.center_z + self.amp_z + pad_z),
        }

    @property
    def curve_label(self) -> str:
        return (
            f"Lissajous ({self.freq_x}:{self.freq_y}:{self.freq_z})  "
            f"A=({self.amp_x*100:.0f},{self.amp_y*100:.0f},{self.amp_z*100:.0f}) cm"
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
# MetricsCalculator
# ============================================================
class MetricsCalculator:

    def __init__(self):
        self._lock = threading.Lock()

        # Live errors (Phase 3): sampled from TF during execution
        self.live_errors: list      = []
        self.live_error_times: list = []

       
        self.errors: list      = []
        self.error_times: list = []

        self.execution_time  = 0.0
        self.valid_waypoints = 0
        self.rejected_ws     = 0
        self.rejected_sing   = 0
        self.rejected_ik     = 0

    # ── Live (Phase 3) ──────────────────────────────────────
    def add_live_error(self, desired, actual, t: float):
        """Thread-safe. Called by LiveErrorPoller during Phase 3."""
        if len(desired) == 3 and len(actual) == 3:
            err = math.sqrt(
                (desired[0] - actual[0])**2 +
                (desired[1] - actual[1])**2 +
                (desired[2] - actual[2])**2
            )
            with self._lock:
                self.live_errors.append(err)
                self.live_error_times.append(t)

    def get_live_snapshot(self):
        with self._lock:
            return list(self.live_error_times), list(self.live_errors)

    # ── FK / canonical (Phase 4) ────────────────────────────
    def add_error(self, desired, actual, t: float):
        if len(desired) == 3 and len(actual) == 3:
            err = math.sqrt(
                (desired[0] - actual[0])**2 +
                (desired[1] - actual[1])**2 +
                (desired[2] - actual[2])**2
            )
            self.errors.append(err)
            self.error_times.append(t)

    # ── Statistics ───────────────────────────────────────────
    def mean_error(self):
        return float(np.mean(self.errors)) if self.errors else 0.0

    def max_error(self):
        return float(np.max(self.errors)) if self.errors else 0.0

    def rms_error(self):
        return float(
            np.sqrt(np.mean(np.square(self.errors)))
        ) if self.errors else 0.0

    def save_error_plot(self):
        if not self.errors and not self.live_errors:
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        if self.live_errors:
            live_mm = np.array(self.live_errors) * 1000.0
            ax.plot(
                self.live_error_times, live_mm,
                color="#4a90d9", linewidth=1.5, alpha=0.7,
                label="Live (TF, Phase 3)"
            )

        if self.errors:
            fk_mm = np.array(self.errors) * 1000.0
            ax.plot(
                self.error_times, fk_mm,
                color="#e07b39", linewidth=2,
                label="FK post-processed (Phase 4)"
            )
            ax.axhline(
                np.mean(fk_mm),
                linestyle="--", linewidth=1.5, color="#e07b39",
                label=f"Mean (FK) = {np.mean(fk_mm):.2f} mm"
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (mm)")
        ax.set_title("3D Lissajous Tracking Error vs Time")
        ax.grid(True, alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            "tracking_error_vs_time_lissajous.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

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
        if self.live_errors:
            live_mm = np.array(self.live_errors) * 1000.0
            print(f"  Live Mean Error (TF)     : {np.mean(live_mm):.3f} mm")
            print(f"  Live Max  Error (TF)     : {np.max(live_mm):.3f} mm")
        if self.errors:
            print(f"  Mean Tracking Error (FK) : {self.mean_error()*1000:.3f} mm")
            print(f"  Max  Tracking Error (FK) : {self.max_error()*1000:.3f} mm")
            print(f"  RMS  Tracking Error (FK) : {self.rms_error()*1000:.3f} mm")
        else:
            print("  No FK tracking error data.")
        print("=" * 60 + "\n")


# ============================================================
# LiveErrorPoller  
# ============================================================
class LiveErrorPoller:
    POLL_HZ = 10  # samples per second
    def __init__(
        self,
        tf_buffer: Buffer,
        metrics: MetricsCalculator,
        valid_desired_xyz: list,    # [[x,y,z], ...] aligned with execution time
        execution_time: float,
        shared_lock: threading.Lock,
        current_pos_ref: list,      # same list Live3DPlotter uses
    ):
        self._tf_buffer        = tf_buffer
        self._metrics          = metrics
        self._desired          = valid_desired_xyz   # list of [x,y,z]
        self._exec_time        = execution_time
        self._lock             = shared_lock
        self._current_pos_ref  = current_pos_ref

        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t0 = time.time()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _get_ee_from_tf(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EEF_LINK, Time(), timeout=Duration(seconds=0.05)
            )
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except Exception:
            return None

    def _interpolate_desired(self, elapsed: float):
        n = len(self._desired)
        if n == 0:
            return None
        # Fractional index
        frac  = min(max(elapsed / self._exec_time, 0.0), 1.0)
        idx_f = frac * (n - 1)
        lo    = int(idx_f)
        hi    = min(lo + 1, n - 1)
        alpha = idx_f - lo
        d_lo  = self._desired[lo]
        d_hi  = self._desired[hi]
        return [
            d_lo[0] + alpha * (d_hi[0] - d_lo[0]),
            d_lo[1] + alpha * (d_hi[1] - d_lo[1]),
            d_lo[2] + alpha * (d_hi[2] - d_lo[2]),
        ]

    def _run(self):
        interval = 1.0 / self.POLL_HZ
        while not self._stop_event.is_set():
            elapsed = time.time() - self._t0

            actual = self._get_ee_from_tf()
            if actual is not None:
                # Update shared current_pos_ref for the plotter
                with self._lock:
                    self._current_pos_ref.clear()
                    self._current_pos_ref.extend(actual)

                desired = self._interpolate_desired(elapsed)
                if desired is not None:
                    self._metrics.add_live_error(desired, actual, elapsed)

            time.sleep(interval)


# ============================================================
# Live3DPlotter
# ============================================================
class Live3DPlotter:
    def __init__(
        self,
        desired_xyz,
        lock,
        actual_xyz_ref,
        current_pos_ref,
        metrics_ref,
        plot_limits=None,
        curve_label="Desired"
    ):
        self.desired_xyz     = desired_xyz
        self.lock            = lock
        self.actual_xyz_ref  = actual_xyz_ref
        self.current_pos_ref = current_pos_ref
        self.metrics_ref     = metrics_ref
        self.plot_limits     = plot_limits or {}
        self.curve_label     = curve_label

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            plt.ion()
            fig = plt.figure(figsize=(15, 6))

            # ── Left: 3-D trajectory ────────────────────────
            ax3d = fig.add_subplot(121, projection="3d")
            d = np.vstack([self.desired_xyz, self.desired_xyz[0]])
            ax3d.plot(d[:, 0], d[:, 1], d[:, 2],
                      "b--", linewidth=2, label=self.curve_label)
            actual_line, = ax3d.plot([], [], [], "g-", linewidth=2, label="Actual")
            current_dot, = ax3d.plot([], [], [], "ro", markersize=8, label="Current EE")

            ax3d.set_title("3D Lissajous Tracking")
            ax3d.set_xlabel("X (m)")
            ax3d.set_ylabel("Y (m)")
            ax3d.set_zlabel("Z (m)")
            lim = self.plot_limits
            ax3d.set_xlim(*lim.get("xlim", (0.20, 0.50)))
            ax3d.set_ylim(*lim.get("ylim", (-0.15, 0.15)))
            ax3d.set_zlim(*lim.get("zlim", (0.15, 0.40)))
            ax3d.legend(fontsize=8)

            # ── Right: Error vs Time ─────────────────────────
            axErr = fig.add_subplot(122)
            axErr.set_title("Tracking Error vs Time")
            axErr.set_xlabel("Time (s)")
            axErr.set_ylabel("Error (mm)")
            axErr.grid(True, alpha=0.4)

            # Live (Phase 3) series — blue
            live_line, = axErr.plot(
                [], [], color="#4a90d9", linewidth=1.5, alpha=0.8,
                label="Live (TF, Phase 3)"
            )
            live_mean_line = axErr.axhline(
                0, linestyle=":", linewidth=1.2, color="#4a90d9", alpha=0.6,
                label="Live mean"
            )

            # FK (Phase 4) series — orange
            fk_line, = axErr.plot(
                [], [], color="#e07b39", linewidth=2,
                label="FK post-proc (Phase 4)"
            )
            fk_mean_line = axErr.axhline(
                0, linestyle="--", linewidth=1.5, color="#e07b39",
                label="FK mean"
            )

            axErr.legend(fontsize=8, loc="upper right")

            # ── Refresh loop ─────────────────────────────────
            while not self._stop.is_set():

                # 3-D trajectory panel
                with self.lock:
                    actual  = list(self.actual_xyz_ref)
                    current = list(self.current_pos_ref)

                if len(actual) > 1:
                    arr = np.array(actual)
                    actual_line.set_data(arr[:, 0], arr[:, 1])
                    actual_line.set_3d_properties(arr[:, 2])

                if len(current) == 3:
                    current_dot.set_data([current[0]], [current[1]])
                    current_dot.set_3d_properties([current[2]])

                # Error panel — live (Phase 3)
                live_times, live_errs = self.metrics_ref.get_live_snapshot()
                if live_times:
                    live_mm = np.array(live_errs) * 1000.0
                    live_line.set_data(live_times, live_mm)
                    mean_live = float(np.mean(live_mm))
                    live_mean_line.set_ydata([mean_live, mean_live])
                    live_mean_line.set_label(f"Live mean = {mean_live:.2f} mm")

                # Error panel — FK (Phase 4)
                fk_errs  = self.metrics_ref.errors
                fk_times = self.metrics_ref.error_times
                if fk_errs:
                    fk_mm = np.array(fk_errs) * 1000.0
                    fk_line.set_data(fk_times, fk_mm)
                    mean_fk = float(np.mean(fk_mm))
                    fk_mean_line.set_ydata([mean_fk, mean_fk])
                    fk_mean_line.set_label(f"FK mean = {mean_fk:.2f} mm")

                if live_times or fk_errs:
                    axErr.relim()
                    axErr.autoscale_view()
                    axErr.legend(fontsize=8, loc="upper right")

                fig.canvas.draw_idle()
                plt.pause(0.2)   # 5 fps — enough for live tracking feel

            # Save combined plot on exit
            fig.savefig(
                "trajectory_and_error_lissajous.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        except Exception as exc:
            print(f"[Live3DPlotter] {exc}")
            traceback.print_exc()

    def save_final(self):
        pass


# ============================================================
# Gen3LissajousTrackingNode
# ============================================================
class Gen3LissajousTrackingNode(Node):
    def __init__(self):
        super().__init__("gen3_lissajous_tracking_node")

        # Parameters
        self.declare_parameter("amp_x",                    0.06)
        self.declare_parameter("amp_y",                    0.06)
        self.declare_parameter("amp_z",                    0.04)
        self.declare_parameter("freq_x",                   3)
        self.declare_parameter("freq_y",                   2)
        self.declare_parameter("freq_z",                   1)
        self.declare_parameter("phase_x",                  0.0)
        self.declare_parameter("phase_y",                  math.pi / 2)
        self.declare_parameter("phase_z",                  math.pi / 4)
        self.declare_parameter("center_x",                 0.35)
        self.declare_parameter("center_y",                 0.00)
        self.declare_parameter("center_z",                 0.25)
        self.declare_parameter("num_points",               200)
        self.declare_parameter("execution_time",           30.0)
        self.declare_parameter("manipulability_threshold", 0.02)

        amp_x   = self.get_parameter("amp_x").value
        amp_y   = self.get_parameter("amp_y").value
        amp_z   = self.get_parameter("amp_z").value
        freq_x  = self.get_parameter("freq_x").value
        freq_y  = self.get_parameter("freq_y").value
        freq_z  = self.get_parameter("freq_z").value
        phase_x = self.get_parameter("phase_x").value
        phase_y = self.get_parameter("phase_y").value
        phase_z = self.get_parameter("phase_z").value
        cx      = self.get_parameter("center_x").value
        cy      = self.get_parameter("center_y").value
        cz      = self.get_parameter("center_z").value

        self._num_points = self.get_parameter("num_points").value
        self._exec_time  = self.get_parameter("execution_time").value
        self._manip_thr  = self.get_parameter("manipulability_threshold").value

        self.get_logger().info(
            f"Lissajous parameters: "
            f"A=({amp_x*100:.1f},{amp_y*100:.1f},{amp_z*100:.1f}) cm  "
            f"f=({freq_x}:{freq_y}:{freq_z})  "
            f"φ=({phase_x:.2f},{phase_y:.2f},{phase_z:.2f}) rad  "
            f"center=({cx},{cy},{cz})  "
            f"n={self._num_points}  t_exec={self._exec_time}s"
        )

        # Shared state
        self._lock         = threading.Lock()
        self._actual_traj: list = []
        self._current_pos: list = []

        # Trajectory generator and support components
        self._gen = Lissajous3DTrajectoryGenerator(
            center_x=cx, center_y=cy, center_z=cz,
            amp_x=amp_x, amp_y=amp_y, amp_z=amp_z,
            freq_x=freq_x, freq_y=freq_y, freq_z=freq_z,
            phase_x=phase_x, phase_y=phase_y, phase_z=phase_z,
            num_points=self._num_points,
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

        # FK client (service — safe to create early)
        self._fk_client = self.create_client(GetPositionFK, FK_SERVICE)

        # Plotter
        self._plotter = Live3DPlotter(
            desired_xyz=self._gen.desired_xyz,
            lock=self._lock,
            actual_xyz_ref=self._actual_traj,
            current_pos_ref=self._current_pos,
            metrics_ref=self._metrics,
            plot_limits=self._gen.plot_limits,
            curve_label=self._gen.curve_label,
        )

        # CSV
        self._csv_file   = None
        self._csv_writer = None
        self._init_csv()

        # IKClient and TrajectoryExecutor set in late_init()
        self._ik_client    = None
        self._executor_obj = None

        self.get_logger().info("Gen3LissajousTrackingNode initialised (waiting for late_init).")

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
            "actual_x",  "actual_y",  "actual_z",  "error",
            "source",    # "live_tf" or "fk"
        ])

    def _log_csv(self, t, desired, actual, error, source="fk"):
        if self._csv_writer:
            self._csv_writer.writerow([
                f"{t:.4f}",
                f"{desired[0]:.6f}", f"{desired[1]:.6f}", f"{desired[2]:.6f}",
                f"{actual[0]:.6f}",  f"{actual[1]:.6f}",  f"{actual[2]:.6f}",
                f"{error:.6f}", source,
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
            with self._lock:
                self._current_pos.clear()
                self._current_pos.extend([t.x, t.y, t.z])
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

        # Phase 1: IK solve
        self.get_logger().info(
            f"Generating 3-D Lissajous waypoints — {self._gen.curve_label}…"
        )
        waypoints = self._gen.generate()

        z_min_curve = self._gen.center_z - self._gen.amp_z
        z_max_curve = self._gen.center_z + self._gen.amp_z
        if z_min_curve < self._ws_checker.z_min or z_max_curve > self._ws_checker.z_max:
            self.get_logger().warn(
                f"Lissajous Z range [{z_min_curve:.3f}, {z_max_curve:.3f}] m "
                f"may exceed workspace Z [{self._ws_checker.z_min}, "
                f"{self._ws_checker.z_max}] m — consider reducing amp_z."
            )

        self.get_logger().info("Solving IK for each waypoint…")
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

        # Phase 2: Move to start
        self.get_logger().info("Moving to trajectory start position (5 s)…")
        start_traj = self._executor_obj.build_trajectory(
            [valid_joint_positions[0]], execution_time=5.0
        )
        ok = self._executor_obj.execute(start_traj)
        if ok:
            self.get_logger().info("Reached start position.")
        else:
            self.get_logger().warn("Move-to-start incomplete; proceeding anyway.")
        time.sleep(1.0)

        # Phase 3: Execute Lissajous WITH live error polling
        traj = self._executor_obj.build_trajectory(valid_joint_positions)
        self.get_logger().info("Executing 3-D Lissajous trajectory (live error polling active)…")

        # Start live error poller JUST before execution
        live_poller = LiveErrorPoller(
            tf_buffer=self._tf_buffer,
            metrics=self._metrics,
            valid_desired_xyz=valid_desired_xyz,
            execution_time=self._exec_time,
            shared_lock=self._lock,
            current_pos_ref=self._current_pos,
        )
        live_poller.start()

        t_start = time.time()
        success = self._executor_obj.execute(traj)   # blocks until done
        t_end   = time.time()

        live_poller.stop()   # clean shutdown after execute() returns

        # Flush live errors to CSV
        with self._metrics._lock:
            for t_rel, desired_approx, err in zip(
                self._metrics.live_error_times,
                [None] * len(self._metrics.live_errors),   # desired interpolated inside poller
                self._metrics.live_errors,
            ):
                # We don't store per-sample desired in metrics; log error only
                self._log_csv(t_rel, [0, 0, 0], [0, 0, 0], err, source="live_tf")

        self._metrics.execution_time  = t_end - t_start
        self._metrics.valid_waypoints = n_valid
        self._metrics.rejected_ws     = self._ws_checker.rejected_count
        self._metrics.rejected_sing   = self._sing_checker.rejected_count
        self._metrics.rejected_ik     = self._ik_client.rejected_count

        self.get_logger().info(
            f"Execution {'succeeded' if success else 'failed/timed out'} "
            f"in {self._metrics.execution_time:.2f}s"
        )

        # Phase 4: FK for canonical error metrics + actual trajectory for plot
        self.get_logger().info("Computing FK for actual positions…")
        for i, joint_pos in enumerate(valid_joint_positions):
            t_rel   = self._metrics.execution_time * (i + 1) / n_valid
            desired = valid_desired_xyz[i]

            actual = self._get_ee_from_fk(joint_pos)
            if not actual:
                with self._lock:
                    actual = list(self._current_pos) if self._current_pos else desired

            self._metrics.add_error(desired, actual, t_rel)
            err = self._metrics.errors[-1]
            self._log_csv(t_rel, desired, actual, err, source="fk")

            with self._lock:
                self._actual_traj.append(actual)

        # Phase 5: Finalize
        self._close_csv()
        self._metrics.print_summary()
        self._metrics.save_error_plot()
        time.sleep(0.9)
        self._plotter.save_final()
        self.get_logger().info(f"Combined plot saved to trajectory_and_error_lissajous.png")
        self.get_logger().info(f"Error-only plot saved to tracking_error_vs_time_lissajous.png")


# ============================================================
# main()
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = Gen3LissajousTrackingNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(0.5)
    node.late_init()

    # Start plotter from main thread (avoids Matplotlib GUI warning)
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