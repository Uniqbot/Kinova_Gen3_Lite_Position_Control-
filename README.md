# Kinova Gen3 Lite 3D Trajectory Tracking (ROS2)

This repository provides a complete pipeline to generate, validate, and execute a 3D circular trajectory for the **Kinova Gen3 Lite** manipulator using ROS2, MoveIt2, and Action Clients. The system features real-time diagnostics, safety-aware trajectory validation, and comprehensive performance tracking.

## 📌 Features

* **Trajectory Generation:** Automatic 3D Cartesian waypoint calculation for circular paths.
* **Safety & Feasibility Filters:** Built-in workspace constraint validation and kinematic singularity avoidance.
* **Synchronized Action Client:** Trajectory execution via a `FollowJointTrajectory` action interface using a multi-threaded executor.
* **Real-Time Diagnostics:** Live 3D position tracing and dynamic statistical tracking error plotting (Mean, Max, and RMS error in mm).
* **Data Logging:** Generates comprehensive CSV logs and high-resolution performance plots after execution.

---

## 🛠️ Prerequisites & Installation

### 1. Install `ros2_kortex` Dependencies
Before running this project, you must install the official Kinova ROS2 drivers and simulation packages. 

Follow the installation instructions provided in the official repository:
👉 **[Official Kinova ros2_kortex Repository](https://github.com/Kinovarobotics/ros2_kortex/tree/main)**

Ensure your ROS2 workspace builds completely and all MoveIt2/Kortex packages are correctly sourced.

### 2. Python Dependencies
The tracking script requires external libraries for matrix operations, kinematics calculations, and live GUI visualization. Install them using `pip`:

```bash
pip install numpy matplotlib scipy
```

### 3. Project Structure
Ensure your ROS2 workspace has the following structure:

```
your_workspace/src/your_package/
├── ...
├── scripts/
│   ├── circle.py
│   ├── lemniscate.py
│   ├── lissajous.py
│   ├── rectangle.py
│   └── square.py          # 3D trajectory generation & tracking scripts
└── README.md              # This documentation file
```

---

## 🚀 Execution Guide

Follow these steps in separate terminal windows (ensure your workspace is sourced in each window).

### Step 1: Launch the Kinova Simulation & Control

Start the official Kortex simulation environment. This spins up the robot hardware interfaces, controllers, and MoveIt2 services:

```bash
ros2 launch kortex_bringup kortex_sim_control.launch.py robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f use_sim_time:=true launch_rviz:=false robot_name:=gen3_lite_gen3_lite_2f
```

### Step 2: Launch MoveIt2 Demo

```bash
ros2 launch kinova_gen3_lite_moveit_config demo.launch.py
```

### Step 3: Run the Trajectory Script

```bash
ros2 run your_package circle.py
```

Replace `circle.py` with `lemniscate.py`, `lissajous.py`, `rectangle.py`, or `square.py` as needed.

### Step 4: Verify Services Are Ready

Wait until the simulation is fully loaded and the following action servers are ready:
- `/compute_ik` (Inverse Kinematics service)
- `/joint_trajectory_controller/follow_joint_trajectory` (Trajectory execution action)


