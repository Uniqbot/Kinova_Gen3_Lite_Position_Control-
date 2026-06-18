# Kinova Gen3 Lite 3D Trajectory Tracking (ROS2)

This repository provides a complete pipeline to generate, validate, and execute a 3D circular trajectory for the **Kinova Gen3 Lite** manipulator using ROS2, MoveIt2, and Action Clients. The system features a live 3D plotter and real-time tracking error metrics to analyze performance on the fly.

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


your_workspace/src/your_package/
├── ...
├── scripts/
│   └── circle.py
    └── lemniscate.py
    └── lissajous.py
    └── rectangle.py
    └── square.py          # Main 3D trajectory generation & tracking script
└── README.md              # This documentation file
