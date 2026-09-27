#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from your_adma_msgs.msg import AdmaData
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

def state_equations(x_s_1, delta_t):
    x_s_1 = x_s_1.copy()
    theta_yaw = x_s_1[2]
    theta_yaw_rate = x_s_1[3]
    vs = x_s_1[4]
    beta_sv = x_s_1[5]
    beta_dot = x_s_1[6]
    ax = x_s_1[7]
    ay = x_s_1[8]
    # New state vector
    x_s = np.zeros_like(x_s_1)
    # Position Update
    x_s[0] = x_s_1[0] + vs * np.cos(theta_yaw + beta_sv) * delta_t + ((ax * np.cos(theta_yaw) - ay * np.sin(theta_yaw)) * (delta_t**2) / 2)
    x_s[1] = x_s_1[1] + vs * np.sin(theta_yaw + beta_sv) * delta_t + ((ax * np.sin(theta_yaw) + ay * np.cos(theta_yaw)) * (delta_t ** 2) / 2)
    # Yaw angle and yaw rate
    x_s[2] = theta_yaw + theta_yaw_rate * delta_t
    x_s[3] = theta_yaw_rate
    # Velocity and sideslip angles
    x_s[4] = vs + (ax * np.cos(beta_sv) + ay * np.sin(beta_sv)) * delta_t
    x_s[5] = beta_sv + beta_dot * delta_t
    # Avoid dividing by very small vs
    if np.abs(vs) > 1e-6:
        x_s[6] = (1.0 / vs * (ay * np.cos(beta_sv) - ax * np.sin(beta_sv))) - theta_yaw_rate
    else:
        x_s[6] = 0.0
    # Constant acceleration assumption
    x_s[7] = ax
    x_s[8] = ay
    if x_s[4] < 1.0:
        x_s[5] = 0.0
        x_s[6] = 0.0
    return x_s


def h(x):
    z_pred = np.zeros(8)
    z_pred[0] = x[0]   # x
    z_pred[1] = x[1]   # y
    z_pred[2] = x[2]   # yaw
    z_pred[3] = x[3]   # yaw_rate
    z_pred[4] = x[4]   # v_s
    z_pred[5] = x[5]   # beta
    z_pred[6] = x[7]   # ax
    z_pred[7] = x[8]   # ay
    return z_pred

def numerical_jacobian_f(x, dt, eps=1e-5):
    n = len(x)
    F = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        fx_plus = state_equations(x_plus, dt)
        fx_minus = state_equations(x_minus, dt)
        F[:, i] = (fx_plus - fx_minus) / (2 * eps)
    return F

def numerical_jacobian_h(x, eps=1e-5):
    m = len(h(x))
    n = len(x)
    H = np.zeros((m, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        h_plus = h(x_plus)
        h_minus = h(x_minus)
        H[:, i] = (h_plus - h_minus) / (2 * eps)
    return H

# State / covariances (defaults from your snippet)
N_STATE = 9
M_MEAS = 8
P_INIT = np.eye(N_STATE) * 0.001
Q = np.eye(N_STATE) * 0.01
std_dev_measurements = [.01, .01, .00026, .00005, .0083, .05, .03, .03]
R_s = np.diag([sd**2 for sd in std_dev_measurements])  # with satellite
R_o = R_s.copy()
R_o[0, 0] = 1e6
R_o[1, 1] = 1e6
R_o[2, 2] = 1e6

def ekf_predict(x, P, delta_t):
    F = numerical_jacobian_f(x, delta_t)
    x_pred = state_equations(x, delta_t)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z_meas, use_satellite=True):
    H = numerical_jacobian_h(x_pred)
    z_pred = h(x_pred)
    y = z_meas - z_pred
    R_used = R_s if use_satellite else R_o
    S = H @ P_pred @ H.T + R_used
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_new, P_new



# -----------------------
# ROS2 Node for EKF
# -----------------------
class DeadReckoningNode(Node):
    def __init__(self):
        super().__init__('EKF_dead_reckoning_node')

        # Subscribers for accel and gyro
        self.adma_sub = self.create_subscription(
            AdmaData,
            'adma/data_scaled',
            self.adma_callback,
            10)


        # Publisher: Odometry with predicted state
        self.pose_pub = self.create_publisher(PoseStamped, 'dead_reckoning/pose', 10)

        # Storage for latest sensor values
        self.latest_accel = None
        self.latest_gyro = None
        self.initialized = False
        # EKF state and covariance
        self.x = np.zeros(N_STATE)     # initial state (9)
        self.P = P_INIT.copy()         # initial covariance

        # timer rate and delta_t
        # self.delta_t = 0.02  # 50 Hz default
        # self.timer = self.create_timer(self.delta_t, self.timer_callback)

        # time bookkeeping to allow variable dt in future
        self._last_time = self.get_clock().now()

        self.get_logger().info("DeadReckoningNode started. Subscribed to /adma_data_scaled. Publishing /dead_reckoning/odometry")

    # -----------------------
    # Callbacks for sensors
    # -----------------------
    def adma_callback(self, msg):

        accel = np.array([msg.acc_body.x,msg.acc_body.y,msg.acc_body.z])
        gyro = np.array([msg.rate_body.x,msg.rate_body.y,msg.rate_body.z])
        pose = np.array([msg.ins_pos_rel_x,msg.ins_pos_rel_y,msg.ins_pitch,msg.ins_roll,msg.ins_yaw])
        vel = np.array([msg.ins_vel_hor.x,msg.ins_vel_hor.y])

        self.latest_accel = accel
        self.latest_gyro = gyro
        self.latest_pose = pose
        self.latest_vel = vel

        if not self.initialized:
            self.x[0] = pose[0]  # x
            self.x[1] = pose[1]  # y
            self.x[2] = pose[4]  # yaw
            self.x[3] = gyro[2]  # yaw rate
            self.x[4] = np.linalg.norm(vel)  # horizontal speed
            self.x[5] = 0 #side slip
            self.x[7] = accel[0]  # ax
            self.x[8] = accel[1]  # ay
            self.initialized = True
            self.get_logger().info("EKF initialized from first ADMA measurement")

        z_meas = np.array([
            pose[0],  # x
            pose[1],  # y
            pose[4],  # yaw
            gyro[2],  # yaw_rate
            np.linalg.norm(vel),  # v_s
            0.0,  # beta (not measured, so set to 0, high R variance)
            accel[0],  # ax
            accel[1]  # ay
            ])

        self.prediction()

        self.apply_measurement(z_meas,use_satellite=False)

        self.publish_dead_reconning()

    # -----------------------
    # Measurement API to be used later
    # -----------------------
    def apply_measurement(self, z_meas: np.ndarray, use_satellite: bool = False):
        """
        z_meas: 8-D measurement vector in same format as your measurement_vector/h() expects:
          [xh(0), yh(1), θyaw(2), θ'yaw(3), vh(4), βh(5), ax(6), ay(7)]
        """
        if z_meas.shape[0] != M_MEAS:
            self.get_logger().error("apply_measurement: z_meas must be 8-element vector")
            return
        self.x, self.P = ekf_update(self.x, self.P, z_meas, use_satellite)
        self.get_logger().debug("EKF update applied")

    def prediction(self):
        if not self.initialized:
            return
        # now = self.get_clock().now()
        # dt = (now - self._last_time).nanoseconds * 1e-9
        # if dt <= 0 or dt > 1.0:
        #     dt = self.delta_t  # fallback to nominal
        # self._last_time = now


        # EKF prediction using current state only
        x_pred, P_pred = ekf_predict(self.x, self.P, delta_t=0.01)
        self.x = x_pred
        self.P = P_pred

    def publish_dead_reconning(self):
        now = self.get_clock().now()

        # Publish odometry message built from predicted state
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now.to_msg()
        pose_msg.header.frame_id = "odom"

        # pose
        pose_msg.pose.position.x = float(self.x[0])  # EKF predicted x
        pose_msg.pose.position.y = float(self.x[1])  # EKF predicted y
        pose_msg.pose.position.z = 0.0

        # Use measured yaw from ADMA
        yaw_meas = float(self.x[2])
        q = R.from_euler('z', yaw_meas).as_quat()
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self.pose_pub.publish(pose_msg)

        # debug
        self.get_logger().debug(
            f"Published pose: x={self.x[0]:.3f}, y={self.x[1]:.3f}, yaw={yaw_meas:.3f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()