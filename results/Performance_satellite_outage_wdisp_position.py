"""This code is for checking localization of onboard sensor alone
To be done:
1) Load ax,ay,vx,vy,yaw rate from onboard sensor as measurement
2) Compute position and yaw angle
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statistics as stats
from scipy.spatial.transform import Rotation as R
def wrap_to_360(angle_deg):
    """Wrap angle (in degrees) to [0, 360)."""
    return angle_deg % 360
#Outage induced for only dataset 6,7 in REV-STED
#Ensure the OBD data is rightly added

folder_path =  "phy-ml"  # 'final-ml' #'final-ml' #"phy-ml-v1-loss"  #"phy-ml"
gt_path = os.path.join(folder_path, 'gt_poses.npy')
pred_path = os.path.join(folder_path, 'pred.npy')
vars_path = os.path.join(folder_path, 'varainces.npy')

# metrics = np.load(os.path.join(folder_path,'metrics.npy'))
# #print(metrics)
# #Load Ground truth and measurements
gt_pose = np.load(os.path.join(folder_path,'gt_poses.npy'))
preds = np.load(os.path.join(folder_path,'pred.npy'))


# preds_2 = np.load(os.path.join('phy-ml_position','pred.npy'))
# mean = np.asarray([5.8357, 0.2822, 0])
# std = np.asarray([1.0488e+01, 5.0725e+00, 0.2917])
# preds_2_scaled = preds_2*std + mean
# preds = np.concatenate([preds_2_scaled[:,:3],preds],axis=-1)

#####Analyse distance##########
# max_length = 140000#len(combined_df)
# outage = int(50*60)
# final_drift_list = []
# window_size = 100
#
# def GTv0(gt_pose,idx):
#     c = np.cos(gt_pose[idx, 2])
#     s = np.sin(gt_pose[idx, 2])
#     z = np.zeros_like(s)
#     o = np.ones_like(s)
#     gTv0 = np.array([
#     [c, -s, gt_pose[idx, 0]],
#     [s,  c, gt_pose[idx, 1]],
#     [z,  z, o]
#            ], dtype=np.float32)
#     return gTv0
#
# for i, start_idx in enumerate(range(200, max_length - outage - 2, outage)):
#     # Initial position at the start of the window
#     traj_pred = np.zeros((outage+1,2))
#     curr_pos = gt_pose[start_idx, 0:2].copy()
#     traj_pred[0,:] = curr_pos
#
#     for k in range(0, outage+1, window_size):
#         # Determine index of the 2s displacement
#         # if k%window_size==0 and k!=0:
#         #     curr_pos += (GTv0(gt_pose,start_idx+k-window_size) @ np.concatenate([preds_scaled[start_idx + k, :2], [0]]))[:2] #taking only x and y
#         #     traj_pred[k,:] = curr_pos
#         curr_pos += (GTv0(gt_pose,start_idx+k-window_size) @ np.concatenate([preds_scaled[start_idx + k, :2], [0]]))[:2] #taking only x and y
#         traj_pred[k,:] = curr_pos
#
#     drift =  traj_pred[-1,:] - gt_pose[start_idx+outage, :2]
#     drift_norm = np.linalg.norm(drift)
#     final_drift_list.append(drift_norm)
#     plt.plot(traj_pred[::100, 0], traj_pred[::100, 1], label=f'Pred traj {i}')
#     plt.plot(gt_pose[start_idx:start_idx+outage, 0], gt_pose[start_idx:start_idx+outage, 1], label='GT')
#     plt.xlabel("X position")
#     plt.ylabel("Y position")
#     plt.title(f"Predicted Trajectories vs GT (Drift = {drift_norm:.2f} m)")
#     plt.legend()
#     plt.axis("equal")
#     plt.show()
#
#     drift =  traj_pred[-1,:] - gt_pose[start_idx+outage, :2]
#     drift_norm = np.linalg.norm(drift)
#     final_drift_list.append(drift_norm)
# print(final_drift_list)
# final_mean = stats.mean(final_drift_list[4:])
# final_var = stats.stdev(final_drift_list[4:])
# print(f"final_drift -> mean: {final_mean:.4f}, stddev: {final_var:.4f}")


#variances = np.load(os.path.join(folder_path,'varaince.npy'))
gt_pose_columns = ['gt_x', 'gt_y', 'gt_yaw', 'gt_yawrate', 'gt_vx', 'gt_vy', 'gt_ax', 'gt_ay', "obd_yawrate"]
preds_columns = ["delta_x","delta_y",'pred_vx', 'pred_vy', 'pred_ax', 'pred_ay']
#preds_columns = ['pred_yawrate', 'pred_vx', 'pred_vy', 'pred_ax', 'pred_ay']

# Safety check
assert gt_pose.shape[1] == len(gt_pose_columns), "Mismatch in gt_pose column count"
assert preds.shape[1] == len(preds_columns), "Mismatch in preds column count"

# Create DataFrames
gt_pose_df = pd.DataFrame(gt_pose, columns=gt_pose_columns)
gt_pose_df["obd_yawrate"] = gt_pose_df["obd_yawrate"] * -1 * (np.pi / 180) - 0.0013#Bias
preds_df = pd.DataFrame(preds, columns=preds_columns)


# Concatenate
combined_df = pd.concat([gt_pose_df, preds_df], axis=1)
#combined_df["covariance_matrix"] = list(variances)
plt.figure(figsize=(10, 5))
plt.plot(gt_pose_df['gt_yawrate']* (180 / np.pi), label='Ground Truth Yaw Rate')
#plt.plot(preds_df['pred_yawrate']* (180 / np.pi), label='Predicted Yaw Rate', linestyle='--')
plt.plot(gt_pose_df['obd_yawrate']* (180 / np.pi), label='obd Yaw Rate', linestyle='--')

# Labels and legend
plt.xlabel('Time Step')
plt.ylabel('Yaw Rate (degrees/s)')
plt.title('Ground Truth vs Predicted Yaw Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#print('hi')





#EKF function
# Define state equations and state vector.
x_s_1 = np.zeros(8) #Format of state is [xs (0),ys (1),θyaw (2),θ'yaw (3),vx (4),vy (5),ax (6),ay (7)]
def state_equations(x_s_1, delta_t):
    x = x_s_1[0]
    y = x_s_1[1]
    yaw = x_s_1[2]
    yaw_rate = x_s_1[3]
    vx = x_s_1[4]
    vy = x_s_1[5]
    ax = x_s_1[6]
    ay = x_s_1[7]

    # Rotation matrix from vehicle frame to global frame
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw],
                  [sin_yaw,  cos_yaw]])

    # Velocity and acceleration in global frame
    vel_global = R @ np.array([vx, vy])
    acc_global = R @ np.array([ax, ay])

    # Initialize new state vector
    x_s = np.zeros_like(x_s_1)

    # Position update (using kinematic equations in global frame)
    x_s[0] = x + vel_global[0] * delta_t + 0.5 * acc_global[0] * delta_t**2
    x_s[1] = y + vel_global[1] * delta_t + 0.5 * acc_global[1] * delta_t**2

    # Yaw and yaw rate
    x_s[2] = yaw + yaw_rate * delta_t
    x_s[3] = yaw_rate  # Assuming constant yaw rate

    # Velocity update in vehicle frame (no change needed here)
    x_s[4] = vx + ax * delta_t
    x_s[5] = vy + ay * delta_t

    # Constant acceleration assumption
    x_s[6] = ax
    x_s[7] = ay

    return x_s


#Format of measurement is [xh (0),yh (1),θyaw (2),θ'yaw (3),vh (4),βh (5), ax (6),ay (7)]
#Signs are verified
def measurement_vector_init(data,iter):
    z_h = np.zeros(8)
    #Check - sign for the position
    z_h[0] = data['gt_x'].iloc[iter]
    z_h[1] = data['gt_y'].iloc[iter]
    z_h[2] = (data['gt_yaw'].iloc[iter])
    z_h[3] = (data["gt_yawrate"].iloc[iter])
    z_h[4] = data['gt_vx'].iloc[iter]
    z_h[5] = data[('gt_vy')].iloc[iter]
    z_h[6] = data[('gt_ax')].iloc[iter]
    z_h[7] = data[('gt_ay')].iloc[iter]
    return z_h




def measurement_vector(data,iter):
    z_h = np.zeros(5)
    #Check - sign for the position
    #z_h[0] = data['gt_x'].iloc[iter]*0
    #z_h[1] = data['gt_y'].iloc[iter]*0
    #z_h[2] = (data['gt_yaw'].iloc[iter])*0
    z_h[0] = (data['obd_yawrate'].iloc[iter])
    z_h[1] = data['pred_vx'].iloc[iter]
    z_h[2] = data[('pred_vy')].iloc[iter]
    z_h[3] = data[('pred_ax')].iloc[iter]
    z_h[4] = data[('pred_ay')].iloc[iter]
    #std =  np.array(data[("covariance_matrix")].iloc[iter])
    # S = np.diag([ 1,1  ,3.5945,0.1175,0.6819,1.6031])
    # std =  S @ std @ S
    #r_used = np.zeros((5,5))
    # yawrate_variance = 1e-3
    # r_used[0, 0] = yawrate_variance
    # r_used[1:, 1:] = std
    return z_h

#Return predicted measurement vector from state [xh (0),yh (1),θyaw (2),θ'yaw (3),vh (4),βh (5), ax (6),ay (7)]
def h(x):
    z_pred = np.zeros(5)
    z_pred[0] = x[3]  # yaw_rate
    z_pred[1] = x[4]  # vx
    z_pred[2] = x[5]  # vy
    z_pred[3] = x[6]  # ax
    z_pred[4] = x[7]  # ay
    return z_pred

# Noise matrix should be tailor made for working
n = 8 #State Dimension
m = 5 #Measurement Dimension
P = np.eye(n)*0.01 #Initial state covariance
Q = np.eye(n)*0.01 #Process noise covariance
std_dev_measurments = [.01,.01,.01, .05,.05] #Measurement noise std with satellites
R_s = np.diag([sd**2 for sd in std_dev_measurments]) #R with satellite
R_o = np.diag([sd**2 for sd in std_dev_measurments]) #R during satellite outage


def ekf_predict(x, P, delta_t):
    F = numerical_jacobian_f(x, delta_t)
    x_pred = state_equations(x, delta_t) # Predicted state estimate
    P_pred = F @ P @ F.T + Q             # Predicted state covariance
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z_meas, use_satellite=True):
    H = numerical_jacobian_h(x_pred)
    z_pred = h(x_pred)
    y = z_meas - z_pred
    R_used = R_s if use_satellite else R_o #Decides the noise matrix
    #R_used = std
    S = H @ P_pred @ H.T + R_used
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_new, P_new

def ekf_special_update(x_pred, P_pred, z_meas,pose,delta_x,delta_y,delta_yaw,use_satellite=True):
    H = numerical_jacobian_h(x_pred)
    z_pred = h(x_pred)
    y = z_meas - z_pred
    R_used = R_s if use_satellite else R_o #Decides the noise matrix
    #R_used = std
    S = H @ P_pred @ H.T + R_used
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(len(x_pred)) - K @ H) @ P_pred

    #Shitty code check
    x_dis = pose[0] + np.cos(pose[2]) * delta_x - np.sin(pose[2]) * delta_y
    y_dis = pose[1] + np.sin(pose[2]) * delta_x + np.cos(pose[2]) * delta_y
    yaw = pose[2]+delta_yaw
    x_new[0] = (0.1*x_dis+0.9*x_new[0])
    x_new[1] = (0.1*y_dis+0.9*x_new[1])
    #x_new[2] = (0.1*yaw+0.9*x_new[2])
    return x_new, P_new


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


# From one iter point calculate all and check code.
max_length = 140000#len(combined_df)
outage = int(50*60)
max_drift_list = []
final_drift_list = []
yaw_drift_list = []

for i,j in enumerate(range(0,max_length-outage-2,outage)):
    print("current_scenario",i)
    iter_start = j                   #129317  #200000 for dynamic side #from 5000

    #Initialization
    delta_t = 0.02                    #0.02 Or get from timestamp differences if available

    # Initialize state from measurement at iter_start
    z_init = measurement_vector_init(combined_df, iter_start)
    x_s_1 = np.zeros(8)
    x_s_1[0] = z_init[0]
    x_s_1[1] = z_init[1]
    x_s_1[2] = z_init[2]
    x_s_1[3] = z_init[3]
    x_s_1[4] = z_init[4]
    x_s_1[5] = z_init[5]
    x_s_1[6] = z_init[6]
    x_s_1[7] = z_init[7]

    # Run state update over 250 iterations
    trajectory = [x_s_1.copy()]
    for i in range(iter_start + 1, iter_start + outage+1):
        dt = delta_t
        # Predict
        x_pred, P_pred = ekf_predict(x_s_1, P, dt)
        # Measurement
        z= measurement_vector(combined_df, i)
        if (i-iter_start)%100==0:
            pose=trajectory[i-iter_start-100]
            delta_x = combined_df['delta_x'].iloc[i]
            delta_y = combined_df['delta_y'].iloc[i]
            delta_yaw = 0             #combined_df['delta_yaw'].iloc[i]*
            x, P = ekf_special_update(x_pred, P_pred, z,  pose, delta_x,delta_y,delta_yaw, use_satellite=False)
        else:
            x, P = ekf_update(x_pred, P_pred, z, use_satellite=False)
        x_s_1 = x.copy() #Recursion
        trajectory.append(x.copy())

    # Convert to DataFrame for easier analysis/plotting
    trajectory_df = pd.DataFrame(trajectory, columns=['xs', 'ys', 'theta_yaw', 'theta_yaw_rate',
                                                      'vxs', 'vys', 'axs', 'ays'])
    measured_x = combined_df['gt_x'].iloc[iter_start:iter_start+outage+1].values
    measured_y = combined_df['gt_y'].iloc[iter_start:iter_start+outage+1].values
    measured_vx = combined_df['gt_vx'].iloc[iter_start:iter_start+outage+1].values
    measured_yaw = combined_df['gt_yaw'].iloc[iter_start:iter_start + outage + 1].values
    estimated_x = trajectory_df['xs'].values
    estimated_y = trajectory_df['ys'].values
    estimated_vx = trajectory_df['vxs'].values
    estimated_yaw = trajectory_df['theta_yaw'].values
    estimated_yaw_wrapped = (estimated_yaw % (2 * np.pi)) - 2 * np.pi

    drift = np.sqrt((estimated_x[-1]-measured_x[-1])**2+(estimated_y[-1] - measured_y[-1])**2)
    final_drift_list.append(drift)

    yaw_est_final = np.degrees(estimated_yaw_wrapped[-1])  # convert from radians → degrees
    yaw_meas_final = np.degrees(measured_yaw[-1])

    # plt.figure(figsize=(10, 6))
    # plt.plot( estimated_vx, label='estimated vel', linestyle='--', color='blue')
    # plt.plot( measured_vx, label='measrued vel', color='orange')
    # plt.title('Trajectory Comparison: Estimated vs Measured vel')
    # plt.legend()
    # plt.grid(True)
    # #plt.axis('equal')
    # plt.tight_layout()
    # plt.show(block=True)


    # plt.figure(figsize=(10, 6))
    # plt.plot( np.degrees(measured_yaw), label='Measured yaw', linestyle='--', color='blue')
    # plt.plot( np.degrees(estimated_yaw_wrapped), label='Estimated yaw', color='orange')
    # plt.title('Trajectory Comparison: Estimated vs Measured yaw')
    # plt.legend()
    # plt.grid(True)
    # #plt.axis('equal')
    # plt.tight_layout()
    # plt.show(block=True)
    # #Smallest angular difference in degrees
    # yaw_drift =np.abs(yaw_est_final - yaw_meas_final)
    # yaw_drift_list.append(yaw_drift)
    #
    drift_per_step = np.sqrt((estimated_x - measured_x) ** 2 + (estimated_y - measured_y) ** 2)
    max_drift = np.max(drift_per_step)
    max_drift_list.append(max_drift)

    print('drift_final', drift)
    print('drift_max_during_outage', max_drift)
    # plt.figure(figsize=(10, 6))
    # plt.plot(measured_x, measured_y, label='Measured INS Position', linestyle='--', color='blue')
    # plt.plot(estimated_x, estimated_y, label='Estimated Position during outage', color='orange')
    # drift_text = f'Max Drift Outage: {max_drift:.2f} m'
    # text_x = min(plt.xlim()) + 1
    # text_y = max(plt.ylim()) - 10
    # plt.text(text_x, text_y, drift_text, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8))
    # plt.xlabel('X Position (m)')
    # plt.ylabel('Y Position (m)')
    # plt.title('Trajectory Comparison: Estimated vs Measured')
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

print("max_drift",max_drift_list)
print("final_drift",final_drift_list)
print("final_drift_yaw_indegreees",yaw_drift_list)

# Mean and variance for final_drift_list
max_mean = stats.mean(max_drift_list)
max_var = stats.stdev(max_drift_list)
final_mean = stats.mean(final_drift_list)
final_var = stats.stdev(final_drift_list)
print(f"max_drift -> mean: {max_mean:.4f}, stddev: {max_var:.4f}")
print(f"final_drift -> mean: {final_mean:.4f}, stddev: {final_var:.4f}")

# print('drift_final', drift)
# print('drift_max_during_outage', max_drift)
# plt.figure(figsize=(10, 6))
# plt.plot(measured_x, measured_y, label='Measured INS Position', linestyle='--', color='blue')
# plt.plot(estimated_x, estimated_y, label='Estimated Position during outage', color='orange')
# drift_text = f'Max Drift Outage: {max_drift:.2f} m'
# text_x = min(plt.xlim()) + 1
# text_y = max(plt.ylim()) - 10
# plt.text(text_x, text_y, drift_text, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8))
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title('Trajectory Comparison: Estimated vs Measured')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()



#Conditioning based EKF
# yaw_condition = combined_df['obd_yawrate'].abs() < 0.5 #in radians
# speed = np.sqrt(combined_df['gt_vx']**2+combined_df['gt_vy']**2)*3.6
# speed_condition = speed < 20  # upper limit
#
# valid_rows = yaw_condition & speed_condition
# mask = valid_rows.to_numpy().astype(int)
#
# d = np.diff(mask, prepend=0, append=0)
# starts = np.where(d == 1)[0]
# ends   = np.where(d == -1)[0]
#
# segments = []
# for s, e in zip(starts, ends):
#     if (e - s) >= 3000 and speed.iloc[s:e].mean() > 2:
#         segments.append((s, e - 1))
#
# print(segments)