"""This code is for checking localization of onboard sensor alone
To be done:
1) Load ax,ay,vx,vy,yaw rate from onboard sensor as measurement
2) Compute position and yaw angle

Improvements:
With polyfit for yaw and KF training in fine tuning stage it has improved
Major scope now is to half the standard deviation
"""
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import statistics as stats
from scipy.spatial.transform import Rotation as R
def wrap_to_360(angle_deg):
    """Wrap angle (in degrees) to [0, 360)."""
    return angle_deg % 360
#Outage induced for only dataset 6,7 in REV-STED
#Ensure the OBD data is rightly added

folder_path =  "final-ml" #'final-ml' #'final-ml' #"phy-ml-v1-loss"  #"phy-ml"
type = "fine_tune"
gt_path = os.path.join(folder_path, f'gt_poses_physics_best.npy')
pred_path = os.path.join(folder_path, f'pred_{type}.npy')
#vars_path = os.path.join(folder_path, 'varainces.npy')

# metrics = np.load(os.path.join(folder_path,'metrics.npy'))
# #print(metrics)
# #Load Ground truth and measurements
gt_pose = np.load(os.path.join(folder_path,f'gt_poses_physics_best.npy'))
preds = np.load(os.path.join(folder_path,f'pred_{type}.npy'))
# mean = np.asarray([5.8357, 0.2822, 0])
# std = np.asarray([1.0488e+01, 5.0725e+00, 0.2917])
# preds_scaled = preds*std + mean
print('hi')


#variances = np.load(os.path.join(folder_path,'varaince.npy'))
gt_pose_columns = ['gt_x', 'gt_y', 'gt_yaw', 'gt_yawrate', 'gt_vx', 'gt_vy', 'gt_ax', 'gt_ay', "obd_yawrate","LatAcc_obd", "brake_pressure_obd", "speedo_obd", "SW_pos_obd","VelFR_obd","VelFL_obd","VelRR_obd","VelRL_obd"]
preds_columns = ['pred_vx', 'pred_vy', 'pred_ax', 'pred_ay']
#preds_columns = ['pred_yawrate', 'pred_vx', 'pred_vy', 'pred_ax', 'pred_ay']

# Safety check
assert gt_pose.shape[1] == len(gt_pose_columns), "Mismatch in gt_pose column count"
assert preds.shape[1] == len(preds_columns), "Mismatch in preds column count"

# Create DataFrames
gt_pose_df = pd.DataFrame(gt_pose, columns=gt_pose_columns)
gt_pose_df["obd_yawrate"] = gt_pose_df["obd_yawrate"] * -1 * (np.pi / 180) #- 0.0013#Bias
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

#Physics Guard Layer for the model
#Zero velocity update
combined_df.loc[combined_df['speedo_obd'] < 2, ['pred_vx','pred_vy','pred_ay','pred_ax']] = 0
#combined_df['obd_yawrate'] = combined_df['obd_yawrate'].rolling(5, center=True, min_periods=1).median()
X = combined_df["obd_yawrate"].values.reshape(-1, 1)
y = combined_df["gt_yawrate"].values
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
features = ['obd_yawrate', "LatAcc_obd", "brake_pressure_obd", "speedo_obd", "SW_pos_obd", "VelFR_obd", "VelFL_obd", "VelRR_obd", "VelRL_obd"]
loaded_model = joblib.load("/home/ws5/Desktop/Code_Base_Genesys/Implemented_ML_Papers/ICRA_2026/Kalman_Pose/obd_to_ins_yaw_mlp_model.pkl")
loaded_poly_model = joblib.load("/home/ws5/Desktop/Code_Base_Genesys/Implemented_ML_Papers/ICRA_2026/Kalman_Pose/obd_yawrate_poly_model.pkl")
combined_df["obd_yawrate_calib"] = loaded_poly_model.predict(combined_df['obd_yawrate'].values.reshape(-1, 1))
#combined_df["obd_yawrate_calib"] = loaded_model.predict(combined_df['obd_yawrate'].values.reshape(-1, 1))




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
    z_h[0] = (data["obd_yawrate_calib"].iloc[iter])
    #z_h[0] = (data["gt_yawrate"].iloc[iter])
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
        # Update
        x, P = ekf_update(x_pred, P_pred, z, use_satellite=False)
        x_s_1 = x.copy() #Recursion
        trajectory.append(x.copy())

    # Convert to DataFrame for easier analysis/plotting
    trajectory_df = pd.DataFrame(trajectory, columns=['xs', 'ys', 'theta_yaw', 'theta_yaw_rate',
                                                      'vxs', 'vys', 'axs', 'ays'])
    measured_x = combined_df['gt_x'].iloc[iter_start:iter_start+outage+1].values
    measured_y = combined_df['gt_y'].iloc[iter_start:iter_start+outage+1].values
    measured_yaw = combined_df['gt_yaw'].iloc[iter_start:iter_start + outage + 1].values


    estimated_x = trajectory_df['xs'].values
    estimated_y = trajectory_df['ys'].values
    estimated_yaw = trajectory_df['theta_yaw'].values
    estimated_yaw_wrapped = (estimated_yaw % (2 * np.pi)) - 2 * np.pi

    drift = np.sqrt((estimated_x[-1]-measured_x[-1])**2+(estimated_y[-1] - measured_y[-1])**2)
    final_drift_list.append(drift)

    yaw_est_final = np.degrees(estimated_yaw_wrapped[-1])  # convert from radians → degrees
    yaw_meas_final = np.degrees(measured_yaw[-1])


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
    yaw_drift =np.abs(yaw_est_final - yaw_meas_final)
    yaw_drift_list.append(yaw_drift)

    drift_per_step = np.sqrt((estimated_x - measured_x) ** 2 + (estimated_y - measured_y) ** 2)
    max_drift = np.max(drift_per_step)
    max_drift_list.append(max_drift)
    #
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





print("max_drift",max_drift_list)
print("final_drift",final_drift_list)
print("final_drift_yaw_indegreees",yaw_drift_list)

# Mean and variance for final_drift_list
max_mean = stats.mean(max_drift_list[:])
max_var = stats.stdev(max_drift_list[:])
final_mean = stats.mean(final_drift_list[:])
final_var = stats.stdev(final_drift_list[:])
print(f"max_drift -> mean: {max_mean:.4f}, stddev: {max_var:.4f}")
print(f"final_drift -> mean: {final_mean:.4f}, stddev: {final_var:.4f}")
print(f"final yaw drift -> mean: {final_mean:.4f}, stddev: {final_var:.4f}")

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