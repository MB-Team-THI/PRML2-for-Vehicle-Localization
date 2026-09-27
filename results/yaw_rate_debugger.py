#Already time synchronized file is present
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("obd data checker/merged_dataset_steer.csv")



primary_col = 'byte_1'
for i in range(0, 8):  # byte0 to byte6
    secondary_col = f'byte_{i}'
    if secondary_col == primary_col:
        continue

    new_col_name = f'decoded_{primary_col}_{secondary_col}'
    df[new_col_name] = np.nan  # initialize column with zeros

    for idx, (p, s) in enumerate(zip(df[primary_col], df[secondary_col])):
        try:
            p_int = int(p)
            s_int = int(s)

            p_bin = f'{p_int:08b}'
            s_bin = f'{s_int:08b}'

            combined_int = int(p_bin[1:] + s_bin, 2)
            combined_int = combined_int*0.05
            if p_bin[0]=='0':
                decoded_val = combined_int
            else:
                decoded_val = -1*combined_int


        except (ValueError, TypeError):
            decoded_val = np.nan # fill zero if error

        df.loc[idx, new_col_name] = decoded_val


    # Plot the result
    df_clean = df.dropna(subset=[new_col_name])
    plt.figure()
    plt.plot(df_clean[new_col_name], label=f'{primary_col} + {secondary_col}')
    plt.plot(df_clean['rate_hor_z'], label=f'ADMA')
    plt.title(f'Decoded Value from {primary_col} + {secondary_col}')
    plt.xlabel('Sample Index')
    plt.ylabel('Decoded Output')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#Checking
# df = pd.read_csv("data/rev-sted/Recording_6_ReVSTED.csv")
#
# plt.figure()
# plt.plot(df["speedo_obd"]*1, label='Speedo_OBD')
# # plt.plot(df["rate_hor_z"], label='yaw rate adma')
# # plt.plot(df["VelFR_obd"]*20/16, label='FR_OBD')
#
# plt.plot(((df["VelFR_obd"]*1) + (df["VelFL_obd"]*1))*1.05/2  , label='avg')
#
# #plt.plot(df["VelFL_obd"], label='FL_OBD')
# #plt.plot(df["VelRL_obd"]*20/16, label='RL_OBD')
# #plt.plot(df["VelRR_obd"]*20/16, label='RR_OBD')
# #plt.plot(np.sqrt(df["ins_vel_hor_x"]**2+df["ins_vel_hor_y"]**2)*3.6, label='ADMA')
# plt.title(f'Decoder checker ')
# plt.xlabel('Sample Index')
# plt.ylabel('Decoded Output')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
#
#
#
# print("Hi")


#Final yaw rate configuration
# if RawOBDData.arbitration_id == 329:  # 0x149
#
#     YawRateInfoMsg = YawRateInfo()  # Creating an instance (copy) of the structure of the message
#     YawRateInfoMsg.timestampGierrateRohsignal = RawOBDData.timestamp  # Saving the timestamp of the can msg in the ros msg
#     GierrateRohsignalHighByteinBin = '{0:08b}'.format(
#         int(CanMsginInt[0, 1]))  # Gierrate - Storing the MOST significant byte in binary
#     GierrateRohsignalLowByteinBin = '{0:08b}'.format(
#         int(CanMsginInt[0, 0]))  # Gierrate - Storing the LEAST significant byte in binary
#
#     YawRateInBin = GierrateRohsignalHighByteinBin + GierrateRohsignalLowByteinBin  # Gierrate - Storing the 16-bit chain
#
#     YawRateInfoMsg.GierrateDegXSec = ((float(int(YawRateInBin, 2))) * 0.005) - 163.84  # binary to float conversion
#
#     self.pub_YawRateInfo.publish(YawRateInfoMsg)