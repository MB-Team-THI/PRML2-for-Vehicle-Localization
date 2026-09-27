import rosbag

rosbag_name = "2024-05-14-15-19-16.bag"


#topic = /adma/data_scaled
#acc_body = msg.acc_body.x, msg.acc_body.y,msg.acc_body.z
#rate_body = msg.rate_body.x, msg.rate_body.y, msg.rate_body.z
#pose = msg.ins_pos_rel_x, msg.ins_pos_rel_y, msg.ins_pitch,msg.ins_roll,msg.ins_yaw
#vel = msg.ins_vel_hor.x,msg.ins_vel_hor.y
for topic, msg, t in rosbag.Bag(rosbag_name).read_messages():
    if topic == '/adma/data_scaled':
        print('time', msg.ins_time_msec, msg.ins_time_week, msg.leap_seconds)
        print('acc_boby',msg.acc_body.x, msg.acc_body.y,msg.acc_body.z)
        print('rate_body',msg.rate_body.x, msg.rate_body.y, msg.rate_body.z )
        print('pose',msg.ins_pos_rel_x, msg.ins_pos_rel_y, msg.ins_pitch,msg.ins_roll,msg.ins_yaw )
        print('vel',msg.ins_vel_hor.x,msg.ins_vel_hor.y)