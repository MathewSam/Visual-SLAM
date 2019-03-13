import argparse
import numpy as np
from utils import *
from SENSOR import Camera,Sensor
from EKF import MotionModel,Landmark,invert_pose

parser = argparse.ArgumentParser()
parser.add_argument("data_file",help="file containing features and IMU information")

if __name__ == '__main__':
	args = parser.parse_args()

	filename = args.data_file
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
	sensor_stream = Sensor(t,linear_velocity,rotational_velocity)
	camera_stream = Camera(features,cam_T_imu,K,b)
	
	VSLAM = MotionModel()
	landmarks = [Landmark(camera_stream.M,camera_stream.oTi) for i in range(features.shape[1])]
	
	world_T_imu = np.zeros((4,4,t.shape[1]))
	# You can use the function below to visualize the robot pose over time

	for i in range(t.shape[1]):
		if i!=0:
			VSLAM.prediction(sensor_stream[i-1][0],sensor_stream[i-1][1])
		for lndmrk_index in range(features.shape[1]):
			if features[0,lndmrk_index,i]!=-1 and features[1,lndmrk_index,i]!=-1 and features[2,lndmrk_index,i]!=-1 and features[3,lndmrk_index,i]!=-1:
				landmarks[lndmrk_index].update_landmark_position(features[:,lndmrk_index,i],VSLAM.mu)
		
		world_T_imu[:,:,i] = invert_pose(VSLAM.mu)
	
	visualize_trajectory_2d(world_T_imu,landmarks,path_name=filename[-8:-4],show_ori=True)

