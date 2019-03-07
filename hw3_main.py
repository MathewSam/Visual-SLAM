'''
## TO DO:
1.	Set up camera wrapper to handle data management
2.	Fix observational model to work

'''
import numpy as np
from utils import *
from SENSOR import Camera,Sensor
from EKF import MotionModel

class ObservationModel:
	def __init__(self,num_features):
		self._mu_t = np.zeros((4,num_features))#Mean positions of all landmarks
		self._sigma_t = np.eye(3*num_features)#variance of each landmark


if __name__ == '__main__':
	filename = "./data/0042.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
	sensor_stream = Sensor(t,linear_velocity,rotational_velocity)
	camera_stream = Camera(features,cam_T_imu,K,b)
	VSLAM = MotionModel()

	world_T_imu = np.zeros((4,4,t.shape[1]))
	# You can use the function below to visualize the robot pose over time

	for i in range(t.shape[1]):
		if i!=0:
			VSLAM.prediction(sensor_stream[i-1][0],sensor_stream[i-1][1])
		#world_T_imu[:,:,i] = VSLAM.mu
		R = VSLAM.mu[:3,:3].T	
		p = np.dot(-VSLAM.mu[:3,:3].T,VSLAM.mu[:-1,-1])
		world_T_imu[:-1,:-1,i] = R
		world_T_imu[:-1,-1,i] = p

	visualize_trajectory_2d(world_T_imu,show_ori=True)

