'''
File to handle sensor data stream
'''
import numpy as np

class Sensor:
	'''
	Class to manage sensor readings for vehicle
	'''
	def __init__(self,t,linear_velocity,rotational_velocity):
		'''
		Initializes class to handle stream of data from sensor
		'''
		self.time_diff = t[0,1:] - t[0,:-1]
		self.velocity = np.vstack([linear_velocity,rotational_velocity])
		
	def __getitem__(self,index):
		'''
		'''	
		return self.time_diff[index],self.velocity[:,index]

class Camera:
	'''
	Class to handle camera observations from vehicle to assist in observation model
	'''
	def __init__(self,features,cam_T_imu,K,b):
		'''
		Initializes the camera model for operations
		'''
		self.features = features
		self._M = np.zeros((4,4))
		self._M[:2,0:-1] = K[:-1,:]
		self._M[2:,0:-1] = K[:-1,:]
		self._M[2,-1] = -b*K[0,0]
		self._oTi = cam_T_imu
	
	def __getitem__(self,index):
		obs_features = self.features[:,:,index]
		filter_condn = np.logical_or(np.logical_or(obs_features[0]!=-1,obs_features[1]!=-1),np.logical_or(obs_features[2]!=-1,obs_features[3]!=-1))
		return obs_features[:,filter_condn]

	@property
	def M(self):
		return self._M

	@property
	def oTi(self):
		return self._oTi 