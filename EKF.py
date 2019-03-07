'''
File to handle classes associated with EKF. The file has a separate class for the motion model and Observatin model.  
'''
import numpy as np
from scipy.linalg import expm

class MotionModel:
	'''
	Defines the EKF filter and associated functionality
	'''
	def __init__(self):
		'''
		Initializes class
		Args:
			self:pointer to current instance of the class
		'''
		self._mu = np.eye(4)#On the assumption that the object is currently at the origin with 0 for roll pitch and yaw
		self._sigma = np.eye(6)#Assuming unit matrix as covariance matrix at first time step

	@staticmethod
	def _hat_map(vector):
		'''
		Generates hat map of provided vector
		'''
		hat = np.zeros((3,3))

		hat[0,1] = -vector[2]
		hat[0,2] = vector[1]
		hat[1,0] = vector[2]
		hat[1,2] = -vector[0]
		hat[2,0] = -vector[1]
		hat[2,1] = vector[0]

		return hat


	def prediction(self,time_diff,velocity):
		'''
		Defines motion model given linear velocity, rotational velocity and time difference between readings
		Args:
			self: pointer to current instance of the class
			time_diff: time difference between consecutive readings
			velocity: combined velocity vector from both the rotation and linear 
		'''
		rho = velocity[:3]
		theta = velocity[3:]
		u_hat = np.zeros((4,4))
		u_hat[:-1,:-1] = self._hat_map(theta)
		u_hat[:-1,-1] = rho
		self._mu = np.dot(expm(-time_diff*u_hat),self._mu)

		u_spike = np.zeros((6,6))
		u_spike[:3,:3] = self._hat_map(theta)
		u_spike[:3,3:] = self._hat_map(rho)
		u_spike[3:,3:] = self._hat_map(theta)
		left = expm(-time_diff*u_spike)
		self._sigma = np.dot(np.dot(left,self._sigma),left.T) + np.random.randn(6,6)

	@property
	def mu(self):
		return self._mu

	@property
	def sigma(self):
		return self._sigma