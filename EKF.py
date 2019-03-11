'''
File to handle classes associated with EKF. The file has a separate class for the motion model and Observatin model.  
'''
import numpy as np
from scipy.linalg import expm

def invert_pose(pose):
	'''
	Returns the invers of the pose 
	Args:
		pose: pose to be inverted
	Returns:
		pose_inv: inverse of pose 
	'''
	pose_inv = np.zeros((4,4))
	pose_inv[:3,:3] = pose[:3,:3].T.copy() 
	pose_inv[:-1,-1] = -np.dot(pose[:3,:3].T,pose[:-1,-1])
	pose_inv[-1,-1] = 1
	return pose_inv

def dpi_dq(q):
	'''
	Differential of scaling matrix
	Args:
		q: input vector. Must be of shape 4,
	Returns:
		pi: differential of homogenous scaling function with respect to input
	'''
	assert q.shape == (4,),"Must be of shape (4,)"
	pi = np.eye(4)
	pi[:,2] = -q/q[2]
	pi[2,2] = 0
	return pi/q[2]

def pi_q(q):
	'''
	Homogenous scaling of vector
	Args:
		q: vector to scale
	Returns:
		pi: scaled vector
	'''
	assert q.shape == (4,),"Must be of shape (4,)"
	return q/q[2]

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

class Landmark:
	def __init__(self,M,oTi):
		'''
		'''
		self.observed =  False
		self._mu = np.zeros((4,))
		self._mu[-1] = 1
		self._sigma = np.eye(3,3)

		self.M = M
		self.oTi = oTi
		self.D = np.vstack([np.eye(3),np.zeros((1,3))])

	def update_landmark_position(self,z,pose):
		'''
		Updates location of landmark given feature location.
		Args:
			self:pointer to current instance of the class
			z:Observed feature location
			pose: position of observer/camera in world frame
			M:
			oTi:
		'''
		if not self.observed:
			v = np.dot(np.linalg.pinv(self.M),z)
			v = v/v[3]
			self._mu = np.dot(np.linalg.pinv(np.dot(self.oTi,pose)),v)
			self.observed = True
		else:
			projection = np.dot(self.oTi,np.dot(pose,self._mu))
			z_hat = np.dot(self.M,pi_q(projection))

			H = np.dot(np.dot(self.M,dpi_dq(projection)),np.dot(np.dot(self.oTi,pose),self.D))
			K = np.dot(np.dot(self._sigma,H.T),np.linalg.inv(np.dot(np.dot(H,self._sigma),H.T) + 0.0005*np.eye(4)))
			self._mu = self._mu + np.dot(np.dot(self.D,K),(z - z_hat))
			self._sigma = np.dot(np.eye(3) - np.dot(K,H),self._sigma)

	@property
	def mu(self):
		return self._mu