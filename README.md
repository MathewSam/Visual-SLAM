# Localization and Mapping with IMU and stero camera
The objective of this project is to implement localization and mapping using the IMU data(linear and rotational velocity) and landmark positions from the stereo cameras from left and right cameras
## Data
The data needs to be provided as a .npy file containing time stamps, features, linear velocity, angular velocity, calibration matrix K , baseline b and transformation from cam to imu frame of reference.

## Requirements
<li>
</item>numpy==1.15.4
</item>scipy=1.1.0
</item>matplotlib==3.0.2
</item>argparse==1.1
</li>

## Classes
### SENSOR.py
#### Sensor
This class in SENSOR.py is responsible for handling the velocity information from load_data function. The linear velocity and angular velocity is stacked and returned along with the time discretization for every time step for ease of use.
#### Camera
This class in SENSOR.py is responsible for encapsulating all the data corresponding to the camera mainly the stereo calibration matrix M and the oTi matrix to move from IMU to optical frame. 
### EKF.py
#### MotionModel
The motion model is handled using the MotionModel class in EKF.py. The motion model class represents a gaussian of where the vehicle is with respect to the world. This gaussian is represented using a 4x4 matrix for the mean and a 6x6 matrix for the variance. The 4x4 matrix is an exponential representation of the 6x1 vector representing the pose of the car in the real world. The 6x6 matrix represents the variance of the gaussian in a 6D vector space. The reason the 4x4 representation is used in the motion model is to make the operations easier to carry out in terms of translation, for both the localization and the mapping step. The variance is initialized as a $6x6$ identity matrix.

## Usage
To run localization and mapping, call the hw3_main.py file from command line. ITs usage is as shown below:
> python hw3_main.py data_file
where the data_file corresponds to the location of the .npy file. For help regarding usage of the file, please run
> python hw3_main.py -h  
