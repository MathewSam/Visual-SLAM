# Localization and Mapping with IMU and stero camera
The objective of this project is to implement localization and mapping using the IMU data(linear and rotational velocity) and landmark positions from the stereo cameras from left and right cameras
## Data
The data needs to be provided as a .npy file containing time stamps, features, linear velocity, angular velocity, calibration matrix K , baseline b and transformation from cam to imu frame of reference.
## Requirements
<li>
</item>numpy==1.15.4
scipy=1.1.0
matplotlib==3.0.2
argparse==1.1
</li>
## Usage
To run localization and mapping, call the hw3_main.py file from command line 
