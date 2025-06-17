from picamera2.encoders import H264Encoder
from picamera2 import Picamera2
from picamera2.outputs import PyavOutput
import time

# Get both cameras
cameras = Picamera2.global_camera_info()
print("Available cameras:", cameras)

# Create camera objects
cam0 = Picamera2(0)
cam1 = Picamera2(1)

# Configure
config0 = cam0.create_video_configuration()
config1 = cam1.create_video_configuration()

cam0.configure(config0)
cam1.configure(config1)

encoder1 = H264Encoder(bitrate=10000000)
output1 = PyavOutput("test1.mp4")

encoder2 = H264Encoder(bitrate=10000000)
output2 = PyavOutput("test2.mp4")

cam0.start_recording(encoder1, output1)
cam1.start_recording(encoder2, output2)

time.sleep(5)
cam0.stop_recording()
cam1.stop_recording()
