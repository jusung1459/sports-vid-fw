import cv2
import time
import numpy as np
from picamera2.encoders import H264Encoder
from picamera2 import Picamera2
from picamera2.outputs import PyavOutput

# === Initialize Cameras ===
cam0 = Picamera2(0)
cam1 = Picamera2(1)

config0 = cam0.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"})
config1 = cam1.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"})

cam0.configure(config0)
cam1.configure(config1)

encoder1 = H264Encoder(bitrate=10000000)
output1 = PyavOutput("test1.mp4")
encoder2 = H264Encoder(bitrate=10000000)
output2 = PyavOutput("test2.mp4")

cam0.start_recording(encoder1, output1)
cam1.start_recording(encoder2, output2)

# === Prepare Output Writer ===
output_size = (3500, 1080)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('panorama_output.mp4', fourcc, 10.0, output_size)
if not out.isOpened():
    raise RuntimeError("Failed to open panorama_output.mp4")

# === Auto-Calibration Function ===
def compute_homography(f_left, f_right, ratio):
    descriptor  = cv2.SIFT_create()
    kp1, des1 = descriptor.detectAndCompute(f_right, None)
    kp2, des2 = descriptor.detectAndCompute(f_left, None)

    if des1 is None or des2 is None:
        return None

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(des1, des2, 2)
    # matches = matcher.match(des1, des2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
           matches.append((m[0].trainIdx, m[0].queryIdx))

    print(len(matches), "matches found")

    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kp1[i].pt for (_, i) in matches])
        ptsB = np.float32([kp2[i].pt for (i, _) in matches])

    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)
    return H

# === Auto-Calibration Process ===
print("Auto-calibrating... Please hold cameras still.")
time.sleep(2)  # Let camera warm up
H_list = []
MAX_ATTEMPTS = 4

for i in range(MAX_ATTEMPTS):
    time.sleep(0.3)
    frame_left = cam0.capture_array("main")
    frame_right = cam1.capture_array("main")

    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2GRAY)
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2GRAY)

    H = compute_homography(frame_left, frame_right, 0.40)
    if H is not None:
        H_list.append(H)
        print(f"Homography found ({len(H_list)})")

    if len(H_list) >= MAX_ATTEMPTS-1:
        H_avg = np.mean(H_list, axis=0)
        print("Homography stabilized.")
        break
else:
    raise RuntimeError("Auto-calibration failed. Check camera alignment.")

# === Main Loop: Stitching and Recording ===
frame_count = 0
try:
    print("Recording and stitching frames...")
    while True:
        f_left = cam0.capture_array("main")
        f_right = cam1.capture_array("main")

        f_left = cv2.cvtColor(f_left, cv2.COLOR_RGB2BGR)
        f_right = cv2.cvtColor(f_right, cv2.COLOR_RGB2BGR)

        # Warp right image into left camera’s perspective
        warped_right = cv2.warpPerspective(f_right, H_avg, (f_left.shape[1] * 2, f_left.shape[0]))

        # Overlay left image
        warped_right[0:f_left.shape[0], 0:f_left.shape[1]] = f_left

        # Crop to output size
        pano = warped_right[:, :output_size[0]]
        pano_resized = cv2.resize(pano, output_size)
        out.write(pano_resized)

        frame_count += 1
        if frame_count >= 10:
            break

except KeyboardInterrupt:
    print("Stopped by Ctrl+C.")

finally:
    cam0.stop()
    cam1.stop()
    out.release()
    print("Done. Saved to panorama_output.mp4")

