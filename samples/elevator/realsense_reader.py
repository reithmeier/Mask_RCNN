# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

print("version:")
print(rs)
print(cv2.getVersionString())
print(np.__version__)
print("---")

pipe = rs.pipeline()
profile = pipe.start()
try:
  for i in range(0, 100):
    frames = pipe.wait_for_frames()
    for f in frames:
      print(f.profile)
finally:
    pipe.stop()

print("---")
