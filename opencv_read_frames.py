from vidgear.gears import CamGear
import cv2 
import os 
import math
from util import TimeArr

# Read the video from specified path 
# options = {"STREAM_RESOLUTION": str(x_pixels) + "p"} 

y_pixels = 1080
x_pixels = int(y_pixels * 16/9)
y_off = 300
x_off = 1100

options = {"STREAM_RESOLUTION": str(y_pixels)+"p"}
t = TimeArr()
stream = CamGear(source='https://youtu.be/xaE0ZBMiNKE', stream_mode = True, logging=True,**options).start()
t.save("loaded stream")

# get Video's metadata as JSON object
duration = stream.ytv_metadata["duration"]
framerate = stream.framerate

interval = 1 #seconds
frameskip = math.ceil(interval*framerate)

try: 
    # creating a folder named data 
    if not os.path.exists('demopan'): 
        os.makedirs('demopan') 

# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 

t.save("misc")
# frame 
curr_frame = 0
n_frames = int(duration*framerate)
for curr_frame in range(n_frames):
    # reading from frame 
    frame = stream.read() 
    
    if frame is not None:
        if curr_frame % (interval*frameskip) == 0:
            # if video is still left continue creating images 
            name = './demopan/frame' + str(curr_frame) + '.jpg'
            print ('Creating...' + name) 
            crop_frame = frame[0:int(y_off/1080*y_pixels), int(x_off/1920*x_pixels):x_pixels]
            # writing the extracted images 
            # cv2.imwrite(name, crop_frame)
    else:
        break
t.save("iteration + light processing")
t.report()


# Release all space and windows once done 
stream.stop()