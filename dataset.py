import io
from typing import List
import imageio.v3 as imageio
from numpy import ndarray
from pytube import YouTube
from tqdm import tqdm
from sys import argv
import logging
from util import TimeArr
import cv2
import numpy as np
from collections import defaultdict

from vision import topright_crop, NNTextDetect, get_img_text
from combo_box import combine_boxes


def id_to_url(vid_id):
    return f"https://www.youtube.com/watch?v={vid_id}"


def pull_video(video_id: str, filetype, resolution="720p") -> io.BytesIO:
    buffy = io.BytesIO()
    yt = YouTube(id_to_url(video_id))
    stream = yt.streams.filter(
        file_extension=filetype, res=resolution
    ).first()
    stream.stream_to_buffer(buffy)
    return buffy, stream


def get_frames(buffer, filetype: str, fps: float, sample_rate_hz: float) -> List[ndarray]:
    sample_every = fps // sample_rate_hz
    frames = [
        frame
        for i, frame in enumerate(imageio.imiter(buffer, format_hint="." + filetype))
        if i % sample_every == 0
    ]
    return frames


def preprocess_frame(frame):
    return topright_crop(frame,0.4,0.4)


def extract_kfeeds(img: ndarray, model: NNTextDetect, verbose=False) -> List[ndarray]:
    txt_boxes = model.detect_text(img)
    if len(txt_boxes) == 0:
        return []

    combo_boxes = combine_boxes(txt_boxes)
    killfeed_line_imgs = []
    for bbox in combo_boxes:
        x0,y0,x1,y1 = bbox
        killfeed_line_imgs.append(img[y0:y1,x0:x1,:])

    if verbose: 
        model.draw_bboxes(img, txt_boxes, False, (255,0,0))
        model.draw_bboxes(img, combo_boxes, True)

    return killfeed_line_imgs


def process_video(video_id: str, fs: float, res="1080p", interactive=False) -> List[ndarray]:
    logging.info(f"Pulling {id_to_url(video_id)}")
    im_buff, stream = pull_video(video_id, "mp4", res)
    frames = get_frames(im_buff, "mp4", stream.fps, fs)
    
    # Extract text in the image
    txt_detection_model = NNTextDetect('weights/east_text_detection_weights.pb')
    frame_texts = defaultdict(list)
    kfeed_imgs = []
    for frame_num, img in enumerate(map(preprocess_frame, frames)):
        for kf_img in extract_kfeeds(img, txt_detection_model, interactive):
            text = get_img_text(kf_img)
            kfeed_imgs.append(img)
            frame_texts[frame_num].append(text)

    # store the data
    data = {
        "text": ""
    }
    
    return kfeed_imgs



def main():
    logging.basicConfig(level = logging.INFO)

    vids = {
        "stabby": "TYkkWeOzlbM",
        "scream": "H9d2c1oevUU",
        "eggop": "-BGMI34rAl8",
        "18s": '2hu1bUHL1mc',
    }
    vid_id = vids['eggop']
    if len(argv) > 1:
        vid_id = vids[argv[1]] if argv[1] in vids else argv[1]

    process_video(vid_id, 1, "1080p", True)


if __name__ == "__main__":
    main()
