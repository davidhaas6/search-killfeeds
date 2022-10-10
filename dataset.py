import io
from typing import List
import imageio.v3 as imageio
from numpy import ndarray
from pytube import YouTube
from tqdm import tqdm
from sys import argv
import logging
from util import TimeArr

from vision import topright_crop, NNTextDetect
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


def process_video(video_id: str, fs: float, res="1080p"):
    logging.info(f"Pulling {id_to_url(video_id)}")
    clock = TimeArr()
    im_buff, stream = pull_video(video_id, "mp4", res)
    clock.save("pulled video")
    logging.info(f"pulled video in {clock.last():.2f} seconds")

    frames = get_frames(im_buff, "mp4", stream.fps, fs)
    clock.save("got frames")

    cropped = map(preprocess_frame, frames)
    det = NNTextDetect('weights/east_text_detection_weights.pb')

    clock.report(logging.info)
    for img in tqdm(cropped, total=len(frames)):
        txt_boxes = det.detect_text(img)
        det.draw_bboxes(img, txt_boxes, True)
        if len(txt_boxes) == 0:
            continue

        combo_boxes = combine_boxes(txt_boxes)
        det.draw_bboxes(img, combo_boxes, True)
        # cv2.imshow(combo_boxes)



def main():
    logging.basicConfig(level = logging.INFO)

    vid_id = '2hu1bUHL1mc'
    if len(argv) >= 2:
        if argv[1] == "stabby": 
            vid_id = "TYkkWeOzlbM"
        elif argv[1] == "scream":
            vid_id = "H9d2c1oevUU"

    process_video(vid_id, 1)


if __name__ == "__main__":
    main()
