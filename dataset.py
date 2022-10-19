import io
import json
import os
from typing import Callable, List
import imageio.v3 as imageio
from numpy import ndarray
from pytube import YouTube
from tqdm import tqdm
from sys import argv
import logging
from util import TimeArr
import cv2
import pandas as pd

from vision import topright_crop, NNTextDetect, get_img_text, TrOCR, thresholdize
from combo_box import combine_boxes


def id_to_url(vid_id):
    return f"https://www.youtube.com/watch?v={vid_id}"


def pull_video(video_id: str, filetype, resolution="720p") -> io.BytesIO:
    buffy = io.BytesIO()
    yt = YouTube(id_to_url(video_id))
    stream = yt.streams.filter(file_extension=filetype, res=resolution).first()
    stream.stream_to_buffer(buffy)
    return buffy, stream


def get_frames(
    buffer, filetype: str, fps: float, sample_rate_hz: float
) -> List[ndarray]:
    sample_every = fps // sample_rate_hz
    frames = [
        frame
        for i, frame in enumerate(imageio.imiter(buffer, format_hint="." + filetype))
        if i % sample_every == 0
    ]
    return frames


def preprocess_frame(frame):
    # frame = thresholdize(frame)
    return topright_crop(frame, 0.4, 0.4)


def extract_kfeeds(img: ndarray, model: NNTextDetect, verbose=False) -> List[ndarray]:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    txt_boxes = model.detect_text(img)
    if len(txt_boxes) == 0:
        return []

    # merge text boxes
    combo_boxes = combine_boxes(txt_boxes)
    killfeed_line_imgs = []
    for bbox in combo_boxes:
        x0, y0, x1, y1 = bbox
        killfeed_line_imgs.append(img[y0:y1, x0:x1, :])

    if verbose:
        model.draw_bboxes(img, txt_boxes, False, (255, 0, 0))
        model.draw_bboxes(img, combo_boxes, True)

    return killfeed_line_imgs


def process_video(
    video_id: str,
    fs_hz: float,
    ocr: Callable[[ndarray], str],
    res="1080p",
    out_dir=None,
    txt_det_model = NNTextDetect("weights/east_text_detection_weights.pb")
) -> List[ndarray]:

    logging.info(f"Pulling {id_to_url(video_id)}")
    if type(out_dir) != str or not os.path.isdir(out_dir):
        out_dir = None
        logging.warning("invalid directory - not writing")

    im_buff, stream = pull_video(video_id, "mp4", res)
    frames = get_frames(im_buff, "mp4", stream.fps, fs_hz)
    preprocessed = map(preprocess_frame, frames)

    # Extract text in the image
    data = []
    for frame_num, img in tqdm(
        enumerate(preprocessed), total=len(frames), desc="Performing OCR..."
    ):
        cur_second = frame_num / fs_hz
        ocr_strings = []
        for i, kf_img in enumerate(extract_kfeeds(img, txt_det_model, False)):
            try:
                img_text = ocr(kf_img)
            except Exception as e:
                print(f"Error '{e}' with OCR for frame {frame_num} line {i}")

            if len(img_text.strip()) > 0:
                ocr_strings.append(img_text)

            if out_dir is not None:
                if len(kf_img.shape) == 2:
                    kf_img = cv2.cvtColor(kf_img, cv2.COLOR_GRAY2RGB)
                fname = f"{video_id}_frame{frame_num}-{i}.png"
                cv2.imwrite(os.path.join(out_dir, fname), kf_img)
            
        if len(ocr_strings) > 0:
            data.append(
                {
                    "video_id": video_id,
                    "timestamp_s": cur_second,
                    "frame": frame_num,
                    "identified_text": ocr_strings,
                }
            )

    if out_dir is not None:
        with open(os.path.join(out_dir, "out.json"), "w") as f:
            f.write(json.dumps(data))

    return data


def main():
    logging.basicConfig(level=logging.INFO)

    vids = {
        "stabby": "TYkkWeOzlbM",
        "scream": "H9d2c1oevUU",
        "eggop": "-BGMI34rAl8",
        "18s": "2hu1bUHL1mc",
    }
    vid_id = vids["eggop"]
    if len(argv) > 1:
        vid_id = vids[argv[1]] if argv[1] in vids else argv[1]
    # trocr = TrOCR()
    vid_data = process_video(vid_id, 1, get_img_text, "1080p", "./test-data/unlabeled_dataset/")


if __name__ == "__main__":
    main()
