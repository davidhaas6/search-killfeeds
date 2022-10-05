import io
from typing import List
import imageio.v3 as imageio
from numpy import ndarray
from pytube import YouTube
from tqdm import tqdm
from sys import argv

from vision import crop_topright_quadrant


def pull_video(video_id: str, filetype, resolution="720p") -> io.BytesIO:
    buffy = io.BytesIO()
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    stream = yt.streams.filter(
        file_extension=filetype, res=resolution, progressive="false"
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
    return crop_topright_quadrant(frame)


def main():
    ftype = 'mp4'
    res = '720p'
    vid_id = argv[1] if len(argv) >= 2 else 'A4rahpCPX9Q'

    im_buff, stream = pull_video(vid_id, ftype, res)
    frames = get_frames(im_buff, ftype, stream.fps, 1)
    cropped = map(preprocess_frame, frames)

    for i, f in tqdm(enumerate(cropped)):
        imageio.imwrite(f"./test-data/out/{i}.png", f, extension=".png")


if __name__ == "__main__":
    main()
