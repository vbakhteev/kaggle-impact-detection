import cv2
import jpeg4py
import numpy as np


def collate_fn(batch):
    return tuple(zip(*batch))


def read_img_cv2(filename):
    image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_img_jpeg4py(path):
    img = jpeg4py.JPEG(str(path)).decode()
    return img


def read_frames(video_path, frame_indexes) -> list:
    cap = cv2.VideoCapture(str(video_path))
    images = []

    for i in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images += [frame]

    cap.release()
    return images


def read_video(video_path, each_n_frame=1):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    i = 0

    while True:
        ret = capture.grab()
        if not ret:
            break

        i += 1
        if i == each_n_frame:
            ret, frame = capture.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames += [frame]
            i = 0

    capture.release()
    return np.stack(frames)


def write_video(video, filename, fps=60):
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), fps, (video.shape[2], video.shape[1]))

    for frame in video:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def get_video_len(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(n_frames)