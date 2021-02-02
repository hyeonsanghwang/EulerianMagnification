import numpy as np
import cv2

from utils.camera.video_stream import VideoStream
from utils.processing.dft import band_amplification_filtering


CAMERA_NUM = 0

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30


def get_frames(stream, resize_ratio=1, event_key=13):
    processing = False
    frames = []
    while True:
        ret, frame = stream.read()
        if processing:
            resized = cv2.resize(frame, dsize=None, fx=1.0/resize_ratio, fy=1.0/resize_ratio)
            frames.append(resized)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(stream.delay())
        if key == 27:
            break
        elif key == event_key:
            if processing:
                break
            else:
                processing = True
    cv2.destroyAllWindows()
    stream.release()
    return frames


def eulerian_magnification(frames, fps):
    data = np.array(frames, np.float)
    magnified = band_amplification_filtering(data, 20, band=(0.15, 0.7), fps=fps, axis=0, pad=2)

    for frame in magnified:
        cv2.imshow('Magnified', frame)
        cv2.waitKey(30)


if __name__ == "__main__":
    stream = VideoStream(CAMERA_NUM, FPS, FRAME_WIDTH, FRAME_HEIGHT)
    frames = get_frames(stream, resize_ratio=1, event_key=13)

    eulerian_magnification(frames, FPS)
