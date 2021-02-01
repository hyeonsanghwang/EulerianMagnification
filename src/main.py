import numpy as np
import cv2

from utils.camera.video_stream import VideoStream
from utils.visualization.signal import show_signal


CAMERA_NUM = 0

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30


def get_frames(stream, event_key=13):
    processing = False
    frames = []
    while True:
        ret, frame = stream.read()
        if processing:
            frames.append(frame)

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


if __name__ == "__main__":
    # stream = VideoStream(CAMERA_NUM, FPS, FRAME_WIDTH, FRAME_HEIGHT)
    # frames = get_frames(stream, event_key=13)
    # print(np.shape(frames))

    show_signal('aaa', [1,2,3],fps_info=(1,))


