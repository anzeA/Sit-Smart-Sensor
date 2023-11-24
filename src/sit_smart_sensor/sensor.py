import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Tuple

import cv2
import torch
from torchvision import transforms

from sit_smart_sensor import SitSmartModel
from notifypy import Notify

# get device
def get_accelerator():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif sys.platform == "darwin":  # MacOS
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")

    return torch.device('cpu')


class RollingAverage:
    def __init__(self, min_samples, time_span):
        self.min_samples = min_samples
        # time_span in seconds
        # make it timedelta
        if isinstance(time_span, int):
            time_span = timedelta(seconds=time_span)
        self.time_span = time_span
        self.total_sum = 0
        self.deque = deque()

    def get_oldest(self):
        if len(self.deque) == 0:
            return None
        return self.deque[0][0]

    def update(self, new_val):
        current_time = datetime.now()
        self.total_sum += new_val
        self.deque.append((new_val, current_time))
        # Remove elements older than time_span seconds
        while self.deque and current_time - self.deque[0][1] > self.time_span:
            removed_val, _ = self.deque.popleft()
            self.total_sum -= removed_val
        # Calculate the rolling average
        if len(self.deque) < self.min_samples:
            return None
        rolling_average = self.total_sum / len(self.deque) if self.deque else 0
        return rolling_average

    def reset(self):
        self.total_sum = 0
        self.deque.clear()


class Sensor:
    def __init__(self, model: SitSmartModel, time_span: int = 30, min_samples: int = 10, show: bool = True,
                 camera_index: Union[int, None] = 0,
                 size: Tuple[int, int] = (360, 640), device: str = 'auto', sleep_time: int = 0,
                 **kwargs):
        self.time_span = time_span
        self.show = show
        self.sleep_time = sleep_time

        if device == 'auto':
            self.device = get_accelerator()
            print(f"Using device {self.device}")
        else:
            self.device = device
        self.rolling_average = RollingAverage(min_samples, time_span)

        self.model = model
        self.model.eval()
        self.model.to(self.device)
        self.size = size
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size),
                                             transforms.ToTensor()])

        if camera_index is not None:
            try:
                self.cap = cv2.VideoCapture(camera_index)
            except:
                available_cameras = self._get_num_cameras()
                if len(available_cameras) == 0:
                    raise ValueError("No cameras available.")
                raise ValueError(
                    f"Camera index {camera_index} is not available. Available cameras are {available_cameras}")
        else:
            self.cap = None

    def _get_num_cameras(self):
        index = 0
        arr = set()
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.add(index)
            cap.release()
            index += 1
        return arr
    def _add_text_to_image(self, frame, probability, probability_avg=None):
        for i, (k, v) in enumerate(probability.items()):
            text = f'{k.replace("_", " ")}: {int(100 * v)}%'
            color = (0, int(255 * v), int(255 * (1 - v)))
            cv2.putText(frame, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if probability_avg is not None:
            text_avg = f'Score: {int(100 * probability_avg)}%'
            color = (0, int(255 * probability_avg), int(255 * (1 - probability_avg)))
            cv2.putText(frame, text_avg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def run(self):
        if self.cap is None:
            raise ValueError("No camera selected. Set camera_index to a valid camera index. Try with 0")
        while True:
            time_start = time.time()
            frame = self.get_image()
            if frame is None:
                print("Failed to grab frame.")
                break
            preprocessed_frame = self.preprocess_image(frame)
            probability = self.predict_image(preprocessed_frame)

            probability_avg = None
            probability_pos = probability['positive'] / (probability['positive'] + probability['negative'] + 1e-9)
            if probability['no_person'] < 0.25:
                probability_avg = self.update(probability_pos)
            if self.show:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR for opencv
                self._add_text_to_image(frame, probability, probability_avg)

                cv2.imshow('SitSmart Demo', frame)
            if probability_avg and probability_avg < 0.1:
                self.rolling_average.reset()
                self.send_notification()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time_end = time.time()
            time_elapsed = time_end - time_start
            score_str = f"Score: {int(100 * probability_avg)}%" if probability_avg else f'Score is not yet available. Currently have {len(self.rolling_average.deque)} out of the required {self.rolling_average.min_samples} samples.'
            print(f"Time elapsed: {time_elapsed:.2f} seconds. FPS: {1 / time_elapsed:.2f}. {score_str}")
            if self.sleep_time > 0:
                time.sleep(self.sleep_time)
        self.cap.release()
        cv2.destroyAllWindows()

    def run_on_video(self, video_path: Union[str, Path], output_path: Union[str, Path, None] = None, **kwargs):
        print('Running on video')
        print(f'Video path: {video_path}')
        if output_path is not None:
            print(f'Output path: {output_path}')
        else:
            print('Output path is not set.')
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Error opening video file. Make sure ffmpeg is installed.")

        if output_path is not None:
            h, w = self.size
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 8., size)
        if self.show:
            print('Show is set to True, but it is ignored when running on video.')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
            preprocessed_frame = self.preprocess_image(frame)
            probability = self.predict_image(preprocessed_frame)

            probability_avg = None
            probability_pos = probability['positive'] / (probability['positive'] + probability['negative'] + 1e-9)
            if probability['no_person'] < 0.25:
                probability_avg = self.update(probability_pos)
            # resize frame
            frame = cv2.resize(frame, self.size[::-1])
            self._add_text_to_image(frame, probability, probability_avg)
            # frame to BGR for opencv
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if output_path is not None:
                out.write(frame)
        if output_path is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()

    def update(self, new_val):
        return self.rolling_average.update(new_val)

    def get_image(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame.")
            self.cap.release()
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
        return frame

    def preprocess_image(self, frame):
        frame = self.transform(frame)
        return frame

    # Create a function to make predictions
    def predict_image(self, frame: torch.Tensor):
        # print prediction time
        with torch.no_grad():
            frame = frame.unsqueeze(0).to(self.device)
            probability = self.model.predict_proba(frame).cpu().numpy()

        return {'negative': probability[0][0], 'no_person': probability[0, 1], 'positive': probability[0][2]}

    def send_notification(self):
        notification = Notify(disable_logging=True)
        notification.application_name = 'Sit Smart Sensor'
        notification.title = 'Incorrect Posture'
        notification.message = 'You are sitting incorrectly. Please correct your posture.'
        notification.icon ='assets/logo.ico'
        notification.send(block=False)
