import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
import torch
from playsound import playsound
from pytorch_grad_cam import EigenCAM
from torchvision import transforms

from sit_smart_sensor import SitSmartModel


# get device
def get_accelerator():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif sys.platform == "darwin": # MacOS
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
        self.time_span = time_span
        self.total_sum = 0
        self.deque = deque()

    def get_oldest(self):
        if len(self.deque) == 0:
            return None
        return self.deque[0][0]

    def update(self, new_val):
        current_time = time.time()
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
    def __init__(self, model: SitSmartModel, time_span: int =30, min_samples: int=10, show:bool=True, sound_path: Union[None, str, Path] =None, camera_index : int=0,
                 explain: bool =False, size : Tuple[int,int] = (360, 640), device : str ='auto',sleep_time : int =0, **kwargs):

        self.time_span = time_span
        self.show = show
        self.sleep_time = sleep_time
        self.sound_path = Path(sound_path) if sound_path is not None else None
        if self.sound_path is not None and not self.sound_path.exists():
            raise FileNotFoundError(f"Sound file {self.sound_path} does not exist.")
        self.sound_path = str(self.sound_path) if self.sound_path is not None else None
        if device == 'auto':
            self.device = get_accelerator()
            print(f"Using device {self.device}")
        else:
            self.device = device
        self.rolling_average = RollingAverage(min_samples, time_span)

        self.model = model
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size),
                                             transforms.ToTensor()])

        # number of available cameras
        available_cameras = self._get_num_cameras()
        if len(available_cameras) == 0:
            raise ValueError("No cameras available.")
        if camera_index not in available_cameras:
            raise ValueError(f"Camera index {camera_index} is not available. Available cameras are {available_cameras}")
        self.cap = cv2.VideoCapture(camera_index)

        # explain
        self.explain = explain
        if self.explain:
            train_n_layers = model.train_n_layers

            if train_n_layers == 0:
                target_layers = [list(self.model.backbone.children())[-1]]
            else:
                target_layers = [list(self.model.classifier.children())[train_n_layers - 1]]
            self.cam = EigenCAM(model=self.model, target_layers=target_layers, use_cuda=True)

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

    def run(self):

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
                if self.explain:
                    frame = self.get_explanation(preprocessed_frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR for opencv
                for i, (k, v) in enumerate(probability.items()):
                    text = f'{k.replace("_", " ")}: {int(100 * v)}%'
                    color = (0, int(255 * v), int(255 * (1 - v)))
                    cv2.putText(frame, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                if probability_avg is not None:
                    text_avg = f'Score: {int(100 * probability_avg)}%'
                    color = (0, int(255 * probability_avg), int(255 * (1 - probability_avg)))
                    cv2.putText(frame, text_avg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                cv2.imshow('SitSmart Demo', frame)
            if probability_avg and probability_avg < 0.1 and self.sound_path is not None:
                self.rolling_average.reset()
                playsound(self.sound_path)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time_end = time.time()
            time_elapsed = time_end - time_start
            print(f"Time elapsed: {time_elapsed:.2f} seconds")
            if self.sleep_time > 0:
                time.sleep(self.sleep_time)
        self.cap.release()
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

    def get_explanation(self, frame):
        from pytorch_grad_cam.utils.image import show_cam_on_image

        input_tensor = frame.unsqueeze(0)

        grayscale_cam = self.cam(input_tensor=input_tensor)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        frame = np.einsum('chw -> hwc', frame.numpy())
        visualization = show_cam_on_image(frame, grayscale_cam, use_rgb=True)
        return visualization
