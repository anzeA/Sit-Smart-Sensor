from collections import deque
import time
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from playsound import playsound
from model import SitSmartModel
class RollingAverage:
    def __init__(self, time_span):
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
        any_removed = False
        # Remove elements older than time_span seconds
        while self.deque and current_time - self.deque[0][1] > self.time_span:
            removed_val, _ = self.deque.popleft()
            self.total_sum -= removed_val
            any_removed = True
        # Calculate the rolling average
        if not any_removed:
            return None
        rolling_average = self.total_sum / len(self.deque) if self.deque else 0
        return rolling_average

    def reset(self):
        self.total_sum = 0
        self.deque.clear()

class Sensor:
    def __init__(self,model_path, time_span=1,show=True,sound_path=None,camera_index=0):
        self.time_span = time_span
        self.show = show
        self.sound_path = Path(sound_path) if sound_path is not None else None
        if self.sound_path is not None and not self.sound_path.exists():
            raise FileNotFoundError(f"Sound file {self.sound_path} does not exist.")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        self.rolling_average = RollingAverage(time_span)
        self.model = SitSmartModel(None,None)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict=state_dict['state_dict'])
        self.model.eval()
        self.transform =  transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                transforms.ToTensor()])
        # number of available cameras
        available_cameras = self._get_num_cameras()
        if len(available_cameras) == 0:
            raise ValueError("No cameras available.")
        if camera_index not in available_cameras:
            raise ValueError(f"Camera index {camera_index} is not available. Available cameras are {available_cameras}")
        self.cap = cv2.VideoCapture(camera_index)
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
            frame = self.get_image()
            if frame is None:
                print("Failed to grab frame.")
                break
            preprocessed_frame = self.preprocess_image(frame)
            probability = self.predict_image(preprocessed_frame)
            probability_avg = self.update(probability)
            if self.show:
                text = f'Probability: {int(100*probability)}%'
                color = (0, int(255*probability), int(255*(1-probability)) )
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2)

                if probability_avg is not None:
                    text_avg = f'Score: {int(100*probability_avg)}%'
                    color = (0, int(255 * probability), int(255 * (1 - probability)))
                    cv2.putText(frame, text_avg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow('SitSmart Demo', frame)
            if probability_avg and probability_avg < 0.1 and self.sound_path is not None:
                # for playing note.wav file
                import threading
                threading.Thread(target=playsound, args=(self.sound_path,), daemon=True).start()
                self.rolling_average.reset()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


    def update(self, new_val):
        return self.rolling_average.update(new_val)

    def preprocess_image(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform(frame)
        return frame

    # Create a function to make predictions
    def predict_image(self,frame:torch.Tensor):
        # print prediction time
        with torch.no_grad():
            frame = frame.unsqueeze(0)
            probability = self.model(frame)
            return probability.item()

    def get_image(self):

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame.")
            self.cap.release()
            return None

        return frame
