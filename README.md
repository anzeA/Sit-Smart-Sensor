<div align="center">

# Sit Smart Sensor
![](https://github.com/anzeA/Sit-Smart-Sensor/actions/workflows/python-app.yml/badge.svg)

![](https://github.com/anzeA/Sit-Smart-Sensor/blob/main/assets/logo.png)

![](https://github.com/anzeA/Sit-Smart-Sensor/blob/main/assets/demo.gif)
</div>

## Why Sit Smart Sensor?

The Sit Smart Sensor project aims to address the issue of neck and back pain caused by incorrect sitting posture while working in front of a computer. The project offers a solution to help individuals maintain proper posture by providing real-time feedback and alerts to encourage healthy sitting habits.

## Installation

To install Sit Smart Sensor, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/anzeA/Sit-Smart-Sensor.git
    cd Sit-Smart-Sensor
    ```
3. Create virtual environment:
    ```bash
    python -m venv venv
    ```
   or using conda:
    ```bash
    conda create -n venv python=3.11
    ```
if you are having any problems with pytorch installation, try to install it first using conda:
   ```bash
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
4. Install the project using pip:
    ```bash
    pip install -e .
    ```

## Usage

To run Sit Smart Sensor, execute the following command from the root directory:

```bash
python -m sit_smart_sensor
```

Customize the path to the root directory by specifying the `root_dir` parameter:

```bash
python -m sit_smart_sensor root_dir=PATH_TO_ROOT_DIR
```

Replace `PATH_TO_ROOT_DIR` with the desired path to the root directory.

### Default Configuration Parameters for running Sit Smart Sensor

| Parameter Name | Description                                                                                                            |
|----------------|------------------------------------------------------------------------------------------------------------------------|
| `sound_path`   | Path to the sound file played upon detecting bad posture. Supports mp3 and wav formats. If unspecified, no sound will be played. |
| `model_path`   | Location of the model checkpoint file.                                                                                 |
| `time_span`    | Duration in seconds of sustained bad posture to trigger an alert.                                                      |
| `min_samples`  | Minimum number of posture samples within the duration to trigger an alert.                                             |
| `show`         | Set to `True` to display the video feed; otherwise, set to `False`.                                                    |
| `camera_index` | Index of the camera to use.                                                                                            |
| `sleep_time`   | Duration in seconds to pause between each frame processing. Use `0` to disable. Useful for CPU-based systems.          |
 | `device`        | Device to use. Valid values are `auto`, `cpu`, `cuda`, `mps`                                                           |

## Released models
Within the `models` directory, I've included three models available for testing the application. Each model has demonstrated accuracy rates exceeding 90% on my personal dataset.

## Note on Dataset Availability

The dataset used for development is not being released as it mostly contains only images of me and my friends. However, you can create your own dataset by following the instructions below.

### Tips and Tricks for Building Your Own Dataset

- Ensure accurate labeling for each image category!
- Maintain a balanced representation of posture types for effective model training.
- Ensure images are from different locations and backgrounds.
- Ensure to have a variety of lighting conditions.
- Capture various facial expressions.
- Capture various clothing styles.
- Monitor model training with tensorboard!
- Don't be afraid to start building your own dataset! The more images you have, the better the model will perform.
- Images are easy to collect as you can collect them while working on your computer.
- About 100 images per class should be enough to train a model with accuracy over 90%.


## Training
First collect the data. Additional information on how to collect the data can be found in the `data` directory.
To train a model from scratch, run the following command from the root directory:

```bash
python -m sit_smart_sensor train_model=True
```
model checkpoints will be saved in `models` directory. When training is finished, you can use the model by specifying the `model_path` parameter.
For additional configuration parameters, refer to the `src/sit_smart_sensor/config/config.yaml`.

## Hyperparameter Tuning
To tune the hyperparameters of the model, run the following command from the root directory:

```bash
python -m sit_smart_sensor hyperparameter_optimization=True
```
