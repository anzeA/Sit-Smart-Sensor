# Image Dataset Structure

This folder organizes the image dataset into three folders:

- **positive**: Images depicting correct sitting posture.
- **negative**: Images displaying incorrect sitting posture.
- **no_person**: Images without any person present or when person is only in the background not looking into a screen. This class is crucial for the model to differentiate when there's no individual in the image. In this way we ensure that sensor is not triggered when there's no person present.
