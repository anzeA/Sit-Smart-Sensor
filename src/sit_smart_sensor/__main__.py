import hydra
import torch
from omegaconf import OmegaConf

from .model import SitSmartModel
from .sensor import Sensor
from .train import train_model
from .hyperparameter_optimization import hyperparameter_search
@hydra.main(config_path="config", config_name="config",version_base="1.2")
def main(cfg):
    if cfg.hyperparameter_optimization:
        print('Starting hyperparameter optimization with config:')
        print(OmegaConf.to_yaml(cfg))
        print('Monitor hyperparameter optimization with tensorboard.')
        print("Run: tensorboard  --logdir", cfg.train.log_dir)
        hyperparameter_search(cfg)
        return
    elif cfg.train_model:
        print('Starting training with config:')
        print(OmegaConf.to_yaml(cfg))
        print('Monitor training with tensorboard.')
        print("Run: tensorboard  --logdir", cfg.train.log_dir)
        train_model(cfg)
        return
    print('Load model....')
    model = SitSmartModel(**cfg.model)
    model.load_state_dict(state_dict=torch.load(cfg.run.model_path) ['state_dict'])
    print('Model loaded successfully')
    print('Star sensor....')
    sensor = Sensor(model = model, **cfg.run)
    sensor.run()




if __name__ == '__main__':
    main()