import json
from copy import deepcopy
from pathlib import Path

import hydra
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from omegaconf import OmegaConf

from .train import train_model

default_cfg = dict()

def update_cfg(parameters):
    global default_cfg
    cfg = deepcopy(default_cfg)
    for k in ["lr", "weight_decay", "model_name","dropout_rate"]:
        cfg['model'][k] = parameters[k]

    for k in ["brightness", "contrast",
              "saturation",
              "hue", "random_gray_scale"]:
        cfg['train_transforms'][k] = parameters[k]
    return cfg
def _train(parameters):
    cfg = update_cfg(parameters)

    cfg.train['save_model_dir'] = None  # don't save model
    cfg.train['enable_progress_bar'] = False  # don't show progress bar

    loss = train_model(cfg)
    return loss

def hyperparameter_search(cfg):
    global default_cfg
    default_cfg = deepcopy(cfg)
    ax_client = AxClient(random_seed=12)

    ax_client.create_experiment(
        name="tune_cnn",  # The name of the experiment.
        parameters=[
            {
                "name": "lr",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [5e-5, 1e-3],  # The bounds for range parameters.
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "weight_decay",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [1e-5, 1e-1],  # The bounds for range parameters.
                "value_type": "float",
                "log_scale": True
            },
            {
                "name": "brightness",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.15],  # The bounds for range parameters.
                "value_type": "float",
            },
            {
                "name": "contrast",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.1],  # The bounds for range parameters.
                "value_type": "float",
            },
            {
                "name": "saturation",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.3],  # The bounds for range parameters.
                "value_type": "float",
            },
            {
                "name": "hue",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.2],  # The bounds for range parameters.
                "value_type": "float",
            },
            {
                "name": "random_gray_scale",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.2],  # The bounds for range parameters.
                "value_type": "float",
            },

            {
                "name": "dropout_rate",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.8],  # The bounds for range parameters.
                "value_type": "float",
            },
            {
                "name": "model_name",  # The name of the parameter.
                "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
                "values": ["resnet18","resnet34","resnet50"], #The possible values for choice parameters .
                "value_type": "str",
             }
        ],
        objectives={cfg.train.monitor: ObjectiveProperties(minimize='loss' in cfg.train.monitor)},

    )

    # Attach the trial
    ax_client.attach_trial(
        parameters=
        {
            # model
            'lr': cfg.model.lr,
            'weight_decay': cfg.model.weight_decay,
            "model_name": cfg.model.model_name,
            "dropout_rate": cfg.model.dropout_rate,
            # data
            'brightness': cfg.train_transforms.brightness,
            'contrast': cfg.train_transforms.contrast,
            'saturation': cfg.train_transforms.saturation,
            'hue': cfg.train_transforms.hue,
            'rotation': cfg.train_transforms.rotation,
            'random_gray_scale': cfg.train_transforms.random_gray_scale,

        }
    )
    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=_train(baseline_parameters))

    for i in range(100):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=_train(parameters))

        best_parameters, values = ax_client.get_best_parameters()
        mean, covariance = values
        print(best_parameters, mean, covariance)
        # save best parameters to json file
        base_path = Path(cfg.train.log_dir)
        dict_to_save = {"best_parameters": best_parameters, "mean": mean}
        path = "best_parameters.json"
        with open(base_path / path, "w") as f:
            json.dump(dict_to_save, f)
        print(f"best parameters saved to {base_path / path}")
        ax_client.save_to_json_file(base_path / "ax_client.json")
        print(f"ax_client saved to {ax_client.save_to_json_file()}")

        # save best parameters to new config file as yaml
        best_cfg = update_cfg(best_parameters)
        with open(base_path /  "best_config.yaml", "w") as f:
            OmegaConf.save(config=best_cfg, f=f)
        print(f"best config saved to {base_path / 'best_config.yaml'}")


        ax_client.get_trials_data_frame().to_csv(base_path / "trials.csv")

