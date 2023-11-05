import json
from copy import deepcopy
from pathlib import Path

import hydra
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from train import train_model

default_cfg = dict()


def train_and_evaluate(parameters):
    cfg = deepcopy(default_cfg)
    cfg['model']['lr'] = parameters['lr']
    cfg['model']['weight_decay'] = parameters['weight_decay']
    for k in ["brightness", "contrast",
              "saturation",
              "hue"]:
        cfg['dataset'][k] = parameters[k]
    cfg['save_model_dir'] = None # don't save model
    cfg['enable_progress_bar'] = False # don't show progress bar
    # return parameters['lr'] + parameters['weight_decay']
    loss = train_model(cfg)
    print(f"loss: {loss}")
    return loss


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    global default_cfg
    default_cfg = deepcopy(cfg)
    ax_client = AxClient(random_seed=12)

    ax_client.create_experiment(
        name="tune_cnn",  # The name of the experiment.
        parameters=[
            {
                "name": "lr",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [1e-5, 1e-1],  # The bounds for range parameters.
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
                "bounds": [0, 0.3],  # The bounds for range parameters.
                "value_type": "float",
            },
            {
                "name": "contrast",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 0.3],  # The bounds for range parameters.
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
                "bounds": [0, 0.3],  # The bounds for range parameters.
                "value_type": "float",
            },
            {
                "name": "rotation",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [0, 20],  # The bounds for range parameters.
                "value_type": "int",
            },
        ],
        objectives={"val_loss": ObjectiveProperties(minimize=True)},  # The objective name and minimization setting.
        # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
        # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
    )

    # Attach the trial
    ax_client.attach_trial(
        parameters=
        {'lr': 0.017009560876162722, 'weight_decay': 4.419591262477975e-05, 'brightness': 0.015573415160179137,
         'contrast': 0.20431235432624817, 'saturation': 0.1468503624200821, 'hue': 0.010385657474398612, 'rotation': 17}
    )
    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=train_and_evaluate(baseline_parameters))

    for i in range(1000):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=train_and_evaluate(parameters))

        best_parameters, values = ax_client.get_best_parameters()
        mean, covariance = values
        print(best_parameters, mean, covariance)
        # save best parameters to json file
        base_path = Path(cfg.log_dir)
        dict_to_save = {"best_parameters": best_parameters, "mean": mean}
        path = "best_parameters.json"
        with open(base_path / path, "w") as f:
            json.dump(dict_to_save, f)
        print(f"best parameters saved to {base_path / path}")
        ax_client.save_to_json_file(base_path / "ax_client.json" )
        print(f"ax_client saved to {ax_client.save_to_json_file()}")

        ax_client.get_trials_data_frame().to_csv(base_path / "trials.csv")


if __name__ == '__main__':
    main()
