import os
import pickle
from argparse import Namespace
from typing import Dict

import yaml
import torch
import optuna
from optuna.pruners import PatientPruner, MedianPruner

from run_train import run_train
from codes.supports import hps_utils

torch.backends.cudnn.deterministic = True

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

# Used for hyperparameter search with TPE.
class Objective:

    def __init__(
        self, 
        base_args: Namespace, 
        search_space: Dict, 
        save_root: str,
        train_info: str=None
    ) -> None:
        """
        Args:
            base_args (Namespace): Hyper parameters fixed during training.
            search_space (Dict): Dictionary of search space for hyper paramter optimization.
                {"arg_name": [range_type, low, high]}
            save_root (str): 
            train_info (str):
        Returns:
            None
        """
        self.args = base_args
        self.search_space = search_space
        self.train_info = train_info
        self.save_root = save_root

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """
        Args:
            trial (optuna.trial.Trial): 
        Returns:
            best_loss (float):
        """
        params = hps_utils.prepare_params(
            self.args, 
            self.search_space,
            trial
        )

        # Avoid errors from transformer.
        if ("heads" in vars(params) and "emb_dim" in vars(params)):
            if params.heads > params.emb_dim:
                raise optuna.TrialPruned()
        if "qkv_dim" in vars(params):
            if params.heads > params.qkv_dim:
                raise optuna.TrialPruned()
        
        # Run train.
        try:
            best_result_dict, _ = run_train(
                params, self.save_root, trial=trial)
            best_loss = best_result_dict["loss"]
        except Exception as e:
            print(str(e))
            best_loss = 1e5
        return best_loss

def run_hps(
    args: Namespace,
    save_root: str,
    search_space: Dict,
) -> str:
    """
    Args:
        args (Namespace):
        save_root (str): 
        search_space (Dict): Hyperparameter search space.
    Returns:
        csv_name (str):
    """
    assert ("num_trials" in args and "max_time" in args)

    # Prepare result storing directories
    save_loc = os.path.join(save_root, "hps/runs")
    os.makedirs(save_loc, exist_ok=True)

    study = optuna.create_study(
        sampler = optuna.samplers.TPESampler(seed=args.seed),
        direction="minimize",
        pruner=PatientPruner(
            MedianPruner(), 
            patience=5
        ),
    )    
    # catch: Continue when `RuntimeError: CUDA out of memory`.
    # callbacks: Save temporal result
    tmp_saver = hps_utils.TemporalResultSaver(save_loc)
    study.optimize(
        Objective(args, search_space, save_loc), 
        n_trials=args.num_trials, 
        timeout=args.max_time, 
        catch=(RuntimeError,), 
        callbacks=[tmp_saver.save_temporal_result]
    )
    
    # Save study result as csv.
    csv_name = os.path.join(save_loc, "../result.csv")
    df_hps = study.trials_dataframe()
    df_hps.to_csv(csv_name)

    # Save study as pickle.
    pkl_name = os.path.join(save_loc, "../result.pkl")
    with open(pkl_name, "wb") as fp:
        pickle.dump(study, fp)

    return csv_name

if __name__ == "__main__":
    pass