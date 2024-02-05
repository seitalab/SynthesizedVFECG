import os
from typing import Tuple, Optional
from argparse import Namespace

import torch
import numpy as np
from optuna.trial import Trial

from codes.supports import utils
from codes.train_model import ModelTrainer

torch.backends.cudnn.deterministic = True

def run_train(
    args: Namespace, 
    save_root: str,
    trial: Optional[Trial]=None
) -> Tuple[float, str]:
    """
    Execute train code for ecg classifier
    Args:
        args (Namespace): Namespace for parameters used.
        save_root (str): 
    Returns:
        best_val_loss (float): 
        save_dir (str):
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare result storing directories
    timestamp = utils.get_timestamp()
    save_setting = f"{timestamp}-{args.host}"
    save_dir = os.path.join(
        save_root, 
        save_setting
    )

    # Trainer prep
    trainer = ModelTrainer(args, save_dir)
    trainer.set_trial(trial)
    trainer.set_model()

    print("Preparing dataloader ...")
    train_loader, valid_loader = trainer.prepare_dataloader()

    weight = utils.calc_class_weight(train_loader.dataset.label)
    trainer.set_lossfunc(weight)
    trainer.set_optimizer()
    trainer.save_params()

    print("Starting training ...")
    trainer.run(train_loader, valid_loader)
    _, best_result = trainer.get_best_loss()

    del trainer

    # Return best validation loss when executing hyperparameter search.
    return best_result, save_dir

if __name__ == "__main__":
    pass
    