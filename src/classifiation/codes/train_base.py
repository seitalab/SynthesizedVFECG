from argparse import Namespace
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from optuna.trial import Trial

from codes.supports.storer import Storer
from codes.data.dataloader import prepare_dataloader
from codes.architectures.model import prepare_model

class BaseTrainer:

    def __init__(
        self,
        args: Namespace,
        save_dir: str,
        mode: str="min"
    ) -> None:
        """
        Args:
            args (Namespace):
            save_dir (str): Directory to output model weights and parameter info.
            mode (str): 
        Returns:
            None
        """

        self.args = args

        self.storer = Storer(save_dir)
        self.model = None
        
        
        assert mode in ["max", "min"]
        self.mode = mode
        self.flip_val = -1 if mode == "max" else 1

        self.best_result = None
        self.best_val = np.inf * self.flip_val # Overwritten during training.

    def prepare_dataloader(self) -> Tuple[Iterable, Iterable]:
        """
        Args:
            None
        Returns:
            train_loader (Iterable):
            valid_loader (Iterable):
        """

        # Prepare dataloader.
        train_loader = prepare_dataloader(
            self.args, "train", is_train=True)
        valid_loader = prepare_dataloader(
            self.args, "val", is_train=False)
        return train_loader, valid_loader

    def set_lossfunc(self, weights=None) -> None:
        """
        Args:
            weights (Optional[np.ndarray]): 
        Returns:
            None
        """
        assert self.model is not None

        if weights is not None:
            weights = torch.Tensor(weights)
        
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
        self.loss_fn.to(self.args.device)

    def set_optimizer(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        assert self.model is not None

        if self.args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        elif self.args.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        elif self.args.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        else:
            raise NotImplementedError
        
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 
        #     patience=self.args.optimizer_patience, 
        #     factor=0.2
        # )

    def set_model(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        model = prepare_model(self.args)
        model = model.to(self.args.device)

        self.model = model

    def set_trial(self, trial: Trial) -> None:
        """
        Args:
            trial (Trial): Optuna trial.
        Returns:
            None
        """
        self.trial = trial

    def save_params(self) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        self.storer.save_params(self.args)

    def show_model_info(self) -> None:
        """
        Show overview of model.

        Args:
            None
        Returns:
            None
        """
        assert self.model is not None

        # seqlen = int(self.args.freq * self.args.length)
        # input_size = (self.args.bs, self.args.num_lead, seqlen)
        # summary(self.model, input_size=input_size, device=self.args.device)

    def get_best_loss(self) -> float:
        """
        Args:
            None
        Returns:
            best_value (float):
        """
        return self.best_val, self.best_result

    def _train(self, iterator: Iterable):
        raise NotImplementedError

    def _evaluate(self, iterator: Iterable):
        raise NotImplementedError

    def run(self, train_loader: Iterable, valid_loader: Iterable):
        raise NotImplementedError