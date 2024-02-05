import os
import json
from typing import Tuple
from argparse import Namespace

import torch
from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

from codes.train_model import ModelTrainer
from codes.data.dataloader import prepare_dataloader

from codes.supports.utils import get_timestamp
# sns.set()

class ModelEvaluator(ModelTrainer):

    def __init__(self, args: Namespace, dump_loc: str, device: str) -> None:
        """
        Args:
            args (Namespace):
            dump_loc (str):
            device (str):
        Returns:
            None
        """
        self.args = args
        self.args.device = device

        self.device = device
        self.model = None

        timestamp = get_timestamp()
        self.dump_loc = os.path.join(dump_loc, timestamp)

        os.makedirs(self.dump_loc, exist_ok=True)

    def set_weight(self, weight_file: str):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert (self.model is not None)

        self.model.to("cpu")

        # Temporal solution.
        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict

        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

    def prepare_dataloader(self, load_only_val=False) -> Tuple:
        """
        Args:
            None
        Returns:
            valid_loader (Iterable):
            test_loader (Iterable):
        """

        # Prepare dataloader.
        valid_loader = prepare_dataloader(
            self.args, "val", is_train=False)
        if load_only_val:
            return valid_loader

        test_loader = prepare_dataloader(
            self.args, "test", is_train=False)
        return valid_loader, test_loader

    def run(self, loader, dump_errors=False) -> Tuple[float, float]:
        """
        Args:
            loader
        Returns:
            eval_score (float):
            eval_loss (float):
        """
        result_dict = self._evaluate(loader, dump_errors=dump_errors)
        report = classification_report(
            result_dict["y_trues"], 
            result_dict["y_preds"]>0.5, 
            digits=5, 
            zero_division=0.0
        )
        return result_dict, report

    def dump_target(self, eval_target: str):
        """
        Args:
            eval_target (str):
        Returns:
            None
        """
        with open(self.dump_loc + "/eval_target.txt", "w") as f:
            f.write(eval_target)