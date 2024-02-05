import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

from optuna.exceptions import TrialPruned

from codes.train_base import BaseTrainer
from codes.supports.monitor import Monitor, EarlyStopper

class ModelTrainer(BaseTrainer):

    def _train(self, loader) -> Tuple[float, float]:
        """
        Run train mode iteration.
        Args:
            loader:
        Returns:
            score (float):
            loss (float): 
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(loader):

            self.optimizer.zero_grad()
            X = X.to(self.args.device).float()
            y = y.to(self.args.device).float()
            pred_y = self.model(X)

            minibatch_loss = self.loss_fn(pred_y, y)

            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss) * len(X))
            monitor.store_num_data(len(X))
            monitor.store_result(y, pred_y)

        monitor.show_per_class_result()
        result_dict = {
            "score": monitor.macro_f1(), 
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record
        }
        return result_dict
        
    def _evaluate(
        self, 
        loader, 
        dump_errors: bool=False
    ) -> Tuple[float, float]:
        """
        Args:
            loader :
        Returns:
            score (float):
            loss (float): 
        """
        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():

            for X, y in tqdm(loader):
                
                X = X.to(self.args.device).float()
                y = y.to(self.args.device).float()

                pred_y = self.model(X)

                minibatch_loss = self.loss_fn(pred_y, y)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                monitor.store_result(y, pred_y)
                if dump_errors:
                    monitor.store_input(X)

        monitor.show_per_class_result()

        if dump_errors:
            monitor.dump_errors(self.dump_loc, dump_type="fp")
            monitor.dump_errors(self.dump_loc, dump_type="fn")
            monitor.dump_errors(self.dump_loc, dump_type="tp")
            monitor.dump_errors(self.dump_loc, dump_type="tn")
        result_dict = {
            "score": monitor.macro_f1(),
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record,
            # "file_idxs": monitor.f_idxs
        }            
        return result_dict

    def run(self, train_loader, valid_loader) -> None:
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
            mode (str): definition of best (min or max).
        Returns:
            None
        """
        self.best = np.inf * self.flip_val # Sufficiently large or small
        if self.trial is None:
            early_stopper = EarlyStopper(
                mode=self.mode, patience=self.args.patience)

        for epoch in range(1, self.args.epochs + 1):
            print("-" * 80)
            print(f"Epoch: {epoch:03d}")
            train_result = self._train(train_loader)
            self.storer.store_epoch_result(
                epoch, train_result, is_eval=False)

            if epoch % self.args.eval_every == 0:
                eval_result = self._evaluate(valid_loader)
                self.storer.store_epoch_result(
                    epoch, eval_result, is_eval=True)

                if self.mode == "max":
                    monitor_target = eval_result["score"]
                    # self.scheduler.step(eval_result["score"])
                else:
                    monitor_target = eval_result["loss"]
                    # self.scheduler.step(eval_result["loss"])

                # Use pruning if hyperparameter search with optuna.
                # Use early stopping if not hyperparameter search (= trial is None).
                if self.trial is not None:
                    self.trial.report(monitor_target, epoch)
                    if self.trial.should_prune():
                        raise TrialPruned()
                else:
                    if early_stopper.stop_training(monitor_target):
                        break

                self._update_best_reesult(monitor_target, eval_result)

            self.storer.store_logs()

    def _update_best_reesult(self, monitor_target, eval_result):
        """
        Args:

        Returns:
            None
        """
        
        if monitor_target * self.flip_val < self.best_val * self.flip_val:
            print(f"Val metric improved {self.best_val:.4f} -> {monitor_target:.4f}")
            self.best_val = monitor_target
            self.best_result = eval_result
            self.storer.save_model(self.model, monitor_target)
            # self.storer.save_plot(
            #     train_result["y_trues"], 
            #     train_result["y_preds"], 
            #     "train",
            #     self.args.multitask_type
            # )
            # self.storer.save_plot(
            #     eval_result["y_trues"], 
            #     eval_result["y_preds"], 
            #     "valid",
            #     self.args.multitask_type
            # )
        else:
            print(f"Val metric did not improve. Current best {self.best_val:.4f}")