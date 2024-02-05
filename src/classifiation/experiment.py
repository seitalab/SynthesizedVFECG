import os
from typing import Optional
from argparse import Namespace

import yaml

from run_hps import run_hps
from run_eval import run_eval
from run_train import run_train
from codes.supports.utils import get_timestamp, ResultManager, SlackReporter
from codes.supports.param_utils import ParameterManager

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["experiment"]

class ExperimentManager:

    def __init__(
        self, 
        exp_id: int, 
        device: str,
        use_cpu: bool=False,
        debug: bool=False
    ):
        """

        Args:
            config_file (str): _description_
            device (str): cuda device or cpu to use.
            use_cpu (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
        """

        exp_config_file = os.path.join(
            config["path"]["yaml_loc"],
            f"exp{exp_id:04d}.yaml"
        )

        # Load parameters.
        fixed_params, self.hps_mode, search_params =\
            self._load_train_params(exp_config_file)
        self.fixed_params = Namespace(**fixed_params)
        self.search_params = Namespace(**search_params)

        self._prepare_save_loc(exp_id)
        self._save_config(exp_config_file)
        self._select_device(device, use_cpu)

        # For debugging.
        if debug:
            self.fixed_params.epochs = 2
            self.fixed_params.eval_every = 1
        else:
            self._prepare_reporter(exp_config_file)
        self.debug = debug

    def _select_device(self, device: str, use_cpu: bool):
        """
        Args:

        Returns:
            None
        """
        if use_cpu:
            device = "cpu"
        self.device = device

    def _prepare_save_loc(self, exp_id: int):
        """
        Args:

        Returns:
            None
        """
        self.save_loc = os.path.join(
            config["path"]["save_root"],
            f"exp{exp_id:04d}"[:-2]+"00",
            f"exp{exp_id:04d}",
            get_timestamp()
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def _prepare_reporter(self, config_file: str):
        """

        Args:
            config_file (str): _description_
            debug (bool): _description_
        """
        self.reporter = SlackReporter()
        parent_message = (
            f"Starting experiment for {self.fixed_params.target_dx} on {self.device}\n"
            f"Dataset: {self.fixed_params.pos_dataset} / {self.fixed_params.neg_dataset}\n"
            f"Experiment config file: `{config_file}`\n"
            f"Result save loc: `{self.save_loc}`"
        )
        self.reporter.post(parent_message)
        self.parent_message = parent_message

    def _save_config(self, config_file: str):
        """

        Args:
            config_file (str): _description_
        Returns:
            None
        """ 
        # Copy config file for current execution.
        config_basename = os.path.basename(config_file)
        stored_config_file = os.path.join(self.save_loc, config_basename)
        command = f"cp {config_file} {stored_config_file}"
        os.system(command)

    def _load_train_params(self, config_file: str):
        """

        Args:
            config_file (str): _description_
        Returns:
            fix_params (Dict): 
            hps_mode (bool): True if hps False if grid search.
            search_params (Dict): hps_params or gs_params.
        """ 
        with open(config_file) as f:
            params = yaml.safe_load(f)

        fixed_params, hps_params, gs_params = {}, {}, {}
        for key, value in params.items():
            if type(value) != dict:
                continue

            if value["param_type"] == "fixed":
                fixed_params[key] = value["param_val"]
            elif value["param_type"] == "grid":
                assert type(value["param_val"]) == list
                gs_params[key] = value["param_val"] # List stored
            elif value["param_type"] == "hps":
                assert type(value["param_val"]) == list
                hps_params[key] = value["param_val"]
            else:
                raise NotImplementedError
        # hps_params and gs_params must not have value at same time.
        assert not (bool(hps_params) and bool(gs_params))

        hps_mode = bool(hps_params)
        if hps_mode:
            return fixed_params, hps_mode, hps_params
        return fixed_params, hps_mode, gs_params

    def run_hps_experiment(self) -> str:
        """
        Args:
            None
        Returns:
            savename (str): _description_
        """
        # Prepare parameters.
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("seed", config["seed"]["hps"])
        param_manager.add_param("device", self.device)

        # Execute hyper parameter search.
        train_params = param_manager.get_parameter()
        csv_name = run_hps(
            train_params, self.save_loc, vars(self.search_params))

        # Copy hyperparameter result file.
        savename = os.path.join(
            self.save_loc, f"ResultTableHPS.csv")
        os.system(f"cp {csv_name} {savename}")

        # Report to slack.
        if not self.debug:
            self.reporter.report(
                f"Hyper parameter search done, result saved at {self.save_loc}", 
                self.parent_message
            )

        return savename

    def run_multiseed_experiment(self, score_sheet: Optional[str], single_run: bool):
        """_summary_

        Args:
            score_sheet (str): _description_
        """        
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("device", self.device)
        if score_sheet is not None:
            param_manager.update_by_search_result(
                score_sheet, vars(self.search_params), is_hps=True)

        # Prepare result storer.
        columns = ["seed", "dataset"] + config["result_cols"]        
        savename = os.path.join(self.save_loc, "ResultTableMultiSeed.csv")
        result_manager = ResultManager(savename=savename, columns=columns)

        for s_idx, seed in enumerate(config["seed"]["multirun"]):
            
            param_manager.add_param("seed", seed)
            save_loc_train = os.path.join(
                self.save_loc, "multirun", "train", f"seed{seed:04d}")
            os.makedirs(save_loc_train, exist_ok=True)
            train_params = param_manager.get_parameter()

            # Run training and store result.
            _, trained_model_loc = run_train(train_params, save_loc_train)

            # Eval with PTBXL.
            save_loc_eval = os.path.join(
                self.save_loc, "multirun", "eval", f"seed{seed:04d}")
            os.makedirs(save_loc_eval, exist_ok=True)

            if train_params.target_dx in config["dataset_dx"]["ptbxl"]:
                test_result_dict_ptbxl = run_eval(
                    eval_target=trained_model_loc, 
                    device=self.device,
                    dump_loc=save_loc_eval, 
                    eval_dataset="ptbxl",
                    multiseed_run=True
                )
                result_row = self._form_result_row(
                    seed, "ptbxl", columns, test_result_dict_ptbxl)
                result_manager.add_result(result_row)
            
            # Eval with G12EC.
            if train_params.target_dx in config["dataset_dx"]["g12ec"]:
                test_result_dict_g12ec = run_eval(
                    eval_target=trained_model_loc, 
                    device=self.device,
                    dump_loc=save_loc_eval, 
                    eval_dataset="g12ec",
                    multiseed_run=True
                )
                result_row = self._form_result_row(
                    seed, "g12ec", columns, test_result_dict_g12ec)
                result_manager.add_result(result_row)

            result_manager.save_result(is_temporal=True)
            if not self.debug:
                self.reporter.report(
                    f'Multi-seed evaluation {s_idx+1}/{len(config["seed"]["multirun"])} done', 
                    self.parent_message
                )
            if single_run:
                break

        result_manager.save_result()
        return result_manager.savename

    def _form_result_row(self, seed, dataset, columns, result_dict):
        """
        Args:

        Returns:

        """

        result_row = [seed, dataset]
        for key in columns[2:]:
            if key not in result_dict:
                result_row.append(None)
            else:
                result_row.append(result_dict[key])
        return result_row

    def main(self, single_run=False, hps_result=None):
        """
        Args:
            None
        Returns:
            None
        """
        # Overwrite.
        if self.fixed_params.hps_result is not None:
            assert hps_result is None
            hps_result = self.fixed_params.hps_result

        # Search.
        if hps_result is None:
            if self.hps_mode: 
                csv_path = self.run_hps_experiment()
            else:
                csv_path = None
        else:
            csv_path = hps_result

        # Report.
        if not self.debug:
            self.reporter.report(
                f"Search done, result saved at {csv_path}", 
                self.parent_message
            )
        
        # Multi seed eval.
        result_file = self.run_multiseed_experiment(csv_path, single_run)
        
        if not self.debug:
            self.reporter.report(
                f"Experiment done, result saved at {result_file}", 
                self.parent_message
            )
        # End

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--exp', 
        default=0
    )
    parser.add_argument(
        '--device', 
        default="cuda:0"
    )
    parser.add_argument(
        '--debug', 
        action="store_true"
    )
    parser.add_argument(
        '--multirun', 
        action="store_true"
    )    
    parser.add_argument(
        '--hps', 
        default=None
    )    
    args = parser.parse_args()

    print(args)

    executer = ExperimentManager(
        int(args.exp), 
        args.device,
        debug=args.debug
    )
    executer.main(
        single_run=not args.multirun,
        hps_result=args.hps
    )
