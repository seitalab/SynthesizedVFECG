import os
import sys
import pickle
from typing import Tuple

sys.path.append("../utils")
from codes.eval_model import ModelEvaluator
from util_funcs import update_clf_mode

def run_eval(
    eval_target: str, 
    device: str, 
    dump_loc: str,
    eval_dataset: str="ptbxl",
    print_report: bool=True,
    dump_errors: bool=False,
    multiseed_run: bool=False
) -> Tuple[float, float]:
    """
    Args:
        eval_target (str): Path to eval target.
        device (str):
        eval_result_loc (str): 
    Returns:
        test_score (float): 
        test_loss (float): 
    """
    eval_info = eval_target
    if multiseed_run:
        dump_loc = os.path.join(dump_loc, "multirun", "eval")

    # Settings
    param_file = os.path.join(eval_target, "params.pkl")
    weightfile = os.path.join(eval_target, "net.pth")

    with open(param_file, "rb") as fp:
        params = pickle.load(fp)
    
    eval_info += "\n\nModel\n" + params.modelname
    eval_info += "\n\nTrained\n" + params.pos_dataset + "\n" + params.neg_dataset
    params = update_clf_mode(params, eval_dataset)
    params.data_lim = None
    eval_info += "\n\nTest\n" + params.pos_dataset + "\n" + params.neg_dataset
    eval_info += "\n\nParameters\n" + str(params)

    # Evaluator
    evaluator = ModelEvaluator(
        params, dump_loc, device)
    evaluator.set_model()
    evaluator.set_lossfunc()
    evaluator.set_weight(weightfile)

    valid_loader, test_loader = evaluator.prepare_dataloader()

    _, val_report = evaluator.run(valid_loader)
    test_result, test_report = evaluator.run(
        test_loader, dump_errors=dump_errors)
    if print_report:
        print(val_report)
        print(test_report)

        eval_info += "\n\nValidation set result\n\n" + val_report
        eval_info += "\n\nTest set result\n\n" + test_report

    evaluator.dump_target(eval_info)

    return test_result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eval', type=str, default="g12ec")
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    dump_loc = "../../evaled"

    assert args.target is not None
    run_eval(
        args.target, 
        args.device, 
        dump_loc, 
        eval_dataset=args.eval, 
        dump_errors=False
    )