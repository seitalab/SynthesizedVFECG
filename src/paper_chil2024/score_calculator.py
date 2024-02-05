import os
import sys
import pickle
from glob import glob

import numpy as np
from sklearn.metrics import (
    f1_score, 
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

sys.path.append("..")
sys.path.append("../utils")
from util_funcs import update_clf_mode

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return specificity

def score_calculator(y_trues, y_preds):
    y_preds = sigmoid(y_preds)

    f1 = f1_score(y_trues, y_preds > 0.5)
    acc = accuracy_score(y_trues, y_preds > 0.5)
    prec = precision_score(y_trues, y_preds > 0.5)
    recall = recall_score(y_trues, y_preds > 0.5)
    spec = specificity_score(y_trues, y_preds > 0.5)
    auroc = roc_auc_score(y_trues, y_preds)
    auprc = average_precision_score(y_trues, y_preds)
    confmat = confusion_matrix(y_trues, y_preds > 0.5).T

    result = {
        # "ConfMatrix": confmat,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(recall, 4),
        "Specificity": round(spec, 4),
        "F1": round(f1, 4),
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
    }

    return result

def calculate_score(
    eval_target: str,
    eval_dataset: str,
    seed: int, 
    is_mae: bool, 
    device: str
):
    if is_mae:
        tmp = list(set(sys.path))
        if "../clf01" in tmp:
            tmp.remove("../clf01")
        sys.path = tmp
        sys.path.append("../gen02")
        ng_keys = list(
            filter(lambda ele: ele.startswith("codes."), sys.modules.keys()))
        for ng_key in ng_keys:
            del sys.modules[ng_key]
        if "clf01.codes.eval_model" in sys.modules.keys():
            del sys.modules["clf01.codes.eval_model"]
        from gen02.codes.eval_model import ModelEvaluator
    else:
        tmp = list(set(sys.path))
        if "../gen02" in tmp:
            tmp.remove("../gen02")
        sys.path = tmp
        sys.path.append("../clf01")
        ng_keys = list(
            filter(lambda ele: ele.startswith("codes."), sys.modules.keys()))
        for ng_key in ng_keys:
            del sys.modules[ng_key]
        if "gen02.codes.eval_model" in sys.modules.keys():
            del sys.modules["gen02.codes.eval_model"]        
        from clf01.codes.eval_model import ModelEvaluator
    
    # Setting files.
    target_dir = glob(
        eval_target + f"/multirun/train/seed{seed:04d}/??????-??????-*")
    if len(target_dir) == 0:
        return None
    target_dir = target_dir[-1]
    param_file = os.path.join(target_dir, "params.pkl")
    weightfile = os.path.join(target_dir, "net.pth")
    if not os.path.exists(weightfile):
        return None

    # Update params.
    with open(param_file, "rb") as fp:
        params = pickle.load(fp)
    params = update_clf_mode(params, eval_dataset)
    params.data_lim = None

    # Set evaluator
    dump_loc = "./tmp"
    # if is_mae:
    #     evaluator = ME1(params, dump_loc, device)
    # else:
    #     evaluator = ME2(params, dump_loc, device)
    evaluator = ModelEvaluator(params, dump_loc, device)
    evaluator.set_model()
    evaluator.set_lossfunc()
    evaluator.set_weight(weightfile)

    _, test_loader = evaluator.prepare_dataloader()
    test_result, _ = evaluator.run(test_loader, dump_errors=False)
    score_dict = score_calculator(
        test_result["y_trues"], test_result["y_preds"])
    
    os.system(f"rm -r {dump_loc}")
    return score_dict

if __name__ == "__main__":

    pass