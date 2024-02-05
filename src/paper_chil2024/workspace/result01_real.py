import os
from glob import glob

import yaml
import pandas as pd
from tqdm import tqdm

import utils

exp_dir01 = "v231222"
exp_dir02 = "v231212"
exp_dir03 = "v240110"
target_dx = "vf"
skip = [2030]

def make_row(exp_id, exp_dir):

    dirname = os.path.join(
        utils.root_dir, 
        exp_dir, 
        f"exp{exp_id//100*100:04d}",
        f"exp{exp_id:04d}"
    )
    result_path = sorted(glob(dirname + "/??????-??????"))[-1]
    dirname_str = "/".join(dirname.split("/")[8:10])
    target = glob(result_path + "/exp????.yaml")[-1]
    with open(target, "r") as f:
        cfg = yaml.safe_load(f)

    result = {}
    result["exp_id"] = exp_id
    result["dx"] = cfg["target_dx"]["param_val"]
    if result["dx"] != target_dx:
        raise
    result["dirname"] = dirname_str
    result["result_path"] = result_path
    result["model_legend"] =\
        utils.modelname_to_legend[cfg["modelname"]["param_val"]]

    result["pos_data"] = cfg["pos_dataset"]["param_val"]
    result["neg_data"] = cfg["neg_dataset"]["param_val"]

    result["n_pos_data"] = utils.n_data_dict[cfg["pos_dataset"]["param_val"]]
    result["n_neg_data"] = utils.n_data_dict[cfg["neg_dataset"]["param_val"]]

    return result

def main(exp_ids01, exp_ids02, exp_ids03):

    exp_ids01 = utils.split_exp_targets(exp_ids01)
    exp_ids02 = utils.split_exp_targets(exp_ids02)
    exp_ids03 = utils.split_exp_targets(exp_ids03)

    result = []
    for exp_id in tqdm(exp_ids01):
        if exp_id in skip:
            continue
        try:
            result.append(make_row(exp_id, exp_dir01))
        except:
            pass
    
    for exp_id in tqdm(exp_ids02):
        try:
            result.append(make_row(exp_id, exp_dir02))
        except:
            pass

    for exp_id in tqdm(exp_ids03):
        try:
            result.append(make_row(exp_id, exp_dir03))
        except:
            pass
    
    df = pd.DataFrame(result, columns=utils.cols)

    savename = utils.save_loc + "result01_real.csv"
    df.to_csv(savename, index=False)
    print(df)

if __name__ == "__main__":

    exp_ids01 = "2015-2212"
    exp_ids02 = "2001-2030"
    exp_ids03 = "2313"

    main(exp_ids01, exp_ids02, exp_ids03)

        

