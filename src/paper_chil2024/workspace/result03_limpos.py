import os
from glob import glob

import yaml
import pandas as pd
from tqdm import tqdm

import utils

exp_dir = "v240124_kake"

def make_row(exp_id):

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
    result["dirname"] = dirname_str
    result["result_path"] = result_path
    result["model_legend"] =\
        utils.modelname_to_legend[cfg["modelname"]["param_val"]]

    result["pos_data"] = cfg["pos_dataset"]["param_val"]
    result["neg_data"] = cfg["neg_dataset"]["param_val"]

    result["n_pos_data"] = min(
        int(cfg["data_lim"]["param_val"][:-1]), 
        utils.n_data_dict[cfg["pos_dataset"]["param_val"]]
    )
    result["n_neg_data"] =\
        utils.n_data_dict[cfg["neg_dataset"]["param_val"]]

    return result

def main(exp_ids):

    exp_ids = utils.split_exp_targets(exp_ids)

    result = []
    for exp_id in tqdm(exp_ids):
        try:
            result.append(make_row(exp_id))
        except:
            pass
    df = pd.DataFrame(result, columns=utils.cols)

    savename = utils.save_loc + "result03_limpos.csv"
    df.to_csv(savename, index=False)
    print(df)

if __name__ == "__main__":

    exp_ids = "2800-2859"
    main(exp_ids)

        

