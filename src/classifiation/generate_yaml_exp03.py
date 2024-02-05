import os
import re
from glob import glob

import pandas as pd

resource_dir = "./resources"
space = "  "

skip_keys = [
    "base_yaml",
    "notes"
]
csv_file = "param_setting_exp03.csv"

def convert(param_dict):
    new_dict = {}
    for key, value in param_dict.items():
        new_dict[key] = {"param_type": "fixed", "param_val": value}
    return new_dict

def load_base_params(hps_loc):
    """
    Args:
        None
    Returns:

    """
    base_yaml = glob(hps_loc + "/exp*.yaml")[-1]
    return open(base_yaml, "r").read()

def remove_hps(base_text: str):

    hps_pattern = (
        fr"{space}param_type:\s*hps\s*\n"
        fr"{space}param_val:\s*\n"
        fr"({space}{space}\-\s+.*?\n)+"
    )
    rep_pattern = (
        f"{space}param_type: hps\n"
        f"{space}param_val: []\n"
    )
    param_text = re.sub(hps_pattern, rep_pattern, base_text)

    return param_text

def load_setting_info(yaml_id):
    """
    Args:

    Returns:

    """
    csvfile = os.path.join(
        resource_dir, 
        csv_file
    )
    df = pd.read_csv(csvfile, index_col=0)
    is_target_row = df.index == yaml_id
    settings = df.loc[is_target_row].iloc[0].to_dict()
    return settings

def update_param_text(param_text, key, value):

    src_pattern = (
        fr"{key}:\s*\n"
        fr"{space}param_type:.*?\n"
        fr"{space}param_val:.*?(\n|$)"
    )

    dst_pattern = (
        fr"{key}: \n"
        fr"{space}param_type: fixed\n"
        fr"{space}param_val: {value}\n"
    )
    param_text = re.sub(src_pattern, dst_pattern, param_text)

    return param_text

def main(yaml_id):
    """
    Args:

    Returns:

    """
    setting = load_setting_info(yaml_id)
    param_text = load_base_params(setting["hps_result"])

    for key, value in setting.items():
        if key in skip_keys:
            continue
        # tmp
        if key == "hps_result":
            if not value.endswith("ResultTableHPS.csv"):
                value += "/ResultTableHPS.csv"
        param_text = update_param_text(param_text, key, value)
    param_text = update_param_text(param_text, "val_lim", 5000)

    savename = os.path.join(
        resource_dir,
        "exp_yamls",
        f"exp{yaml_id:04d}.yaml"
    )
    with open(savename, "w") as f:
        f.write(param_text)

if __name__ == "__main__":

    import sys

    try:
        exp_ids = sys.argv[1]
            
        if exp_ids.isdigit():
            exp_ids = [int(exp_ids)]
        elif exp_ids.find(",") != -1:
            exp_ids = [int(v) for v in exp_ids.split(",")]
        elif exp_ids.find("-") != -1:
            s_e = exp_ids.split("-")
            s, e = int(s_e[0]), int(s_e[-1])
            exp_ids = [i for i in range(s, e+1)]
        else:
            raise
    except:
        exp_ids = [9001]
    
    print(exp_ids)
    for exp_id in exp_ids:
        print(exp_id)
        main(exp_id)
