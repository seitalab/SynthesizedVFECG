import os

import yaml
import pandas as pd

resource_dir = "./resources"
param_order = [
    "### General", "modelname", "target_dx", "num_lead",
    "### Dataset", "pos_dataset", "neg_dataset", "data_lim", "val_lim", 
    "### Hyperparameter Search", "num_trials", "max_time", "hps_epochs",
    "### Training settings", "epochs", "eval_every", "patience", "batch_size", 
    "learning_rate", "weight_decay", "optimizer", "optimizer_patience", 
    "### Data preprocessing", "max_duration", "freq", "downsample", "mask_ratio", "max_shift_ratio",
    "### Model architecture", "backbone_out_dim"
]
skip_param = [
    "base"
]
space = "  "

def convert(param_dict):
    new_dict = {}
    for key, value in param_dict.items():
        new_dict[key] = {"param_type": "fixed", "param_val": value}
    return new_dict

def load_base_params(base_ver):
    """
    Args:
        None
    Returns:

    """
    base_yaml = os.path.join(
        resource_dir,
        f"base_params_{base_ver}.yaml"
    )
    with open(base_yaml, "r") as f:
        base_params = yaml.safe_load(f)
    return base_params

def load_arch_params(architecture):
    """
    Args:
        None
    Returns:

    """
    arch_yaml = os.path.join(
        resource_dir,
        f"arch_{architecture}.yaml"
    )
    with open(arch_yaml, "r") as f:
        arch_params = yaml.safe_load(f)
    
    if arch_params is None:
        arch_params = {}
    return arch_params

def load_setting_info(yaml_id):
    """
    Args:

    Returns:

    """
    csvfile = os.path.join(
        resource_dir, 
        "param_settings.csv"
    )
    df = pd.read_csv(csvfile, index_col=0)
    is_target_row = df.index == yaml_id
    settings = df.loc[is_target_row].iloc[0].to_dict()

    for key in settings:
        try:
            settings[key] = int(settings[key])
        except:
            pass
    return convert(settings)

def set_duration(target_dx):
    """
    Args:

    Returns:

    """
    if target_dx == "vf":
        duration = 9
    else:
        duration = 10
    return convert({"max_duration": duration})

def convert_to_text(param_name, param_info):
    """
    Args:

    Returns:

    """
    param_type = param_info["param_type"]
    if param_type == "fixed":
        param_detail = param_info["param_val"]
    elif param_type == "hps":
        param_detail = ""
        for p in param_info["param_val"]:
            if type(p) == float: # to avoid 1e-5
                param_detail += f"\n{space}{space}- {p:f}"
            else:
                param_detail += f"\n{space}{space}- {p}"
    else:
        raise NotImplementedError(f"param type {param_type} not implemented")

    block = f"{param_name}:\n"
    block += f"{space}param_type: {param_type}\n"
    block += f"{space}param_val: {param_detail}"
    return block

def dict_to_yamltxt(params):
    """
    Args:

    Returns:

    """
    param_txt = ""
    for param_name in param_order:
        if param_name.startswith("###"):
            param_txt += f"\n{param_name[2:]}\n"
        else:
            block = convert_to_text(
                param_name, params[param_name])
            param_txt += block + "\n"
            params.pop(param_name)

    for param_key, param_values in params.items():
        if param_key in skip_param:
            continue
        block = convert_to_text(param_key, param_values)
        param_txt += block + "\n"

    param_txt = param_txt.replace("param_val: nan", "param_val: null")
    param_txt = param_txt.replace("param_val: None", "param_val: null")
    return param_txt.strip()

def main(yaml_id):
    """
    Args:

    Returns:

    """
    setting = load_setting_info(yaml_id)
    base_params = load_base_params(setting["base"]["param_val"])
    arch_params = load_arch_params(setting["modelname"]["param_val"])
    duration_param = set_duration(setting["target_dx"]["param_val"])

    params = setting | base_params | arch_params | duration_param
    params["modelname"]["param_val"] = params["modelname"]["param_val"].split("_")[0]
    param_txt = dict_to_yamltxt(params)
    savename = os.path.join(
        resource_dir,
        "exp_yamls",
        f"exp{yaml_id:04d}.yaml"
    )
    with open(savename, "w") as f:
        f.write(param_txt)

if __name__ == "__main__":

    import sys

    try:
        exp_ids = sys.argv[1]
            
        if exp_ids.find(",") != -1:
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
        main(exp_id)
