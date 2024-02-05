import re
import pandas as pd

space = "  "
csvfile = "./resources/param_settings_exp04.csv"

def split_exp_targets(exp_targets):
    if exp_targets.isdigit():
        exp_ids = [int(exp_targets)]
    elif exp_targets.find(",") != -1:
        exp_ids = [int(v) for v in exp_targets.split(",")]
    elif exp_targets.find("-") != -1:
        s_e = exp_targets.split("-")
        s, e = int(s_e[0]), int(s_e[-1])
        exp_ids = [i for i in range(s, e+1)]
    else:
        raise
    return exp_ids

def load_base_yaml_as_text(base_exp_id):

    target_file = f"./resources/exp_yamls/exp{int(base_exp_id):04d}.yaml"
    base_yaml_text = open(target_file).read()
    return base_yaml_text

def update_cfg(cfg, param, param_val):

    src_pattern = (
        fr"{param}:\s*\n"
        fr"{space}param_type:\s*fixed\s*\n"
        fr"{space}param_val:\s.*?\n"
    )
    dst_pattern = (
        fr"{param}:\n"
        f"{space}param_type: fixed\n"
        f"{space}param_val: {param_val}\n"
    )
    param_text = re.sub(src_pattern, dst_pattern, cfg)

    return param_text

def insert_to_cfg(cfg, param, param_val):

    cfg = (
        "# Inserted\n"
        f"{param}:\n"
        f"{space}param_type: fixed\n"
        f"{space}param_val: {param_val}\n"
    ) + cfg

    return cfg

def main(exp_id):
    
    df = pd.read_csv(csvfile)
    row = df[df.exp_id == exp_id]
    assert row.shape[0] == 1

    base_cfg = load_base_yaml_as_text(row.base_yaml)

    # tmp: remove hps_result: null
    null_hps = "hps_result:\n  param_type: fixed\n  param_val: null"
    base_cfg = base_cfg.replace(null_hps, "")

    exp_cfg = update_cfg(
        base_cfg, "data_lim", row.data_lim.values[0])
    exp_cfg = insert_to_cfg(
        exp_cfg, "hps_result", row.hps_result.values[0])

    savename = f"./resources/exp_yamls/exp{exp_id:04d}.yaml"
    with open(savename, "w") as f:
        f.write(exp_cfg)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--exp', 
        default="9001,9002"
    )
    parser.add_argument(
        '--device', 
        default="cuda:0"
    )

    args = parser.parse_args()

    exp_ids = split_exp_targets(args.exp)

    for exp_id in exp_ids:
        main(exp_id)
