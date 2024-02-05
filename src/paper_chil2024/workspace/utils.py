
root_dir = "../../experiment"
save_loc = "./csvs/"
modelname_to_legend = {
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "transformer": "Transformer",
    "fedformer": "FEDformer",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "film": "FiLM",
    "luna": "LUNA",
    "nystrom": "Nystromformer",
    "lintrans": "Linear-Transformer",
    "performer": "Performer",
    "effnetb0": "EfficientNet-B0",
    "lstm": "LSTM",
    "gru": "GRU",
    "embgru": "Emb-GRU",
    "emblstm": "Emb-LSTM",
    "s4": "S4",
    "mega": "MEGA"
}

n_data_dict = {
    "cardially": 160,
    "PTBXL-NORM": 4598
}

cols = [
    "exp_id", 
    "model_legend", 
    "pos_data",
    "neg_data",
    "n_pos_data",
    "n_neg_data",
    "dx", 
    "dirname",
    "result_path"
]

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