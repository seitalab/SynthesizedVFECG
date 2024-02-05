
def update_clf_mode(params, realdataset="g12ec"):

    # Select positive dataset for evaluation.
    if params.target_dx == "af":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-AFIB"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-Afib"
    elif params.target_dx == "pvc":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-PVC"
        elif realdataset == "g12ec":
            params.pos_dataset = "G12EC-VPB"
    elif params.target_dx == "vf":
        params.pos_dataset = "cardially"

    elif params.target_dx == "aflt":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-AFLT"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "pac":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-PAC"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "irbbb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-IRBBB"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "crbbb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-CRBBB"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "std":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-STD_"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "wpw":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-WPW"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "3avb":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-3AVB"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "asmi":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-ASMI"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "imi":
        if realdataset == "ptbxl":
            params.pos_dataset = "PTBXL-IMI"
        elif realdataset == "g12ec":
            raise

    elif params.target_dx == "irbbb-crbbb":
        params.pos_dataset = "PTBXL-CRBBB"
    else:
        raise NotImplementedError(f"{params.target_dx} is not implemented")
    
    # Negative dataset.
    if params.target_dx == "irbbb-crbbb":
        params.neg_dataset = "PTBXL-IRBBB"
    else:
        if realdataset == "ptbxl":
            params.neg_dataset = "PTBXL-NORM"
        elif realdataset == "g12ec":
            params.neg_dataset = "G12EC-NormalSinus"
        else:
            raise NotImplementedError(f"{realdataset} is invalid.")
    
    return params