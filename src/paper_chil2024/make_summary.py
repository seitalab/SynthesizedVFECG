import os
import yaml
import pandas as pd

from score_calculator import calculate_score

cfg_file = "../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

table_loc = "./workspace/csvs"
result_loc = "./result_csvs"

SKIP = ["FiLM"]
# SKIP = []

class SummaryMaker:

    def __init__(self, device: str):

        self.device = device

        self.save_loc = result_loc

    def _calc_multirun_result(
        self, 
        target_loc, 
        dataset,
        is_mae
    ):
        """
        """
        results = []
        for seed in cfg["experiment"]["seed"]["multirun"]:
            result = calculate_score(
                target_loc,
                dataset,
                seed, 
                is_mae,
                self.device
            )
            # Quit evaluation if we don't have all 5 run results.
            if result is None:
                return None
            results.append(
                pd.DataFrame([result], index=[seed]))
            
        df_result = pd.concat(results)
        return df_result
    
    def trial(self, target, is_mae):

        df = self._calc_multirun_result(target, "ptbxl", is_mae)
        print(df.mean(axis=0))
        print(df.std(axis=0))

    def make_result_table(self, tablename: str, dataset: str):
        """
        """
        df_targets = pd.read_csv(os.path.join(table_loc, tablename))

        results = []
        for i, row in df_targets.iterrows():

            print("*"*80)
            print(f"{i+1} / {len(df_targets)} | {row.model_legend}")
            if row.model_legend in SKIP:
                continue

            result = row.to_dict()

            if not pd.isna(row.result_path):
                df = self._calc_multirun_result(
                    row.result_path, dataset, False)
            else:
                df = None

            if df is not None:
                for col in df.columns:
                    vals = df.loc[:, col].values
                    val_txt = f"{vals.mean():.04f} ± {vals.std():.04f}"
                    result[col] = val_txt
            results.append(result)
        df_results = pd.DataFrame(results)

        # Save result.
        savename = os.path.join(
            self.save_loc,
            f"{tablename[:-4]}_{dataset}.csv"
        )
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        df_results.to_csv(savename)

if __name__ == "__main__":

    device = "cuda:1"
    summarizer = SummaryMaker(device)

    tbs = [
        # "result01_real.csv",
        # "result02_syn.csv",
        "result03_limpos.csv",
        # "result04_app_limdata.csv",
        "result05_syn_vardata.csv"
    ]

    for dataset in ["ptbxl", "g12ec"][:1]:
        for tb in tbs:
            summarizer.make_result_table(tb, dataset)
    print("done")