from datetime import datetime
from typing import List

import yaml
import numpy as np
import pandas as pd
from slack_sdk import WebClient

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
slack_config = config["slack"]

def get_timestamp() -> str:
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

def calc_class_weight(
    labels: List
) -> np.ndarray:
    """
    Calculate class weight for multilabel task.
    Args:
        labels (np.ndarray): Label data array of shape [num_sample, num_classes]
    Returns:
        class_weight (np.ndarray): Array of shape [num_classes].
    """
    labels = np.array(labels)[:, np.newaxis]
    num_samples = labels.shape[0]

    positive_per_class = labels.sum(axis=0)
    negative_per_class = num_samples - positive_per_class

    class_weight = negative_per_class / positive_per_class

    return class_weight

class SlackReporter:

    def __init__(self):
        """
        Args:
            None
        Returns:
            None
        """        
        self.client = WebClient(
            token=slack_config["token"]
        )
        self.channel_id = slack_config["channel_id"]

    def report(
        self, 
        message: str,
        parent_message: str, 
    ) -> None:
        """
        Args:
            message (str): 
            parent_message (str): 
        Returns:
            None
        """
        history = self.client.conversations_history(
            channel=self.channel_id
        )
        posts = history["messages"][:slack_config["max_past"]]

        for post in posts:
            if post["text"] != parent_message:
                continue
            self.client.chat_postMessage(
                channel=self.channel_id,
                thread_ts=post["ts"],
                text=message
            )
            break
    
    def post(self, message: str):
        """
        Args:
            message (str): 
        Returns:
            None
        """
        self.client.chat_postMessage(
            text=message, 
            channel=self.channel_id,
        )

class ResultManager:

    def __init__(self, savename: str, columns: List):
        self.savename = savename
        self.columns = columns
        self.results = []

    def add_result(self, row: List):
        """
        Add one row to results.
        
        Args:
            row (List): _description_
        Returns: 
            None
        """
        self.results.append(row)

    def get_result_df(self) -> pd.DataFrame:
        """
        Args:
            None
        Returns:
            df_result: 
        """
        df_result = pd.DataFrame(
            self.results, columns=self.columns)
        return df_result
    
    def save_result(self, is_temporal: bool=False):
        """

        Args:
            is_temporal (bool, optional): _description_. Defaults to False.
        """
        df_result = pd.DataFrame(
            self.results, columns=self.columns)
        
        savename = self.savename
        if is_temporal:
            savename = savename.replace(".csv", "_tmp.csv")
        df_result.to_csv(savename)
