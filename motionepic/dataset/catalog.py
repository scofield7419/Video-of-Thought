import os


class DatasetCatalog:
    def __init__(self):

        self.webvid_enc = {
            "target": "motionepic.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/webvid/webvid.json",
                video_folder="",
            ),
        }