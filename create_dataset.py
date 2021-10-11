import ast
import logging
import os
import sys
from dataclasses import dataclass, field
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset

from transformers import (
    HfArgumentParser,
)

from data_utils import (
    normalizer
)



logger = logging.getLogger(__name__)

@dataclass
class DataArguments:

    output_dir: str = field(
        default = ".",
        metadata = {"help":"output dir where config is saved"}
    )
    dataset_name: str = field(
        default = None,
        metadata = {"help":"name of the dataset to use"}
    )
    dataset_config_name: Optional[str] = field(
        default = None,
        metadata = {"help":"configuration of the dataset to use"}
    )
    train_file: Optional[str] = field (
        default = None,
        metadata = {"help":"txt training file"}
    )
    cache_dir: Optional[str] = field (
        default = None,
        metadata = {"help":"Where to store the pretrained models"}
    )
    train_test_splir_ratio: Optional[float] = field (
        default = 0.1,
        metadata = {"help":"Where to store the pretrained models"}
    )

def main():
    parser = HfArgumentParser([DataArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Preparing the dataset")
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.cache_dir,
            split="train"
    )
    else:
        data_files = {"train": data_args.train_file}
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"

        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=data_args.cache_dir,
        )

    logger.info(f"dataset: {dataset}")

    def data_preparation(item_dict):
        if "text" not in item_dict:
            return None

        text = item_dict["text"]
        text = normalizer(text)

        return {"text": text}

    data_dict = []
    for item in tqdm(dataset["train"], position=0, total=len(dataset)):
        item = data_preparation(item)
        if item:
            data_dict.append(item)

    data_df = pd.DataFrame(data_dict)

    logger.info(f"Preparation - [before] consists of {len(dataset)} records!")
    logger.info(f"Preparation - [after]  consists of {len(data_df)} records!")

    train, test = train_test_split(data_df, test_size=data_args.train_test_splir_ratio, random_state=101)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    logger.info(f"Preparation of [train] set consists of {len(train)} records!")
    logger.info(f"Preparation of [test] set consists of {len(test)} records!")

    os.makedirs(data_args.output_dir, exist_ok=True)
    train.to_csv(os.path.join(data_args.output_dir, "train.csv"), sep="\t", encoding="utf-8", index=False)
    test.to_csv(os.path.join(data_args.output_dir, "test.csv"), sep="\t", encoding="utf-8", index=False)
    logger.info(f"Data saved here {data_args.output_dir}")

if __name__ == "__main__":
    main()