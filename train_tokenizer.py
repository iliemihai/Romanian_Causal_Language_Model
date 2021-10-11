import ast
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    HfArgumentParser,
)

from data_utils import (
    normalizer
)

logger = logging.getLogger(__name__)

@dataclass
class TokenizerArguments:
    """
       arguments to wich tokenizer we are going to set up
    """
    output_dir: str = field(
        default = ".",
        metadata = {"help":"output dir where config is saved"}
    )
    dataset_name: Optional[str] = field(
        default = None,
        metadata = {"help":"name of dataset to use"}
    )
    dataset_config_name: Optional[str] = field (
        default = None,
        metadata = {"help":"custom config for specific dataset name"}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded"},
    )
    special_tokens: Optional[str] = field(
        default = None,
        metadata = {"help":"special tokens"}
    )
    vocab_size: Optional[int] = field(
        default = 50257,
        metadata = {"help":"size of final vocabulary"}
    )
    min_frequency: Optional[int] = field(
        default = 2,
        metadata = {"help":"min frequency of pair in text"}
    )
    show_progress: Optional[bool] = field(
        default = True,
        metadata = {"help":"size of final vocabulary"}
    )

    def __post_init__(self):
        if self.special_tokens is None:
            special_tokens = [
                "<s>", "<pad>", "</s>", "<mask>", "<unk>"
            ]
            special_tokens += [f"[U{i}]" for i in range(1, 21)]
        else:
            special_tokens = list(self.special_tokens.split(","))
        self.special_tokens = special_tokens
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


def main():

    parser = HfArgumentParser([TokenizerArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        tokenizer_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        tokenizer_args = parser.parse_args_into_dataclasses()[0]

        # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Training tokenizer")

    if tokenizer_args.dataset_name is not None:
        raw_dataset = load_dataset(
            tokenizer_args.dataset_name,
            tokenizer_args.dataset_config_name,
            cache_dir=tokenizer_args.cache_dir,
            split="train"
        )
    else:
        data_files = {"train": tokenizer_args.train_file}
        extension = tokenizer_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=tokenizer_args.cache_dir,
        )

    logger.info("Preprocessing the dataset")
    dataset = raw_dataset["train"]#.map(normalizer)
    logger.info(f"Preprocessed dataset kept {len(dataset)} out of {len(raw_dataset)}")

    tokenizer = ByteLevelBPETokenizer()

    def batch_iterative(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]

    tokenizer.train_from_iterator(
        batch_iterative(),
        vocab_size=tokenizer_args.vocab_size,
        special_tokens=tokenizer_args.special_tokens,
        min_frequency=tokenizer_args.min_frequency,
        show_progress=tokenizer_args.show_progress,
    )

    logger.info(f"Your tokenizer saved here {tokenizer_args.output_dir}")
    os.makedirs(tokenizer_args.output_dir, exist_ok=True)
    tokenizer.save_model(tokenizer_args.output_dir)
    tokenizer.save(f"{tokenizer_args.output_dir}/tokenizer.json", pretty=True)

if __name__ == "__main__":
    main()