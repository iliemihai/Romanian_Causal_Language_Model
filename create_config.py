import ast
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from transformers import (
    HfArgumentParser,
    AutoConfig
)

logger = logging.getLogger(__name__)

@dataclass
class ConfigArguments:

    output_dir: str = field(
        default = ".",
        metadata = {"help":"output dir where config is saved"}
    )
    name_or_path: Optional[str] = field(
        default = None,
        metadata = {"help":"model checkpoints for weight initialisation"}
    )
    params: Optional[str] = field (
        default = None,
        metadata = {"help":"custom config for specific name"}
    )

    def __post__init__(self):
        if self.params:
            try:
                self.params = ast.literal_eval(self.params)
            except Exception as e:
                print("Your custom parameters are not accepted due to  {e}")

def main():
    parser = HfArgumentParser([ConfigArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        config_args = parser.parse_args_into_dataclasses()[0]

        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"Setting up configuration {config_args.name_or_path} with extra params {config_args.params}")

    if config_args.params and isinstance(config_args.params, dict):
        config = AutoConfig.from_pretrained(config_args.name_or_path, **config_args.params)
    else:
        config = AutoConfig.from_pretrained(config_args.name_or_path)

    logger.info(f"Your configuration saved here {config_args.output_dir}/config.json")
    config.save_pretrained(config_args.output_dir)


if __name__ == "__main__":
    main()