import logging, logging.config
from lib.utils.log import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

from lib.utils.cli import CustomCli
import sys

sys.path.append("..")

from lib.models import *
from lib.datasets import *


def main():
    logger = logging.getLogger()
    cli = CustomCli(
        MaskModel,
        MaskModule,
        run=False,
        parser_kwargs={"default_config_files": ["configs/defaults.yml"],},
    )
    cli.setup_callbacks()
    cli.log_info()

    logger.info(f"{'=' * 10} start training {'=' * 10}")
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
