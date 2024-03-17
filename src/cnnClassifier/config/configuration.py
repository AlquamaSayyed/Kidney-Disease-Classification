from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_ymal, create_directories
from src.cnnClassifier.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_ymal(config_filepath)
        self.params = read_ymal(params_filepath)
        create_directories([self.config.artifact_root])

    def data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
