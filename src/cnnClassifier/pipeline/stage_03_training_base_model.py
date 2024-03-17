from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.train_base_model import Training
from src.cnnClassifier import logger

STAGE_NAME = "Training Base Model Stage"

class BaseModelTrainingPipeline:
    def __int__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

