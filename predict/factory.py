from config.settings import MODEL_NAME
from predict.adapters.chronos_adapter import ChronosAdapter
from predict.adapters.kronos_adapter import KronosAdapter

class PredictorFactory:
    _instances = {}

    @classmethod
    def get_adapter(cls):
        if MODEL_NAME not in cls._instances:
            if MODEL_NAME.startswith("kronos"):
                cls._instances[MODEL_NAME] = KronosAdapter(MODEL_NAME)
            else:
                cls._instances[MODEL_NAME] = ChronosAdapter(MODEL_NAME)
        return cls._instances[MODEL_NAME]