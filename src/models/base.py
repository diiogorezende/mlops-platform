from schemas import (
    BaseModelConfig,
    XGBoostConfig,
    LightGBMConfig,
    CatBoostConfig,
    GradientBoostingConfig,
    NeuralNetConfig
)
from models.tree_model import build_tree_model
from models.neural_net import build_neural_net

class ModelFactory:
    """
    Factory class that receives the Hydra config and returns the model for training.
    """
    @staticmethod
    def create(cfg: BaseModelConfig):
        if isinstance(cfg, (XGBoostConfig, LightGBMConfig, CatBoostConfig, GradientBoostingConfig)):
            return build_tree_model(cfg)
        elif isinstance(cfg, NeuralNetConfig):
            return build_neural_net(cfg)
        else:
            raise ValueError(f"Unsupported model config type: {type(cfg)}")