from hydra.core.config_store import ConfigStore
import hydra
from omegaconf import DictConfig
from schemas import (
    PipelineConfig, 
    CatBoostConfig, 
    GradientBoostingConfig, 
    LightGBMConfig, 
    XGBoostConfig, 
    ResNet18Config, 
    ResNet34Config, 
    ResNet50Config
)

cs = ConfigStore.instance()
cs.store(name="train_config", node=PipelineConfig)
cs.store(group="model", name="resnet18", node=ResNet18Config)
cs.store(group="model", name="resnet34", node=ResNet34Config)
cs.store(group="model", name="resnet50", node=ResNet50Config)
cs.store(group="model", name="gradient_boosting", node=GradientBoostingConfig)
cs.store(group="model", name="xgboost", node=XGBoostConfig)
cs.store(group="model", name="lightgbm", node=LightGBMConfig)
cs.store(group="model", name="catboost", node=CatBoostConfig)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: PipelineConfig):
    print(config)


if __name__ == "__main__":
    main()