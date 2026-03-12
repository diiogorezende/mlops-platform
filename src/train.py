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
cs.store(name="resnet18_schema", node=ResNet18Config)
cs.store(name="resnet34_schema", node=ResNet34Config)
cs.store(name="resnet50_schema", node=ResNet50Config)
cs.store(name="gradient_boosting_schema", node=GradientBoostingConfig)
cs.store(name="xgboost_schema", node=XGBoostConfig)
cs.store(name="lightgbm_schema", node=LightGBMConfig)
cs.store(name="catboost_schema", node=CatBoostConfig)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: PipelineConfig):
    # ConfigDict
    OmegaConfDict = DictConfig(config)
    print(OmegaConfDict)


if __name__ == "__main__":
    main()