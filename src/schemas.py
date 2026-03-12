from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from omegaconf import MISSING



# Contrato dos dados
@dataclass
class DataConfig:
    data_dir: str = MISSING
    target_col: str = MISSING
    download: Optional[bool] = None
    categorical_cols: List[str] = field(default_factory=list)
    numerical_cols: List[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = MISSING
    num_workers: int = 4
    pin_memory: bool = True



# Contrato do Treinamento
@dataclass
class TrainConfig:
    learning_rate: float = MISSING
    num_epochs: int = MISSING
    weight_decay: float = 0.0
    optimizer: str = MISSING
    accelerator: str = "auto"
    devices: Optional[int] = None
    log_every_n_steps: int = 10
    early_stopping_patience: int = 5
    gradient_clip_val: float = 1.0



# Contrato dos Modelos
@dataclass
class BaseModelConfig:
    num_classes: int = 10



@dataclass
class NeuralNetConfig(BaseModelConfig):
    in_channels: int = 1
    pretrained: bool = False
    freeze_backbone: bool = False
    dropout: float = 0.5



@dataclass
class ResNet18Config(NeuralNetConfig):
    _target_: str = "torchvision.models.resnet18"



@dataclass
class ResNet34Config(NeuralNetConfig):
    _target_: str = "torchvision.models.resnet34"



@dataclass
class ResNet50Config(NeuralNetConfig):
    _target_: str = "torchvision.models.resnet50"



@dataclass
class GradientBoostingConfig(BaseModelConfig):
    n_estimators: int = MISSING
    max_depth: int = MISSING
    learning_rate: float = MISSING
    subsample: float = 1.0
    max_features: Optional[float] = None



@dataclass
class XGBoostConfig(BaseModelConfig):
    n_estimators: int = MISSING
    max_depth: int = MISSING
    learning_rate: float = MISSING
    subsample: float = 1.0
    colsample_bytree: float = 1.0



@dataclass
class LightGBMConfig(BaseModelConfig):
    n_estimators: int = MISSING
    max_depth: int = MISSING
    learning_rate: float = MISSING
    num_leaves: int = 31
    subsample: float = 1.0
    feature_fraction: float = 1.0



@dataclass
class CatBoostConfig(BaseModelConfig):
    iterations: int = MISSING
    depth: int = MISSING
    learning_rate: float = MISSING
    subsample: float = 1.0
    rsm: float = 1.0



# Contrato Mestre
@dataclass
class PipelineConfig:
    data: DataConfig = MISSING
    train: TrainConfig = MISSING
    model: BaseModelConfig = MISSING