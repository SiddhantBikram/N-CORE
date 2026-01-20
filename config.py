import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Model configurations
    model_name: str = "facebook/hubert-base-ls960"
    pretrained_path: Optional[str] = None
    
    # Data configurations
    audio_base_dir: str = ""
    csv_path: str = ""
    dataset_name: str = "dataset"
    
    # Training configurations
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 100
    batch_size: int = 4  # Physical batch size
    accumulation_steps: int = 4  # Effective batch size = batch_size * accumulation_steps
    lr: float = 1e-5
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    test_size: float = 0.2
    
    # Random seed
    seed: int = 42
    
    # Number of perturbed samples
    num_perturbations: int = 1
    
    # Loss weights
    lambda_grl: float = 1/10
    lambda_reg: float = 1/200
    grl_alpha: float = 1.0
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps


@dataclass
class EmotionClassificationConfig(BaseConfig):
    """Configuration for emotion classification with speaker perturbation."""
    
    model_save_path: str = "models/hubert_grl_emotion_model.pt"
    perturbed_audio_dir: str = "perturbed_audio/speaker_perturbed"
    
    # NANSY-style augmentation parameters for speaker perturbation
    use_speaker_augmentation: bool = True
    formant_shift_range: Tuple[float, float] = (0.7, 1.4)
    pitch_shift_range: Tuple[float, float] = (0.5, 2.0)
    pitch_range_ratio_range: Tuple[float, float] = (0.7, 1.5)
    peq_gain_range: Tuple[float, float] = (-12, 12)
    
    # Task description
    primary_task: str = "emotion"
    adversarial_task: str = "speaker"


@dataclass
class SpeakerClassificationConfig(BaseConfig):
    """Configuration for speaker classification with affective perturbation."""
    
    model_save_path: str = "models/hubert_grl_speaker_model.pt"
    perturbed_audio_dir: str = "perturbed_audio/emotion_perturbed"
    
    # Perturbation type
    perturbation_type: str = "affective" 
    
    # Affective feature perturbation parameters
    target_rms: float = 0.05
    
    # Content perturbation parameters
    n_spectral_bands: int = 20
    
    # Task description
    primary_task: str = "speaker"
    adversarial_task: str = "emotion"


# Dataset configurations for different datasets
DATASET_CONFIGS: List[Dict[str, str]] = [
    {
        "name": "recanvo",
        "audio_base_dir": "ReCANVo",
        "csv_path": "recanvo.csv"
    },
    {
        "name": "recanvo_balanced",
        "audio_base_dir": "ReCANVo",
        "csv_path": "recanvo_balanced.csv"
    },
    {
        "name": "vivae",
        "audio_base_dir": "VIVAE/full_set",
        "csv_path": "vivae.csv"
    }
]


def get_dataset_config(dataset_name: str) -> Dict[str, str]:
    """Get dataset configuration by name."""
    for config in DATASET_CONFIGS:
        if config["name"] == dataset_name:
            return config
    raise ValueError(f"Dataset '{dataset_name}' not found. Available: {[c['name'] for c in DATASET_CONFIGS]}")
