import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
from tqdm import tqdm

import librosa
import soundfile as sf

from datasets import Dataset, Audio, ClassLabel, Features, Value
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor

from perturbation import apply_perturbation


def generate_perturbed_samples(
    dataset_df: pd.DataFrame,
    audio_base_dir: str,
    perturbed_dir: str,
    num_perturbations: int,
    perturbation_type: str = "nansy",
    **perturbation_kwargs
) -> pd.DataFrame:

    print(f"Generating/Verifying {num_perturbations} perturbed sample(s) using {perturbation_type} perturbation...")
    os.makedirs(perturbed_dir, exist_ok=True)
    
    # Initialize column if not present
    if 'Perturbed_Filenames' not in dataset_df.columns:
        dataset_df['Perturbed_Filenames'] = pd.Series([[] for _ in range(len(dataset_df))], index=dataset_df.index)
    else:
        dataset_df['Perturbed_Filenames'] = dataset_df['Perturbed_Filenames'].apply(
            lambda x: x if isinstance(x, list) else []
        )
    
    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Processing audio"):
        orig_path = os.path.join(audio_base_dir, row['Filename'])
        file_name = os.path.basename(orig_path)
        
        # Check if all required perturbed files exist
        required_paths = [
            os.path.join(perturbed_dir, f"perturbed_{i}_{file_name}")
            for i in range(num_perturbations)
        ]
        all_exist = all(os.path.exists(p) for p in required_paths)
        
        if all_exist:
            dataset_df.at[idx, 'Perturbed_Filenames'] = required_paths
            continue
        
        # Generate perturbed samples
        generated_paths = []
        if not os.path.exists(orig_path):
            print(f"Warning: Original audio file not found {orig_path}. Skipping.")
            dataset_df.at[idx, 'Perturbed_Filenames'] = []
            continue
        
        try:
            audio_np, sr = librosa.load(orig_path, sr=16000)
            
            for i in range(num_perturbations):
                perturbed_path = os.path.join(perturbed_dir, f"perturbed_{i}_{file_name}")
                
                # Add some randomization for multiple perturbations
                kwargs = perturbation_kwargs.copy()
                if perturbation_type == "affective":
                    kwargs["target_rms"] = kwargs.get("target_rms", 0.05) * (1 + random.uniform(-0.03, 0.03))
                elif perturbation_type == "content":
                    kwargs["n_bands"] = max(2, kwargs.get("n_bands", 10) + random.randint(-2, 2))
                
                perturbed_audio = apply_perturbation(
                    audio_np.copy(),
                    sr=sr,
                    perturbation_type=perturbation_type,
                    **kwargs
                )
                sf.write(perturbed_path, perturbed_audio, sr)
                generated_paths.append(perturbed_path)
            
            dataset_df.at[idx, 'Perturbed_Filenames'] = generated_paths
            
        except Exception as e:
            print(f"Error processing {orig_path}: {str(e)}")
            dataset_df.at[idx, 'Perturbed_Filenames'] = []
    
    return dataset_df


def prepare_hf_dataset(
    df: pd.DataFrame,
    audio_base_dir: str,
    num_perturbations: int
) -> Tuple[Dataset, Dict[int, str], Dict[int, str], int, int]:

    data_dict = {
        "audio_path": [],
        "perturbed_paths": [],
        "participant": [],
        "label": [],
        "emotion_id": [],
        "participant_id": []
    }
    
    for _, row in df.iterrows():
        orig_path = os.path.join(audio_base_dir, row["Filename"])
        perturbed_paths = row.get("Perturbed_Filenames", [])
        
        # Verify all files exist
        all_files_present = os.path.exists(orig_path)
        if num_perturbations > 0:
            if not (isinstance(perturbed_paths, list) and
                    len(perturbed_paths) == num_perturbations and
                    all(isinstance(p, str) and os.path.exists(p) for p in perturbed_paths)):
                all_files_present = False
        
        if all_files_present:
            data_dict["audio_path"].append(orig_path)
            data_dict["perturbed_paths"].append(perturbed_paths)
            data_dict["participant"].append(row["Participant"])
            data_dict["label"].append(row["Label"])
            data_dict["emotion_id"].append(row["Emotion_ID"])
            data_dict["participant_id"].append(row["Participant_ID"])
    
    if not data_dict["audio_path"]:
        raise ValueError("No valid samples found after processing DataFrame.")
    
    # Create mappings
    emotion_labels = sorted(list(set(data_dict["label"])))
    emotion_ids = sorted(list(set(data_dict["emotion_id"])))
    speaker_ids = sorted(list(set(data_dict["participant_id"])))
    
    id_to_emotion = {eid: lbl for eid, lbl in zip(data_dict["emotion_id"], data_dict["label"])}
    id_to_emotion = {k: id_to_emotion[k] for k in emotion_ids if k in id_to_emotion}
    
    id_to_speaker = {pid: name for pid, name in zip(data_dict["participant_id"], data_dict["participant"])}
    id_to_speaker = {k: id_to_speaker[k] for k in speaker_ids if k in id_to_speaker}
    
    # Create HF Dataset
    features = Features({
        "audio": Audio(sampling_rate=16000),
        "perturbed_paths": [Value("string")],
        "participant": Value("string"),
        "label": ClassLabel(names=emotion_labels),
        "emotion_id": Value("int32"),
        "participant_id": Value("int32")
    })
    
    dataset_input = {
        "audio": data_dict["audio_path"],
        "perturbed_paths": data_dict["perturbed_paths"],
        "participant": data_dict["participant"],
        "label": data_dict["label"],
        "emotion_id": data_dict["emotion_id"],
        "participant_id": data_dict["participant_id"]
    }
    
    hf_dataset = Dataset.from_dict(dataset_input, features=features)
    
    return (
        hf_dataset,
        id_to_emotion,
        id_to_speaker,
        len(emotion_ids),
        len(speaker_ids)
    )


@dataclass
class DataCollatorWithPairedAudio:
    processor: Any
    num_perturbations: int
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Process original audio
        orig_audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rate = features[0]["audio"]["sampling_rate"] if features else 16000
        
        batch = self.processor(
            orig_audio_arrays,
            sampling_rate=sampling_rate,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        # Process perturbed audio
        list_perturbed_inputs = []
        list_perturbed_masks = []
        
        if self.num_perturbations > 0:
            for i in range(self.num_perturbations):
                perturbed_arrays = []
                for f in features:
                    paths = f.get("perturbed_paths", [])
                    if i < len(paths) and isinstance(paths[i], str) and os.path.exists(paths[i]):
                        audio_array, _ = librosa.load(paths[i], sr=sampling_rate)
                        perturbed_arrays.append(audio_array)
                    else:
                        # Fallback to original
                        perturbed_arrays.append(f["audio"]["array"])
                
                perturbed_batch = self.processor(
                    perturbed_arrays,
                    sampling_rate=sampling_rate,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt"
                )
                list_perturbed_inputs.append(perturbed_batch["input_values"])
                list_perturbed_masks.append(perturbed_batch["attention_mask"])
        
        batch["list_of_perturbed_input_values"] = list_perturbed_inputs
        batch["list_of_perturbed_attention_masks"] = list_perturbed_masks
        
        # Add labels
        if features and "emotion_id" in features[0]:
            batch["emotion_labels"] = torch.tensor([f["emotion_id"] for f in features], dtype=torch.long)
        if features and "participant_id" in features[0]:
            batch["speaker_labels"] = torch.tensor([f["participant_id"] for f in features], dtype=torch.long)
        
        return batch


def seed_worker(worker_id):
    """Seed worker for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataset_with_perturbations(
    csv_path: str,
    audio_base_dir: str,
    perturbed_dir: str,
    num_perturbations: int,
    perturbation_type: str,
    processor: AutoProcessor,
    test_size: float = 0.2,
    batch_size: int = 4,
    seed: int = 42,
    stratify_by: str = "emotion",
    **perturbation_kwargs
) -> Tuple[DataLoader, DataLoader, Dict, Dict, int, int, pd.DataFrame, pd.DataFrame]:

    # Load and process metadata
    df = pd.read_csv(csv_path)
    df = generate_perturbed_samples(
        df, audio_base_dir, perturbed_dir,
        num_perturbations, perturbation_type,
        **perturbation_kwargs
    )
    
    # Create HF dataset
    hf_dataset, id_to_emotion, id_to_speaker, num_emotions, num_speakers = prepare_hf_dataset(
        df, audio_base_dir, num_perturbations
    )
    
    # Split dataset
    stratify_col = "emotion_id" if stratify_by == "emotion" else "participant_id"
    train_indices, test_indices = train_test_split(
        np.arange(len(hf_dataset)),
        test_size=test_size,
        random_state=seed,
        stratify=hf_dataset[stratify_col]
    )
    
    train_dataset = hf_dataset.select(train_indices)
    test_dataset = hf_dataset.select(test_indices)
    
    # Create data loaders
    collator = DataCollatorWithPairedAudio(
        processor=processor,
        num_perturbations=num_perturbations
    )
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=generator,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=generator,
        num_workers=0
    )
    
    # Create info DataFrames
    def create_info_df(dataset, id_to_emotion_map):
        if len(dataset) == 0:
            return pd.DataFrame()
        return pd.DataFrame({
            "AudioPath": [dataset[i]["audio"]["path"] for i in range(len(dataset))],
            "PerturbedPaths": [dataset[i]["perturbed_paths"] for i in range(len(dataset))],
            "ParticipantName": dataset["participant"],
            "EmotionID": dataset["emotion_id"],
            "EmotionLabel": [id_to_emotion_map.get(eid, f"Unknown:{eid}") for eid in dataset["emotion_id"]],
            "ParticipantID": dataset["participant_id"]
        })
    
    train_info = create_info_df(train_dataset, id_to_emotion)
    test_info = create_info_df(test_dataset, id_to_emotion)
    
    return (
        train_loader, test_loader,
        id_to_emotion, id_to_speaker,
        num_emotions, num_speakers,
        train_info, test_info
    )
