#!/usr/bin/env python
import os
import argparse
import json
from datetime import datetime

from transformers import AutoProcessor

from config import EmotionClassificationConfig, get_dataset_config
from model import create_emotion_classifier
from dataset import load_dataset_with_perturbations
from trainer import Trainer
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train emotion classifier with N-CORE")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="recanvo",
                        help="Dataset name (recanvo, recanvo_balanced, vivae)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Override path to CSV file")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Override path to audio directory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Physical batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/hubert-base-ls960",
                        help="HuBERT model name")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained weights")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save best model")
    
    # N-CORE arguments
    parser.add_argument("--num_perturbations", type=int, default=1,
                        help="Number of perturbations per sample")
    parser.add_argument("--lambda_grl", type=float, default=0.1,
                        help="Weight for adversarial loss")
    parser.add_argument("--lambda_reg", type=float, default=0.005,
                        help="Weight for correlation regularization")
    parser.add_argument("--grl_alpha", type=float, default=1.0,
                        help="Gradient reversal scale factor")
    
    # Perturbation parameters
    parser.add_argument("--formant_shift_min", type=float, default=0.7,
                        help="Minimum formant shift ratio")
    parser.add_argument("--formant_shift_max", type=float, default=1.4,
                        help="Maximum formant shift ratio")
    parser.add_argument("--pitch_shift_min", type=float, default=0.5,
                        help="Minimum pitch shift ratio")
    parser.add_argument("--pitch_shift_max", type=float, default=2.0,
                        help="Maximum pitch shift ratio")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create config
    cfg = EmotionClassificationConfig(
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        num_perturbations=args.num_perturbations,
        lambda_grl=args.lambda_grl,
        lambda_reg=args.lambda_reg,
        grl_alpha=args.grl_alpha,
        formant_shift_range=(args.formant_shift_min, args.formant_shift_max),
        pitch_shift_range=(args.pitch_shift_min, args.pitch_shift_max)
    )
    
    # Get dataset config
    if args.csv_path and args.audio_dir:
        cfg.csv_path = args.csv_path
        cfg.audio_base_dir = args.audio_dir
        cfg.dataset_name = "custom"
    else:
        dataset_cfg = get_dataset_config(args.dataset)
        cfg.csv_path = dataset_cfg["csv_path"]
        cfg.audio_base_dir = dataset_cfg["audio_base_dir"]
        cfg.dataset_name = dataset_cfg["name"]
    
    # Set save path
    if args.save_path:
        cfg.model_save_path = args.save_path
    else:
        cfg.model_save_path = f"models/emotion_classifier_{cfg.dataset_name}_seed{args.seed}.pt"
    
    # Set perturbed audio directory
    cfg.perturbed_audio_dir = f"perturbed_audio/nansy/{cfg.dataset_name}"
    
    print("=" * 80)
    print("N-CORE: Emotion Classification with Speaker-Invariant Learning")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {cfg.dataset_name}")
    print(f"  Audio directory: {cfg.audio_base_dir}")
    print(f"  CSV path: {cfg.csv_path}")
    print(f"  Device: {cfg.device}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Effective batch size: {cfg.effective_batch_size}")
    print(f"  Number of perturbations: {cfg.num_perturbations}")
    print(f"  Lambda GRL: {cfg.lambda_grl}")
    print(f"  Lambda Reg: {cfg.lambda_reg}")
    
    # Load processor
    print("\nLoading HuBERT processor...")
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    
    # Load dataset
    print("\nLoading dataset and generating perturbations...")
    train_loader, test_loader, id_to_emotion, id_to_speaker, num_emotions, num_speakers, train_info, test_info = \
        load_dataset_with_perturbations(
            csv_path=cfg.csv_path,
            audio_base_dir=cfg.audio_base_dir,
            perturbed_dir=cfg.perturbed_audio_dir,
            num_perturbations=cfg.num_perturbations,
            perturbation_type="nansy",
            processor=processor,
            test_size=cfg.test_size,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            stratify_by="emotion",
            formant_range=cfg.formant_shift_range,
            pitch_range=cfg.pitch_shift_range,
            pitch_range_ratio_range=cfg.pitch_range_ratio_range,
            gain_range=cfg.peq_gain_range
        )
    
    print(f"\nDataset loaded:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Number of emotions: {num_emotions}")
    print(f"  Number of speakers: {num_speakers}")
    
    if not train_info.empty:
        print(f"\nEmotion distribution (train):")
        print(train_info['EmotionLabel'].value_counts().sort_index())
    
    # Create model
    print("\nInitializing model...")
    model = create_emotion_classifier(
        num_emotions=num_emotions,
        num_speakers=num_speakers,
        model_name=cfg.model_name,
        pretrained_path=cfg.pretrained_path,
        grl_alpha=cfg.grl_alpha
    )
    model.to(cfg.device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=cfg.device,
        num_primary_classes=num_emotions,
        num_adversarial_classes=num_speakers,
        id_to_primary_map=id_to_emotion,
        id_to_adversarial_map=id_to_speaker,
        primary_task="emotion"
    )
    
    # Train
    print("\nStarting training...")
    best_metrics = trainer.train(
        train_loader=train_loader,
        eval_loader=test_loader,
        num_epochs=cfg.num_epochs,
        accumulation_steps=cfg.accumulation_steps,
        lambda_grl=cfg.lambda_grl,
        lambda_reg=cfg.lambda_reg,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.early_stopping_patience,
        save_path=cfg.model_save_path
    )
    
    # Save results
    results = {
        'dataset': cfg.dataset_name,
        'seed': cfg.seed,
        'best_accuracy': best_metrics.accuracy,
        'best_f1': best_metrics.f1,
        'best_uar': best_metrics.uar,
        'best_auc': best_metrics.auc,
        'best_epoch': best_metrics.epoch,
        'speaker_accuracy_at_best': best_metrics.adversarial_accuracy,
        'config': {
            'num_epochs': cfg.num_epochs,
            'batch_size': cfg.batch_size,
            'accumulation_steps': cfg.accumulation_steps,
            'lr': cfg.lr,
            'num_perturbations': cfg.num_perturbations,
            'lambda_grl': cfg.lambda_grl,
            'lambda_reg': cfg.lambda_reg,
            'grl_alpha': cfg.grl_alpha
        }
    }
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{results_dir}/emotion_classification_{cfg.dataset_name}_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Emotion Accuracy: {best_metrics.accuracy:.4f}")
    print(f"Best Emotion F1: {best_metrics.f1:.4f}")
    print(f"Best Emotion UAR: {best_metrics.uar:.4f}")
    print(f"Best Emotion AUC: {best_metrics.auc:.4f}")
    print(f"Speaker Accuracy at best: {best_metrics.adversarial_accuracy:.4f}")
    print(f"Best Epoch: {best_metrics.epoch}")


if __name__ == "__main__":
    main()
