import os
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics import classification_report

from transformers import AdamW, get_scheduler

from utils import compute_metrics, EarlyStopping, AverageMeter


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    accuracy: float = 0.0
    f1: float = 0.0
    uar: float = 0.0
    auc: float = 0.0
    epoch: int = 0
    adversarial_accuracy: float = 0.0


class Trainer:
    """
    Trainer class for N-CORE models.
    
    Handles training loop with gradient accumulation, evaluation,
    and early stopping.
    
    Args:
        model: The model to train
        device: Device to use for training
        num_primary_classes: Number of primary task classes
        num_adversarial_classes: Number of adversarial task classes
        id_to_primary_map: Mapping from ID to primary class name
        id_to_adversarial_map: Mapping from ID to adversarial class name
        primary_task: Name of primary task ("emotion" or "speaker")
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        num_primary_classes: int,
        num_adversarial_classes: int,
        id_to_primary_map: Dict[int, str],
        id_to_adversarial_map: Dict[int, str],
        primary_task: str = "emotion"
    ):
        self.model = model
        self.device = device
        self.num_primary_classes = num_primary_classes
        self.num_adversarial_classes = num_adversarial_classes
        self.id_to_primary_map = id_to_primary_map
        self.id_to_adversarial_map = id_to_adversarial_map
        self.primary_task = primary_task
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.best_metrics = TrainingMetrics()
    
    def setup_optimizer(
        self,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        num_training_steps: int = 1000,
        warmup_ratio: float = 0.1
    ):
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay for regularization
            num_training_steps: Total number of optimizer steps
            warmup_ratio: Fraction of steps for warmup
        """
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * warmup_ratio),
            num_training_steps=num_training_steps
        )
    
    def train_epoch(
        self,
        train_loader,
        accumulation_steps: int = 1,
        lambda_grl: float = 0.1,
        lambda_reg: float = 0.005
    ) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            accumulation_steps: Number of gradient accumulation steps
            lambda_grl: Weight for adversarial loss
            lambda_reg: Weight for correlation loss
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Metrics accumulators
        loss_meter = AverageMeter("Loss")
        primary_loss_meter = AverageMeter("Primary Loss")
        adversarial_loss_meter = AverageMeter("Adversarial Loss")
        correlation_loss_meter = AverageMeter("Correlation Loss")
        
        primary_preds, primary_true, primary_scores = [], [], []
        adversarial_preds, adversarial_true = [], []
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
        for batch_idx, batch in pbar:
            # Move data to device
            inputs = batch["input_values"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            list_pert_inputs = [p.to(self.device, non_blocking=True) for p in batch["list_of_perturbed_input_values"]]
            list_pert_masks = [m.to(self.device, non_blocking=True) for m in batch["list_of_perturbed_attention_masks"]]
            
            # Get labels based on task
            if self.primary_task == "emotion":
                primary_labels = batch["emotion_labels"].to(self.device, non_blocking=True)
                adversarial_labels = batch["speaker_labels"].to(self.device, non_blocking=True)
            else:
                primary_labels = batch["speaker_labels"].to(self.device, non_blocking=True)
                adversarial_labels = batch["emotion_labels"].to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(
                input_values=inputs,
                attention_mask=attn_mask,
                list_of_perturbed_input_values=list_pert_inputs,
                list_of_perturbed_attention_masks=list_pert_masks,
                primary_labels=primary_labels,
                adversarial_labels=adversarial_labels,
                lambda_grl=lambda_grl,
                lambda_reg=lambda_reg
            )
            
            loss = outputs.get("loss")
            if loss is None:
                continue
            
            # Scale loss for accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            # Update metrics
            loss_meter.update(loss.item())
            primary_loss_meter.update(outputs.get("primary_loss", torch.tensor(0.0)).item())
            adversarial_loss_meter.update(outputs.get("adversarial_loss", torch.tensor(0.0)).item())
            correlation_loss_meter.update(outputs.get("correlation_loss", torch.tensor(0.0)).item())
            
            # Optimizer step
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Collect predictions
            primary_logits = outputs["primary_logits"]
            primary_preds.extend(torch.argmax(primary_logits.detach(), dim=-1).cpu().tolist())
            primary_true.extend(primary_labels.cpu().tolist())
            primary_scores.extend(torch.softmax(primary_logits.detach(), dim=-1).cpu().numpy())
            
            adversarial_logits = outputs["adversarial_logits"]
            adversarial_preds.extend(torch.argmax(adversarial_logits.detach(), dim=-1).cpu().tolist())
            adversarial_true.extend(adversarial_labels.cpu().tolist())
            
            pbar.set_postfix({
                'L': loss.item(),
                'PriL': outputs.get("primary_loss", torch.tensor(0.0)).item(),
                'AdvL': outputs.get("adversarial_loss", torch.tensor(0.0)).item()
            })
        
        # Compute metrics
        primary_metrics = compute_metrics(
            primary_true, primary_preds, primary_scores, self.num_primary_classes
        )
        adversarial_metrics = compute_metrics(
            adversarial_true, adversarial_preds, None, self.num_adversarial_classes
        )
        
        return {
            'loss': loss_meter.avg,
            'primary_loss': primary_loss_meter.avg,
            'adversarial_loss': adversarial_loss_meter.avg,
            'correlation_loss': correlation_loss_meter.avg,
            'primary_accuracy': primary_metrics['accuracy'],
            'primary_f1': primary_metrics['f1'],
            'primary_uar': primary_metrics['uar'],
            'primary_auc': primary_metrics['auc'],
            'adversarial_accuracy': adversarial_metrics['accuracy'],
            'adversarial_f1': adversarial_metrics['f1']
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_loader,
        lambda_grl: float = 0.1,
        lambda_reg: float = 0.005,
        print_report: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            eval_loader: Evaluation data loader
            lambda_grl: Weight for adversarial loss
            lambda_reg: Weight for correlation loss
            print_report: Whether to print classification report
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter("Loss")
        primary_loss_meter = AverageMeter("Primary Loss")
        adversarial_loss_meter = AverageMeter("Adversarial Loss")
        correlation_loss_meter = AverageMeter("Correlation Loss")
        
        primary_preds, primary_true, primary_scores = [], [], []
        adversarial_preds, adversarial_true = [], []
        
        pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            inputs = batch["input_values"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            list_pert_inputs = [p.to(self.device, non_blocking=True) for p in batch["list_of_perturbed_input_values"]]
            list_pert_masks = [m.to(self.device, non_blocking=True) for m in batch["list_of_perturbed_attention_masks"]]
            
            if self.primary_task == "emotion":
                primary_labels = batch["emotion_labels"].to(self.device, non_blocking=True)
                adversarial_labels = batch["speaker_labels"].to(self.device, non_blocking=True)
            else:
                primary_labels = batch["speaker_labels"].to(self.device, non_blocking=True)
                adversarial_labels = batch["emotion_labels"].to(self.device, non_blocking=True)
            
            outputs = self.model(
                input_values=inputs,
                attention_mask=attn_mask,
                list_of_perturbed_input_values=list_pert_inputs,
                list_of_perturbed_attention_masks=list_pert_masks,
                primary_labels=primary_labels,
                adversarial_labels=adversarial_labels,
                lambda_grl=lambda_grl,
                lambda_reg=lambda_reg
            )
            
            loss = outputs.get("loss")
            if loss is not None:
                loss_meter.update(loss.item())
            
            primary_loss_meter.update(outputs.get("primary_loss", torch.tensor(0.0)).item())
            adversarial_loss_meter.update(outputs.get("adversarial_loss", torch.tensor(0.0)).item())
            correlation_loss_meter.update(outputs.get("correlation_loss", torch.tensor(0.0)).item())
            
            primary_logits = outputs["primary_logits"]
            primary_preds.extend(torch.argmax(primary_logits, dim=-1).cpu().tolist())
            primary_true.extend(primary_labels.cpu().tolist())
            primary_scores.extend(torch.softmax(primary_logits, dim=-1).cpu().numpy())
            
            adversarial_logits = outputs["adversarial_logits"]
            adversarial_preds.extend(torch.argmax(adversarial_logits, dim=-1).cpu().tolist())
            adversarial_true.extend(adversarial_labels.cpu().tolist())
        
        primary_metrics = compute_metrics(
            primary_true, primary_preds, primary_scores, self.num_primary_classes
        )
        adversarial_metrics = compute_metrics(
            adversarial_true, adversarial_preds, None, self.num_adversarial_classes
        )
        
        if print_report and primary_true and primary_preds:
            print(f"\n{self.primary_task.capitalize()} Classification Report (Primary Task):")
            unique_labels = sorted(list(set(primary_true) | set(primary_preds)))
            target_names = [self.id_to_primary_map.get(lid, f"ID {lid}") for lid in unique_labels]
            print(classification_report(
                primary_true, primary_preds,
                target_names=target_names,
                digits=4, zero_division=0
            ))
        
        return {
            'loss': loss_meter.avg,
            'primary_loss': primary_loss_meter.avg,
            'adversarial_loss': adversarial_loss_meter.avg,
            'correlation_loss': correlation_loss_meter.avg,
            'primary_accuracy': primary_metrics['accuracy'],
            'primary_f1': primary_metrics['f1'],
            'primary_uar': primary_metrics['uar'],
            'primary_auc': primary_metrics['auc'],
            'adversarial_accuracy': adversarial_metrics['accuracy'],
            'adversarial_f1': adversarial_metrics['f1']
        }
    
    def train(
        self,
        train_loader,
        eval_loader,
        num_epochs: int = 100,
        accumulation_steps: int = 4,
        lambda_grl: float = 0.1,
        lambda_reg: float = 0.005,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        patience: int = 10,
        save_path: Optional[str] = None
    ) -> TrainingMetrics:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            num_epochs: Maximum number of epochs
            accumulation_steps: Gradient accumulation steps
            lambda_grl: Weight for adversarial loss
            lambda_reg: Weight for correlation loss
            lr: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            save_path: Path to save best model
            
        Returns:
            Best training metrics
        """
        # Calculate total optimizer steps
        num_optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        total_optimizer_steps = num_optimizer_steps_per_epoch * num_epochs
        
        # Setup optimizer
        self.setup_optimizer(lr, weight_decay, total_optimizer_steps)
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=patience, mode='max')
        
        print(f"\nTraining Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Primary task: {self.primary_task}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size (effective): {train_loader.batch_size * accumulation_steps}")
        print(f"  Lambda GRL: {lambda_grl}")
        print(f"  Lambda Reg: {lambda_reg}")
        print(f"  Primary classes: {self.num_primary_classes}")
        print(f"  Adversarial classes: {self.num_adversarial_classes}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, accumulation_steps, lambda_grl, lambda_reg
            )
            
            print(f"TRAIN - Loss: {train_metrics['loss']:.4f} | "
                  f"Primary: Acc={train_metrics['primary_accuracy']:.4f} "
                  f"F1={train_metrics['primary_f1']:.4f} UAR={train_metrics['primary_uar']:.4f} | "
                  f"Adversarial: Acc={train_metrics['adversarial_accuracy']:.4f}")
            
            # Evaluate
            eval_metrics = self.evaluate(
                eval_loader, lambda_grl, lambda_reg, print_report=True
            )
            
            print(f"VALID - Loss: {eval_metrics['loss']:.4f} | "
                  f"Primary: Acc={eval_metrics['primary_accuracy']:.4f} "
                  f"F1={eval_metrics['primary_f1']:.4f} UAR={eval_metrics['primary_uar']:.4f} "
                  f"AUC={eval_metrics['primary_auc']:.4f} | "
                  f"Adversarial: Acc={eval_metrics['adversarial_accuracy']:.4f}")
            
            # Check for improvement
            current_score = eval_metrics['primary_accuracy']
            if current_score > self.best_metrics.accuracy:
                self.best_metrics = TrainingMetrics(
                    accuracy=eval_metrics['primary_accuracy'],
                    f1=eval_metrics['primary_f1'],
                    uar=eval_metrics['primary_uar'],
                    auc=eval_metrics['primary_auc'],
                    epoch=epoch + 1,
                    adversarial_accuracy=eval_metrics['adversarial_accuracy']
                )
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")
            
            # Early stopping check
            if early_stopping(current_score, epoch + 1):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        print(f"\nTraining completed!")
        print(f"Best {self.primary_task} accuracy: {self.best_metrics.accuracy:.4f} at epoch {self.best_metrics.epoch}")
        
        return self.best_metrics
