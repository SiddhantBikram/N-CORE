import os
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from transformers import HubertModel


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRL(nn.Module):    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class HubertClassifierWithGRL(nn.Module):
    def __init__(
        self,
        num_primary_classes: int,
        num_adversarial_classes: int,
        model_name: str = "facebook/hubert-base-ls960",
        pretrained_path: Optional[str] = None,
        grl_alpha: float = 1.0,
        primary_task: str = "emotion"
    ):
        super().__init__()
        
        self.primary_task = primary_task
        
        # Load HuBERT encoder
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained HuBERT encoder from {pretrained_path}")
            self.hubert = HubertModel.from_pretrained(pretrained_path)
        else:
            print(f"Loading HuBERT encoder from Hugging Face: {model_name}")
            self.hubert = HubertModel.from_pretrained(model_name)
        
        hidden_size = self.hubert.config.hidden_size
        
        # Primary task classifier
        self.primary_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_primary_classes)
        )
        
        # Gradient Reversal Layer
        self.grl = GRL(alpha=grl_alpha)
        
        # Adversarial task classifier
        self.adversarial_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_adversarial_classes)
        )
    
    def get_embeddings(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract pooled embeddings from HuBERT."""
        hubert_outputs = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = hubert_outputs.last_hidden_state
        pooled_embeddings = torch.mean(hidden_states, dim=1)
        return pooled_embeddings
    
    def compute_correlation_loss(
        self,
        all_embeddings: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:

        if len(all_embeddings) <= 1:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        stacked_embeddings = torch.stack(all_embeddings, dim=1)
        num_samples = stacked_embeddings.shape[1]
        
        pairwise_losses = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                emb_i = stacked_embeddings[:, i, :]
                emb_j = stacked_embeddings[:, j, :]
                diff_sq_norm = torch.norm(emb_i - emb_j, p=2, dim=1).pow(2)
                pairwise_losses.append(diff_sq_norm)
        
        if not pairwise_losses:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        all_pairwise_tensor = torch.stack(pairwise_losses, dim=1)
        mean_pairwise_per_sample = torch.mean(all_pairwise_tensor, dim=1)
        correlation_loss = torch.mean(mean_pairwise_per_sample)
        
        return correlation_loss
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        list_of_perturbed_input_values: Optional[List[torch.Tensor]] = None,
        list_of_perturbed_attention_masks: Optional[List[torch.Tensor]] = None,
        primary_labels: Optional[torch.Tensor] = None,
        adversarial_labels: Optional[torch.Tensor] = None,
        lambda_grl: float = 0.1,
        lambda_reg: float = 0.005
    ) -> Dict[str, Any]:

        outputs = {}
        
        # Get embeddings from original audio
        pooled_embeddings_original = self.get_embeddings(input_values, attention_mask)
        
        # Primary task logits
        primary_logits = self.primary_classifier(pooled_embeddings_original)
        outputs["primary_logits"] = primary_logits
        
        # Adversarial task logits (through GRL)
        reversed_embeddings = self.grl(pooled_embeddings_original)
        adversarial_logits = self.adversarial_classifier(reversed_embeddings)
        outputs["adversarial_logits"] = adversarial_logits
        
        # Store embeddings
        outputs["embeddings"] = pooled_embeddings_original
        
        # Collect all embeddings for correlation loss
        all_embeddings = [pooled_embeddings_original]
        
        # Process perturbed inputs
        if list_of_perturbed_input_values and len(list_of_perturbed_input_values) > 0:
            for i, perturbed_input in enumerate(list_of_perturbed_input_values):
                perturbed_mask = None
                if list_of_perturbed_attention_masks and i < len(list_of_perturbed_attention_masks):
                    perturbed_mask = list_of_perturbed_attention_masks[i]
                
                perturbed_embeddings = self.get_embeddings(perturbed_input, perturbed_mask)
                all_embeddings.append(perturbed_embeddings)
        
        # Compute correlation loss
        correlation_loss = self.compute_correlation_loss(
            all_embeddings, input_values.device, input_values.dtype
        )
        outputs["correlation_loss"] = correlation_loss
        
        # Compute losses if labels provided
        total_loss = torch.tensor(0.0, device=input_values.device, dtype=input_values.dtype)
        
        if primary_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            primary_loss = loss_fn(primary_logits, primary_labels)
            outputs["primary_loss"] = primary_loss
            total_loss = total_loss + primary_loss
        
        if adversarial_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            adversarial_loss = loss_fn(adversarial_logits, adversarial_labels)
            outputs["adversarial_loss"] = adversarial_loss
            # Subtract because GRL already reverses gradients
            total_loss = total_loss - lambda_grl * adversarial_loss
        
        # Add correlation regularization
        total_loss = total_loss + lambda_reg * correlation_loss
        
        if primary_labels is not None or adversarial_labels is not None:
            outputs["loss"] = total_loss
        
        return outputs


# Convenience aliases for backward compatibility
def create_emotion_classifier(
    num_emotions: int,
    num_speakers: int,
    model_name: str = "facebook/hubert-base-ls960",
    pretrained_path: Optional[str] = None,
    grl_alpha: float = 1.0
) -> HubertClassifierWithGRL:
    """Create a classifier for emotion recognition with speaker adversarial."""
    return HubertClassifierWithGRL(
        num_primary_classes=num_emotions,
        num_adversarial_classes=num_speakers,
        model_name=model_name,
        pretrained_path=pretrained_path,
        grl_alpha=grl_alpha,
        primary_task="emotion"
    )


def create_speaker_classifier(
    num_speakers: int,
    num_emotions: int,
    model_name: str = "facebook/hubert-base-ls960",
    pretrained_path: Optional[str] = None,
    grl_alpha: float = 1.0
) -> HubertClassifierWithGRL:
    """Create a classifier for speaker recognition with emotion adversarial."""
    return HubertClassifierWithGRL(
        num_primary_classes=num_speakers,
        num_adversarial_classes=num_emotions,
        model_name=model_name,
        pretrained_path=pretrained_path,
        grl_alpha=grl_alpha,
        primary_task="speaker"
    )
