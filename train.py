"""
FrEVL Training Code - Fixed to Match Paper Architecture Exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import clip
from tqdm import tqdm
import json
import os
import random
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from PIL import Image
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class FrEVLConfig:
    hidden_dim: int = 768
    num_layers: int = 4  
    num_heads: int = 8   
    dropout: float = 0.1 
    ffn_expansion: int = 4  
    
    # Training hyperparameters
    batch_size: int = 32  
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2  
    max_epochs: int = 20
    gradient_clip: float = 1.0
    
    # Multi-objective training 
    lambda_task: float = 1.0
    lambda_con: float = 0.1
    lambda_reg: float = 0.01
    temperature: float = 0.07
    
    # Label smoothing and regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # CLIP backbone, use ViT-L/14 for FrEVL-B 
    clip_model: str = "ViT-L/14"  
    freeze_encoders: bool = True
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    max_train_samples: int = None
    max_val_samples: int = None

class FrEVL(nn.Module):
    
    def __init__(self, config: FrEVLConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained CLIP
        print(f"Loading pretrained CLIP model: {config.clip_model}")
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=config.device)
        self.clip_model.eval()
        self.clip_model = self.clip_model.float()
        
        # Freeze CLIP encoders
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get dimensions based on CLIP model
        if "ViT-B" in config.clip_model:
            self.vision_dim = self.text_dim = 512
        elif "ViT-L" in config.clip_model:
            self.vision_dim = self.text_dim = 768
        else:
            self.vision_dim = self.text_dim = 512
        
        
        # Linear projection layers 
        self.vision_proj = nn.Linear(self.vision_dim, config.hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, config.hidden_dim)
        
        # Bidirectional cross-attention mechanism 
        # L=4 transformer layers for both modalities
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * config.ffn_expansion,  
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  
        )
        
        # Separate transformers for vision and text processing
        self.vision_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Cross-modal attention layers for bidirectional information exchange
        self.cross_attention_v2t = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.cross_attention_t2v = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer normalization for cross-attention
        self.cross_norm_v = nn.LayerNorm(config.hidden_dim)
        self.cross_norm_t = nn.LayerNorm(config.hidden_dim)
        
        # Feature fusion 
        fusion_dim = config.hidden_dim * 4
        
        # 2-layer MLP prediction head 
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Initialize weights with small values 
        self._init_weights()
    
    
    def _init_weights(self):
        """Initialize weights with small values (gain=0.02)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def extract_features(self, images, texts):
        """Extract CLIP features """
        with torch.no_grad():
            # Encode images and text
            vision_features = self.clip_model.encode_image(images)
            text_tokens = clip.tokenize(texts, truncate=True).to(self.config.device)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # L2 normalize to unit hypersphere
            vision_features = F.normalize(vision_features, p=2, dim=-1).float()
            text_features = F.normalize(text_features, p=2, dim=-1).float()
        
        return vision_features, text_features
    
    def forward(self, images, texts, return_features=False):
        # Extract frozen CLIP features
        vision_features, text_features = self.extract_features(images, texts)
        
        # Project to fusion space (Equations 4-5)
        h_v = F.gelu(self.vision_proj(vision_features))  # [B, hidden_dim]
        h_t = F.gelu(self.text_proj(text_features))      # [B, hidden_dim]
        
        # Add sequence dimension for transformer processing
        h_v = h_v.unsqueeze(1)  # [B, 1, hidden_dim]
        h_t = h_t.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Self-attention within each modality (L=4 layers each)
        h_v_self = self.vision_transformer(h_v)  # [B, 1, hidden_dim]
        h_t_self = self.text_transformer(h_t)    # [B, 1, hidden_dim]
        
        # Bidirectional cross-attention (Equations 6-9)
        # Vision attends to text
        h_v_cross, _ = self.cross_attention_v2t(h_v_self, h_t_self, h_t_self)
        h_v_final = self.cross_norm_v(h_v_self + h_v_cross)  
        
        # Text attends to vision
        h_t_cross, _ = self.cross_attention_t2v(h_t_self, h_v_self, h_v_self)
        h_t_final = self.cross_norm_t(h_t_self + h_t_cross)  
        
        # Remove sequence dimension
        h_v_final = h_v_final.squeeze(1)  # [B, hidden_dim]
        h_t_final = h_t_final.squeeze(1)  # [B, hidden_dim]
        
        if return_features:
            # For mixup - return concatenated features
            return torch.cat([h_v_final, h_t_final], dim=-1)
        
        # Feature fusion 
        multiplicative = h_v_final * h_t_final           
        difference = torch.abs(h_v_final - h_t_final)   
        fused_features = torch.cat([
            h_v_final,      # Vision features
            h_t_final,      # Text features
            multiplicative, # Multiplicative interaction  
            difference      # Difference features
        ], dim=-1)
        
        output = self.prediction_head(fused_features)
        return output.squeeze(-1)
    
    def forward_mixed(self, features):
        """Forward pass for pre-mixed features"""
        # For mixup, features are already concatenated [h_v; h_t]
        # Need to split and apply fusion
        half_dim = features.size(-1) // 2
        h_v = features[..., :half_dim]
        h_t = features[..., half_dim:]
        
        # Apply comprehensive fusion
        multiplicative = h_v * h_t
        difference = torch.abs(h_v - h_t)
        
        fused_features = torch.cat([h_v, h_t, multiplicative, difference], dim=-1)
        output = self.prediction_head(fused_features)
        return output.squeeze(-1)

class LabelSmoothingBCELoss(nn.Module):
    """Binary cross entropy with label smoothing"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, outputs, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(outputs, targets_smooth)

class VQADataset(torch.utils.data.Dataset):
    """VQA Dataset with balanced sampling"""
    
    def __init__(self, questions_file, annotations_file, images_dir, 
                 transform=None, max_samples=None, augment=False):
        self.images_dir = images_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        
        print(f"Loading VQA dataset from {questions_file}")
        with open(questions_file, 'r') as f:
            questions = json.load(f)['questions']
        
        with open(annotations_file, 'r') as f:
            annotations = {ann['question_id']: ann for ann in json.load(f)['annotations']}
        
        # Balance yes/no samples
        yes_samples = []
        no_samples = []
        
        for q in tqdm(questions, desc="Loading VQA"):
            if q['question_id'] in annotations:
                ann = annotations[q['question_id']]
                answer = ann['multiple_choice_answer'].lower()
                
                if answer not in ['yes', 'no']:
                    continue
                
                image_filename = f"COCO_{'train' if 'train' in images_dir else 'val'}2014_{q['image_id']:012d}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                if os.path.exists(image_path):
                    sample = {
                        'image_path': image_path,
                        'text': q['question'],
                        'label': 1 if answer == 'yes' else 0,
                    }
                    
                    if answer == 'yes':
                        yes_samples.append(sample)
                    else:
                        no_samples.append(sample)
            
            if max_samples and len(yes_samples) + len(no_samples) >= max_samples:
                break
        
        # Balance dataset
        min_class = min(len(yes_samples), len(no_samples))
        self.samples = yes_samples[:min_class] + no_samples[:min_class]
        np.random.shuffle(self.samples)
        
        print(f"Loaded {len(self.samples)} balanced VQA samples")
        print(f"  Yes: {sum(s['label']==1 for s in self.samples)}")
        print(f"  No: {sum(s['label']==0 for s in self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'text': sample['text'],
            'label': torch.tensor(sample['label']).float(),
        }

class Trainer:
    """Trainer with multi-objective loss"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        self.optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs, eta_min=1e-6
        )
        
        # Multi-objective loss
        self.task_criterion = LabelSmoothingBCELoss(smoothing=config.label_smoothing)
        
        # Tracking
        self.train_history = {'loss': [], 'acc': [], 'f1': []}
        self.val_history = {'loss': [], 'acc': [], 'f1': []}
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def contrastive_loss(self, outputs, labels, temperature=0.07):
        batch_size = outputs.size(0)
        
        # Create positive/negative pairs
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        
        # Compute similarities
        similarities = torch.matmul(outputs.unsqueeze(1), outputs.unsqueeze(0)) / temperature
        
        # Contrastive loss
        pos_exp = torch.exp(similarities) * pos_mask.float()
        all_exp = torch.exp(similarities)
        
        loss = -torch.log(pos_exp.sum(dim=1) / all_exp.sum(dim=1)).mean()
        return loss
    
    def mixup_data(self, images, texts, labels, alpha=1.0):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = labels.size(0)
        index = torch.randperm(batch_size).to(self.config.device)
        
        # Get features for mixing
        features_a = self.model(images, texts, return_features=True)
        
        # Shuffle for mixing
        texts_b = [texts[i] for i in index.cpu().numpy()]
        images_b = images[index]
        features_b = self.model(images_b, texts_b, return_features=True)
        
        # Mix features
        mixed_features = lam * features_a + (1 - lam) * features_b
        labels_a, labels_b = labels, labels[index]
        
        return mixed_features, labels_a, labels_b, lam
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        
        running_loss = 0.0
        running_task_loss = 0.0
        running_con_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.config.device)
            texts = batch['text']
            labels = batch['label'].to(self.config.device)
            
            # Apply mixup occasionally
            use_mixup = self.config.mixup_alpha > 0 and random.random() < 0.3
            
            if use_mixup:
                mixed_features, labels_a, labels_b, lam = self.mixup_data(
                    images, texts, labels, self.config.mixup_alpha
                )
                outputs = self.model.forward_mixed(mixed_features)
                task_loss = (lam * self.task_criterion(outputs, labels_a) + 
                           (1 - lam) * self.task_criterion(outputs, labels_b))
                contrastive_loss = 0  # Skip contrastive for mixup
            else:
                outputs = self.model(images, texts)
                task_loss = self.task_criterion(outputs, labels)
                contrastive_loss = self.contrastive_loss(outputs, labels, self.config.temperature)
            
            # Multi-objective loss 
            total_loss = (self.config.lambda_task * task_loss + 
                         self.config.lambda_con * contrastive_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            # Track metrics
            running_loss += total_loss.item()
            running_task_loss += task_loss.item()
            if not use_mixup:
                running_con_loss += contrastive_loss.item()
            
            if not use_mixup:  # Only track accuracy for non-mixup samples
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if batch_idx % 10 == 0 and total > 0:
                current_acc = 100 * correct / total
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.1f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        # Epoch metrics
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * accuracy_score(all_labels, all_preds) if all_labels else 0
        epoch_f1 = f1_score(all_labels, all_preds, average='binary') if all_labels else 0
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate(self, dataloader):
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.config.device)
                texts = batch['text']
                labels = batch['label'].to(self.config.device)
                
                outputs = self.model(images, texts)
                loss = F.binary_cross_entropy_with_logits(outputs, labels)
                
                running_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(dataloader)
        val_acc = 100 * accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        val_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        
        return val_loss, val_acc, val_f1, val_precision, val_recall
    
    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, path)
        print(f"Saved best model with accuracy: {val_acc:.2f}%")
    
    def train(self, train_loader, val_loader):
        print(f"Config: LR={self.config.learning_rate}, Hidden={self.config.hidden_dim}, "
              f"Layers={self.config.num_layers}, Heads={self.config.num_heads}")
        print("-" * 70)
        
        for epoch in range(self.config.max_epochs):
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.train_history['f1'].append(train_f1)
            
            # Validation
            val_loss, val_acc, val_f1, val_prec, val_rec = self.validate(val_loader)
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)
            self.val_history['f1'].append(val_f1)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc)
            else:
                self.patience_counter += 1
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.3f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.3f}")
            print(f"        Precision: {val_prec:.3f}, Recall: {val_rec:.3f}")
            print(f"Best Val Acc: {self.best_val_acc:.2f}%, Patience: {self.patience_counter}/{self.config.patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
            
            print("-" * 70)
        
        print(f"\nTraining complete. Best accuracy: {self.best_val_acc:.2f}%")
        print(f"Model saved to: {self.config.checkpoint_dir}/best_model.pth")

def main():
    config = FrEVLConfig()
    
    # Create model
    model = FrEVL(config).to(config.device)
    
    # Data paths
    data_root = "./data"
    
    # Datasets
    train_dataset = VQADataset(
        os.path.join(data_root, 'vqa', 'v2_OpenEnded_mscoco_train2014_questions.json'),
        os.path.join(data_root, 'vqa', 'v2_mscoco_train2014_annotations.json'),
        os.path.join(data_root, 'coco', 'train2014'),
        transform=model.preprocess,
        max_samples=config.max_train_samples,
        augment=True
    )
    
    val_dataset = VQADataset(
        os.path.join(data_root, 'vqa', 'v2_OpenEnded_mscoco_val2014_questions.json'),
        os.path.join(data_root, 'vqa', 'v2_mscoco_val2014_annotations.json'),
        os.path.join(data_root, 'coco', 'val2014'),
        transform=model.preprocess,
        max_samples=config.max_val_samples,
        augment=False
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Train
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()