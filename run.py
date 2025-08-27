"""
FrEVL Complete Implementation 
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
    """Configuration with improved regularization settings"""
    # Model architecture
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.5  
    ffn_expansion: int = 2
    
    # Training hyperparameters
    batch_size: int = 64 
    learning_rate: float = 5e-4  
    weight_decay: float = 1e-4  
    warmup_steps: int = 500
    max_epochs: int = 20
    gradient_clip: float = 1.0
    
    # Regularization
    label_smoothing: float = 0.1  
    mixup_alpha: float = 0.2  
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # CLIP backbone
    clip_model: str = "ViT-B/32"
    freeze_encoders: bool = True
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    max_train_samples: int = None  # Use full dataset
    max_val_samples: int = None

class FrEVL(nn.Module):
    """FrEVL model with improved regularization"""
    
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
        
        # Get dimensions
        self.vision_dim = 512 if "ViT-B" in config.clip_model else 768
        self.text_dim = 512 if "ViT-B" in config.clip_model else 768
        
     
        self.fusion = nn.Sequential(
            nn.Linear(self.vision_dim + self.text_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout * 0.8),  
            
            nn.Linear(config.hidden_dim // 4, 1)
        )
        
        # Initialize weights with smaller values to prevent overfitting
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, images, texts):
        """Extract CLIP features with caching for efficiency"""
        with torch.no_grad():
            # Encode images
            vision_features = self.clip_model.encode_image(images)
            vision_features = F.normalize(vision_features, p=2, dim=-1).float()
            
            # Encode texts
            text_tokens = clip.tokenize(texts, truncate=True).to(self.config.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1).float()
        
        return vision_features, text_features
    
    def forward(self, images, texts, return_features=False):
        """Forward pass with optional feature return for mixup"""
        # Extract CLIP features
        vision_features, text_features = self.extract_features(images, texts)
        
        # Concatenate features
        combined = torch.cat([vision_features, text_features], dim=-1)
        
        if return_features:
            return combined
        
        # Apply fusion network
        output = self.fusion(combined)
        return output.squeeze(-1)
    
    def forward_mixed(self, features):
        """Forward pass for pre-mixed features (used in mixup)"""
        output = self.fusion(features)
        return output.squeeze(-1)

class LabelSmoothingBCELoss(nn.Module):
    """Binary cross entropy with label smoothing"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, outputs, targets):
        # Apply label smoothing
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
        
        # Load data
        print(f"Loading VQA dataset from {questions_file}")
        with open(questions_file, 'r') as f:
            questions = json.load(f)['questions']
        
        with open(annotations_file, 'r') as f:
            annotations = {ann['question_id']: ann for ann in json.load(f)['annotations']}
        
        # Separate yes/no samples for balanced loading
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
        
        # Balance the dataset
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
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Text augmentation for training
        text = sample['text']
        if self.augment and random.random() < 0.1:
            # Simple augmentation - randomly change case
            if random.random() < 0.5:
                text = text.lower()
            else:
                text = text.capitalize()
        
        return {
            'image': image,
            'text': text,
            'label': torch.tensor(sample['label']).float(),
        }

class ImprovedTrainer:
    """Trainer with regularization and early stopping"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler - reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # maximize validation accuracy
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
        
        # Loss with label smoothing
        self.criterion = LabelSmoothingBCELoss(smoothing=config.label_smoothing)
        
        self.train_history = {'loss': [], 'acc': [], 'f1': []}
        self.val_history = {'loss': [], 'acc': [], 'f1': []}
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def mixup_data(self, images, texts, labels, alpha=1.0):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = labels.size(0)
        index = torch.randperm(batch_size).to(self.config.device)
        
        # Get features for both original and shuffled
        features_orig = self.model(images, texts, return_features=True)
        
        # Shuffle texts for the mixed batch
        texts_shuffled = [texts[i] for i in index.cpu().numpy()]
        images_shuffled = images[index]
        
        features_shuffled = self.model(images_shuffled, texts_shuffled, return_features=True)
        
        # Mix features
        mixed_features = lam * features_orig + (1 - lam) * features_shuffled
        
        # Mix labels
        labels_a, labels_b = labels, labels[index]
        
        return mixed_features, labels_a, labels_b, lam
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        
        running_loss = 0.0
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
            if self.config.mixup_alpha > 0 and random.random() < 0.5:
                mixed_features, labels_a, labels_b, lam = self.mixup_data(
                    images, texts, labels, self.config.mixup_alpha
                )
                outputs = self.model.forward_mixed(mixed_features)
                loss = lam * self.criterion(outputs, labels_a) + \
                       (1 - lam) * self.criterion(outputs, labels_b)
            else:
                outputs = self.model(images, texts)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            

            if self.config.mixup_alpha == 0 or random.random() >= 0.5:
                running_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            

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
                loss = F.binary_cross_entropy_with_logits(outputs, labels)  # No smoothing for val
                
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
        print(f"  ✓ Saved best model with accuracy: {val_acc:.2f}%")
    
    def plot_history(self):
        """Plot training history"""
        epochs = range(1, len(self.train_history['loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(epochs, self.train_history['loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.val_history['loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs, self.train_history['acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.val_history['acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_history.png'))
        plt.close()
    
    def train(self, train_loader, val_loader):
        print("\nStarting training with regularization...")
        print(f"Config: LR={self.config.learning_rate}, Dropout={self.config.dropout}, "
              f"Weight Decay={self.config.weight_decay}, Label Smoothing={self.config.label_smoothing}")
        print("-" * 50)
        
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
            
            # Update learning rate based on validation accuracy
            self.scheduler.step(val_acc)
            
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
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 50)
        
        # Plot and save history
        self.plot_history()
        print(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Results saved to {self.config.checkpoint_dir}")

def main():
    # Configuration
    config = FrEVLConfig()
    
    # Create model
    print("Initializing FrEVL model with regularization...")
    model = FrEVL(config).to(config.device)
    
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    
    # Data paths
    data_root = "./data"
    
    # Create datasets
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
    
    # Create dataloaders
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
    trainer = ImprovedTrainer(model, config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()