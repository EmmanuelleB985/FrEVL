"""
FrEVL Dataset Download and Preparation Script
Downloads and prepares COCO Captions, VQA v2, and SNLI-VE datasets
"""

import os
import json
import wget
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import hashlib
import requests
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# ==========================================
# Dataset URLs and Configuration
# ==========================================

DATASET_URLS = {
    'coco': {
        'train_images': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2014.zip',
        'test_images': 'http://images.cocodataset.org/zips/test2015.zip',
        'train_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'test_annotations': 'http://images.cocodataset.org/annotations/image_info_test2015.zip',
        'karpathy_split': 'https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip'
    },
    'vqa': {
        'train_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
        'val_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
        'test_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip',
        'train_annotations': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
        'val_annotations': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
        'balanced_pairs': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip'
    },
    'snli_ve': {
        'dataset': 'https://github.com/necla-ml/SNLI-VE/archive/master.zip',
        'flickr30k_images': 'https://forms.illinois.edu/sec/229675',  # Requires manual download
        'splits': {
            'train': 'https://raw.githubusercontent.com/necla-ml/SNLI-VE/master/data/snli_ve_train.jsonl',
            'dev': 'https://raw.githubusercontent.com/necla-ml/SNLI-VE/master/data/snli_ve_dev.jsonl',
            'test': 'https://raw.githubusercontent.com/necla-ml/SNLI-VE/master/data/snli_ve_test.jsonl'
        }
    }
}


class DatasetDownloader:
    """Handle dataset downloading and extraction"""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, dest_path: str, desc: str = None) -> bool:
        """Download file with progress bar"""
        if os.path.exists(dest_path):
            print(f"✓ {dest_path} already exists, skipping download")
            return True
        
        print(f"Downloading {desc or url}...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"✗ Error downloading {url}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    
    def extract_archive(self, archive_path: str, extract_to: str):
        """Extract zip or tar archive"""
        print(f"Extracting {archive_path}...")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.namelist(), desc="Extracting"):
                    zip_ref.extract(member, extract_to)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path}")
    
    def verify_file(self, file_path: str, expected_size_mb: Optional[int] = None) -> bool:
        """Verify downloaded file"""
        if not os.path.exists(file_path):
            return False
        
        actual_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        if expected_size_mb:
            if abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.1:  # 10% tolerance
                print(f"⚠ Size mismatch for {file_path}: {actual_size_mb:.1f}MB vs {expected_size_mb}MB expected")
                return False
        
        print(f"Verified {file_path} ({actual_size_mb:.1f}MB)")
        return True

# ==========================================
# COCO Dataset Preparation
# ==========================================

class COCODatasetPrep:
    """Prepare COCO Captions dataset"""
    
    def __init__(self, data_dir: str = './data/coco'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = DatasetDownloader(data_dir)
    
    def download(self):
        """Download COCO dataset"""
        print("\n" + "="*50)
        print("Downloading COCO Captions Dataset")
        print("="*50)
        
        # Create directories
        for split in ['train2014', 'val2014', 'test2015', 'annotations']:
            (self.data_dir / split).mkdir(exist_ok=True)
        
        # Download images
        datasets = [
            ('train_images', 'train2014.zip'),
            ('val_images', 'val2014.zip'),
            ('test_images', 'test2015.zip'),
            ('train_annotations', 'annotations_trainval2014.zip')
        ]
        
        for key, filename in datasets:
            url = DATASET_URLS['coco'][key]
            dest_path = self.data_dir / filename
            
            if self.downloader.download_file(url, str(dest_path), f"COCO {key}"):
                self.downloader.extract_archive(str(dest_path), str(self.data_dir))
        
        print("COCO download complete")
    
    def prepare_splits(self):
        """Prepare train/val/test splits following Karpathy split"""
        print("\nPreparing COCO splits...")
        
        # Load annotations
        ann_file = self.data_dir / 'annotations' / 'captions_train2014.json'
        if not ann_file.exists():
            print("✗ Annotations not found. Please run download first.")
            return
        
        with open(ann_file, 'r') as f:
            train_anns = json.load(f)
        
        ann_file = self.data_dir / 'annotations' / 'captions_val2014.json'
        with open(ann_file, 'r') as f:
            val_anns = json.load(f)
        
        # Create unified dataset file
        dataset = {
            'images': train_anns['images'] + val_anns['images'],
            'annotations': train_anns['annotations'] + val_anns['annotations']
        }
        
        # Create image id to file mapping
        id_to_file = {img['id']: img['file_name'] for img in dataset['images']}
        
        # Group annotations by image
        img_to_caps = {}
        for ann in dataset['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_caps:
                img_to_caps[img_id] = []
            img_to_caps[img_id].append(ann['caption'])
        
        # Create split files
        splits = {
            'train': list(img_to_caps.keys())[:113287],
            'val': list(img_to_caps.keys())[113287:113287+5000],
            'test': list(img_to_caps.keys())[113287+5000:113287+10000]
        }
        
        for split_name, img_ids in splits.items():
            split_data = []
            for img_id in img_ids:
                if img_id in img_to_caps:
                    for caption in img_to_caps[img_id]:
                        split_data.append({
                            'image_id': img_id,
                            'image_file': id_to_file.get(img_id, ''),
                            'caption': caption
                        })
            
            # Save split file
            output_file = self.data_dir / f'{split_name}_captions.json'
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            print(f"Created {split_name} split with {len(split_data)} caption pairs")
    
    def create_cached_embeddings(self):
        """Pre-compute and cache CLIP embeddings for efficiency"""
        print("\nPre-computing CLIP embeddings for COCO...")
        
        try:
            import clip
            import torch
            from PIL import Image
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            cache_dir = self.data_dir / 'cached_embeddings'
            cache_dir.mkdir(exist_ok=True)
            
            # Process each split
            for split in ['train', 'val', 'test']:
                split_file = self.data_dir / f'{split}_captions.json'
                if not split_file.exists():
                    continue
                
                with open(split_file, 'r') as f:
                    data = json.load(f)
                
                embeddings = {
                    'image_embeddings': {},
                    'text_embeddings': []
                }
                
                print(f"Processing {split} split...")
                
                # Cache unique image embeddings
                unique_images = list(set([item['image_file'] for item in data]))
                
                for img_file in tqdm(unique_images[:100], desc="Images"):  # Limit for demo
                    img_path = self.data_dir / 'train2014' / img_file
                    if not img_path.exists():
                        img_path = self.data_dir / 'val2014' / img_file
                    
                    if img_path.exists():
                        image = Image.open(img_path).convert('RGB')
                        image_input = preprocess(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            image_features = model.encode_image(image_input)
                            embeddings['image_embeddings'][img_file] = image_features.cpu().numpy()
                
                # Cache text embeddings
                for item in tqdm(data[:1000], desc="Captions"):  # Limit for demo
                    text = clip.tokenize([item['caption']], truncate=True).to(device)
                    
                    with torch.no_grad():
                        text_features = model.encode_text(text)
                        embeddings['text_embeddings'].append({
                            'caption': item['caption'],
                            'embedding': text_features.cpu().numpy()
                        })
                
                # Save embeddings
                cache_file = cache_dir / f'{split}_embeddings.npz'
                np.savez_compressed(cache_file, **embeddings)
                print(f"Cached {len(embeddings['image_embeddings'])} image embeddings")
                
        except ImportError:
            print("CLIP not installed. Skipping embedding cache.")

# ==========================================
# VQA v2 Dataset Preparation
# ==========================================

class VQADatasetPrep:
    """Prepare VQA v2 dataset"""
    
    def __init__(self, data_dir: str = './data/vqa'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = DatasetDownloader(data_dir)
    
    def download(self):
        """Download VQA v2 dataset"""
        print("\n" + "="*50)
        print("Downloading VQA v2 Dataset")
        print("="*50)
        
        # Download questions and annotations
        files_to_download = [
            ('train_questions', 'v2_Questions_Train_mscoco.zip'),
            ('val_questions', 'v2_Questions_Val_mscoco.zip'),
            ('test_questions', 'v2_Questions_Test_mscoco.zip'),
            ('train_annotations', 'v2_Annotations_Train_mscoco.zip'),
            ('val_annotations', 'v2_Annotations_Val_mscoco.zip')
        ]
        
        for key, filename in files_to_download:
            url = DATASET_URLS['vqa'][key]
            dest_path = self.data_dir / filename
            
            if self.downloader.download_file(url, str(dest_path), f"VQA {key}"):
                self.downloader.extract_archive(str(dest_path), str(self.data_dir))
        
        print("VQA v2 download complete")
        print("Note: VQA uses COCO images. Make sure COCO is downloaded.")
    
    def prepare_dataset(self):
        """Prepare VQA dataset for training"""
        print("\nPreparing VQA dataset...")
        
        # Load questions
        for split in ['train', 'val']:
            q_file = self.data_dir / f'v2_OpenEnded_mscoco_{split}2014_questions.json'
            a_file = self.data_dir / f'v2_mscoco_{split}2014_annotations.json'
            
            if not q_file.exists() or not a_file.exists():
                print(f"Missing files for {split} split")
                continue
            
            with open(q_file, 'r') as f:
                questions = json.load(f)['questions']
            
            with open(a_file, 'r') as f:
                annotations = json.load(f)['annotations']
            
            # Combine questions and answers
            qa_pairs = []
            for q, a in zip(questions, annotations):
                qa_pairs.append({
                    'question_id': q['question_id'],
                    'image_id': q['image_id'],
                    'question': q['question'],
                    'answers': [ans['answer'] for ans in a['answers']],
                    'answer_type': a.get('answer_type', 'unknown'),
                    'question_type': a.get('question_type', 'unknown')
                })
            
            # Save processed dataset
            output_file = self.data_dir / f'{split}_qa.json'
            with open(output_file, 'w') as f:
                json.dump(qa_pairs, f, indent=2)
            
            print(f"Processed {split} split: {len(qa_pairs)} QA pairs")
    
    def create_answer_vocabulary(self):
        """Create answer vocabulary for classification"""
        print("\nCreating answer vocabulary...")
        
        all_answers = []
        for split in ['train', 'val']:
            qa_file = self.data_dir / f'{split}_qa.json'
            if not qa_file.exists():
                continue
            
            with open(qa_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                all_answers.extend(item['answers'])
        
        # Count answer frequencies
        from collections import Counter
        answer_counts = Counter(all_answers)
        
        # Keep top 3000 answers (following paper)
        vocab = ['<unk>'] + [ans for ans, _ in answer_counts.most_common(3000)]
        
        # Create answer to index mapping
        ans2idx = {ans: idx for idx, ans in enumerate(vocab)}
        
        # Save vocabulary
        vocab_file = self.data_dir / 'answer_vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump({'vocab': vocab, 'ans2idx': ans2idx}, f, indent=2)
        
        print(f"✓ Created vocabulary with {len(vocab)} answers")

# ==========================================
# SNLI-VE Dataset Preparation
# ==========================================

class SNLIVEDatasetPrep:
    """Prepare SNLI-VE dataset"""
    
    def __init__(self, data_dir: str = './data/snli-ve'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = DatasetDownloader(data_dir)
    
    def download(self):
        """Download SNLI-VE dataset"""
        print("\n" + "="*50)
        print("Downloading SNLI-VE Dataset")
        print("="*50)
        
        # Download SNLI-VE annotations
        for split_name, url in DATASET_URLS['snli_ve']['splits'].items():
            dest_path = self.data_dir / f'snli_ve_{split_name}.jsonl'
            self.downloader.download_file(url, str(dest_path), f"SNLI-VE {split_name}")
        
        print("\n" + "!"*50)
        print("IMPORTANT: Flickr30k images required!")
        print("Please download Flickr30k images manually from:")
        print("https://forms.illinois.edu/sec/229675")
        print("Extract to: data/snli-ve/flickr30k_images/")
        print("!"*50)
        
        # Create placeholder directory
        (self.data_dir / 'flickr30k_images').mkdir(exist_ok=True)
        
        print("✓ SNLI-VE annotations downloaded")
    
    def prepare_dataset(self):
        """Prepare SNLI-VE dataset"""
        print("\nPreparing SNLI-VE dataset...")
        
        label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        
        for split in ['train', 'dev', 'test']:
            input_file = self.data_dir / f'snli_ve_{split}.jsonl'
            if not input_file.exists():
                print(f"⚠ Missing {split} split file")
                continue
            
            data = []
            with open(input_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if item['gold_label'] in label_map:
                        data.append({
                            'image': item['Flickr30K_ID'] + '.jpg',
                            'hypothesis': item['sentence2'],
                            'label': label_map[item['gold_label']],
                            'label_text': item['gold_label']
                        })
            
            # Save processed data
            output_file = self.data_dir / f'{split}_processed.json'
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Processed {split} split: {len(data)} examples")
            
            # Print label distribution
            if data:
                labels = [d['label'] for d in data]
                unique, counts = np.unique(labels, return_counts=True)
                print(f"  Label distribution: {dict(zip(unique, counts))}")

# ==========================================
# Dataset Statistics and Verification
# ==========================================

class DatasetVerifier:
    """Verify dataset integrity and print statistics"""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
    
    def verify_coco(self):
        """Verify COCO dataset"""
        print("\n" + "="*50)
        print("COCO Dataset Verification")
        print("="*50)
        
        coco_dir = self.data_dir / 'coco'
        
        # Check images
        image_dirs = ['train2014', 'val2014']
        total_images = 0
        for img_dir in image_dirs:
            dir_path = coco_dir / img_dir
            if dir_path.exists():
                num_images = len(list(dir_path.glob('*.jpg')))
                print(f"✓ {img_dir}: {num_images} images")
                total_images += num_images
        
        # Check annotations
        for split in ['train', 'val', 'test']:
            ann_file = coco_dir / f'{split}_captions.json'
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                print(f"✓ {split} captions: {len(data)} pairs")
        
        print(f"\nTotal images: {total_images}")
    
    def verify_vqa(self):
        """Verify VQA dataset"""
        print("\n" + "="*50)
        print("VQA v2 Dataset Verification")
        print("="*50)
        
        vqa_dir = self.data_dir / 'vqa'
        
        for split in ['train', 'val']:
            qa_file = vqa_dir / f'{split}_qa.json'
            if qa_file.exists():
                with open(qa_file, 'r') as f:
                    data = json.load(f)
                print(f"✓ {split}: {len(data)} QA pairs")
                
                # Sample statistics
                if data:
                    q_types = [d.get('question_type', 'unknown') for d in data[:1000]]
                    unique_types = list(set(q_types))[:5]
                    print(f"  Sample question types: {unique_types}")
        
        # Check vocabulary
        vocab_file = vqa_dir / 'answer_vocab.json'
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            print(f"✓ Answer vocabulary: {len(vocab['vocab'])} answers")
    
    def verify_snli_ve(self):
        """Verify SNLI-VE dataset"""
        print("\n" + "="*50)
        print("SNLI-VE Dataset Verification")
        print("="*50)
        
        snli_dir = self.data_dir / 'snli-ve'
        
        for split in ['train', 'dev', 'test']:
            data_file = snli_dir / f'{split}_processed.json'
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                print(f"✓ {split}: {len(data)} examples")
        
        # Check Flickr30k images
        flickr_dir = snli_dir / 'flickr30k_images'
        if flickr_dir.exists():
            num_images = len(list(flickr_dir.glob('*.jpg')))
            if num_images > 0:
                print(f"Flickr30k images: {num_images} found")
            else:
                print("No Flickr30k images found. Please download manually.")
        else:
            print("Flickr30k images directory not found")


def main():
    """Main dataset preparation script"""
    
    print("="*60)
    print("FrEVL Dataset Preparation")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Download and prepare datasets for FrEVL')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, choices=['coco', 'vqa', 'snli-ve', 'all'],
                       default='all', help='Which dataset to download')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download, only prepare')
    parser.add_argument('--cache-embeddings', action='store_true',
                       help='Pre-compute CLIP embeddings')
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and prepare datasets
    if args.dataset in ['coco', 'all']:
        coco_prep = COCODatasetPrep(str(data_dir / 'coco'))
        if not args.skip_download:
            coco_prep.download()
        coco_prep.prepare_splits()
        if args.cache_embeddings:
            coco_prep.create_cached_embeddings()
    
    if args.dataset in ['vqa', 'all']:
        vqa_prep = VQADatasetPrep(str(data_dir / 'vqa'))
        if not args.skip_download:
            vqa_prep.download()
        vqa_prep.prepare_dataset()
        vqa_prep.create_answer_vocabulary()
    
    if args.dataset in ['snli-ve', 'all']:
        snli_prep = SNLIVEDatasetPrep(str(data_dir / 'snli-ve'))
        if not args.skip_download:
            snli_prep.download()
        snli_prep.prepare_dataset()
    
    # Verify datasets
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    verifier = DatasetVerifier(args.data_dir)
    if args.dataset in ['coco', 'all']:
        verifier.verify_coco()
    if args.dataset in ['vqa', 'all']:
        verifier.verify_vqa()
    if args.dataset in ['snli-ve', 'all']:
        verifier.verify_snli_ve()
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    

if __name__ == "__main__":
    main()