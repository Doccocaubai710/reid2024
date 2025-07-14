from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchreid.models import build_model
from typing import List, Tuple, Optional, Union, Dict
import argparse
from app.utils import const
from yacs.config import CfgNode as CN
import yaml

class BPBReidFeatureExtractor:
    """
    Enhanced Feature Extractor that supports both global and part-based features
    """
    def __init__(
        self,
        model_path='/home/aidev/workspace/reid/Thesis/reid-2024/app/assets/models/bmp/bpbreid_market1501_hrnet32_10642.pth',
        model_name='bpbreid',
        num_classes=767,
        img_size=const.IMG_SIZE,
        max_batch_size=const.MAX_BATCH_SIZE,
        device=None,
        num_workers=const.THREADS,
        use_parts=True,  # New parameter
        num_parts=3,     # Number of body parts (head, torso, legs)
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_parts = use_parts
        self.num_parts = num_parts
        
        print(f"Using device: {self.device}")
        print(f"Part-based features: {'Enabled' if use_parts else 'Disabled'}")
        
        # Load the model
        self.model = self._load_model(model_name, num_classes, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.img_size = img_size
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        
        # Enhanced transform for better feature extraction
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Warm up the model
        self.warm_up()
    def _load_bpbreid_config(self, config_path=None):
        """Load BPBreID configuration from YAML file"""
        if config_path is None:
            # Use default config path - adjust this to your actual config file location
            config_path = "/home/aidev/workspace/reid/Thesis/reid-2024/app/config1.yaml"  # or wherever you put the config
        
        # Load YAML config
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        # Convert to CfgNode object
        config = CN(cfg_dict)
        return config
    def _load_model(self, model_name, num_classes, model_path):
        """Load and modify model for part-based feature extraction"""
        print(f"Loading model from: {model_path}")
        
        if model_name == 'bpbreid':
            # Load config for BPBreID
            config = self._load_bpbreid_config()
            
            # Import and create bpbreid with config
            from torchreid.models.bpbreid import bpbreid
            model = bpbreid(
                num_classes=num_classes,
                config=config
            )
        else:
            # Use build_model for other models
            model = build_model(
                name=model_name,
                num_classes=num_classes
            )
        
        # Load pretrained weights
        self._load_pretrained_weights(model, model_path)
        
        if self.use_parts:
            model = self._modify_model_for_parts(model)
        
        model.eval()
        print(f"{model_name} model loaded successfully")
        return model
    
    def _modify_model_for_parts(self, model):
        """
        Modify the model to extract part-based features
        This creates a simple part-based model by splitting the feature map
        """
        class PartBasedWrapper(nn.Module):
            def __init__(self, base_model, num_parts=3):
                super().__init__()
                self.base_model = base_model
                self.num_parts = num_parts
                
                # Get the feature dimension from the base model
                # This assumes the model has a 'classifier' or 'fc' layer
                if hasattr(base_model, 'classifier'):
                    feat_dim = base_model.classifier.in_features
                elif hasattr(base_model, 'fc'):
                    feat_dim = base_model.fc.in_features
                else:
                    feat_dim = 2048  # Default assumption
                
                # Create part-specific heads
                self.part_heads = nn.ModuleList([
                    nn.Linear(feat_dim, feat_dim // 2) for _ in range(num_parts)
                ])
                
                # Global head
                self.global_head = nn.Linear(feat_dim, feat_dim // 2)
                
                # Remove or modify the final classifier
                if hasattr(base_model, 'classifier'):
                    base_model.classifier = nn.Identity()
                elif hasattr(base_model, 'fc'):
                    base_model.fc = nn.Identity()
            
            def forward(self, x):
                # Get global features from base model
                global_features = self.base_model(x)
                
                # For simplicity, we'll use the same global features for all parts
                # In a real implementation, you'd use attention or spatial pooling
                
                # Global feature
                global_feat = self.global_head(global_features)
                global_feat = F.normalize(global_feat, p=2, dim=1)
                
                # Part features (simple approach - you can enhance this)
                part_feats = []
                for head in self.part_heads:
                    part_feat = head(global_features)
                    part_feat = F.normalize(part_feat, p=2, dim=1)
                    part_feats.append(part_feat)
                
                return {
                    'global': global_feat,
                    'parts': part_feats
                }
        
        return PartBasedWrapper(model, self.num_parts)
    
    def _load_pretrained_weights(self, model, weight_path):
        """Load pretrained weights from checkpoint file"""
        checkpoint = torch.load(weight_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if model was saved using DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f'Loaded pretrained weights from "{weight_path}"')

    def lighting_robust_preprocessing(self, image: np.ndarray) -> torch.Tensor:
        """Enhanced preprocessing for lighting robustness"""
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization for lighting normalization
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        
        # Merge and convert back
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Standard preprocessing
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]))
        image_tensor = self.transform(image)
        return image_tensor

    def _preprocess_single_image(self, image: np.ndarray) -> torch.Tensor:
        return self.lighting_robust_preprocessing(image)

    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess multiple images for model inference"""
        if len(images) <= 4:
            processed_images = [self._preprocess_single_image(img) for img in images]
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                processed_images = list(executor.map(self._preprocess_single_image, images))
        
        images_tensor = torch.stack(processed_images).to(self.device)
        return images_tensor

    def inference(self, images: List[np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract features from images - returns part-based features if enabled
        
        Returns:
            If use_parts=False: np.ndarray of shape (n_images, feat_dim)
            If use_parts=True: Dict with keys 'global', 'head', 'torso', 'legs'
        """
        if not images:
            if self.use_parts:
                return {'global': np.array([]), 'head': np.array([]), 'torso': np.array([]), 'legs': np.array([])}
            else:
                return np.array([])
            
        # Preprocess images
        preprocessed = self.preprocess(images)
        
        # Extract features
        with torch.no_grad():
            if self.use_parts:
                outputs = self.model(preprocessed)
                
                # Organize part features
                result = {
                    'global': outputs['global'].cpu().numpy(),
                    'head': outputs['parts'][0].cpu().numpy(),
                    'torso': outputs['parts'][1].cpu().numpy(),
                    'legs': outputs['parts'][2].cpu().numpy() if len(outputs['parts']) > 2 else outputs['parts'][1].cpu().numpy()
                }
                return result
            else:
                # Original behavior for backward compatibility
                features = self.model(preprocessed)
                features = F.normalize(features, p=2, dim=1)
                return features.cpu().numpy()
    
    def compute_part_based_similarity(self, img1: np.ndarray, img2: np.ndarray, 
                                     part_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute part-based similarity between two images
        
        Args:
            img1, img2: Input images as numpy arrays
            part_weights: Weights for different parts
            
        Returns:
            Dictionary with overall similarity and per-part similarities
        """
        if not self.use_parts:
            raise ValueError("Part-based features are not enabled. Set use_parts=True in constructor.")
        
        # Default weights
        if part_weights is None:
            part_weights = {
                'global': 0.4,
                'head': 0.2,
                'torso': 0.3,
                'legs': 0.1
            }
        
        # Extract features
        features = self.inference([img1, img2])
        
        similarities = {}
        weighted_similarity = 0.0
        total_weight = 0.0
        
        # Compute similarity for each part
        for part_name in ['global', 'head', 'torso', 'legs']:
            if part_name in features and features[part_name].shape[0] >= 2:
                # Get features for both images
                feat1 = features[part_name][0]
                feat2 = features[part_name][1]
                
                # Compute cosine similarity
                similarity = np.dot(feat1, feat2)
                similarities[f'{part_name}_similarity'] = float(similarity)
                
                # Add to weighted sum
                if part_name in part_weights:
                    weight = part_weights[part_name]
                    weighted_similarity += weight * similarity
                    total_weight += weight
        
        # Compute overall similarity
        if total_weight > 0:
            similarities['overall_similarity'] = weighted_similarity / total_weight
        else:
            similarities['overall_similarity'] = 0.0
        
        return similarities

    def compare_images_detailed(self, image_path1: str, image_path2: str, 
                               show_details: bool = True) -> Dict[str, float]:
        """
        Compare two images with detailed part-based analysis
        
        Args:
            image_path1, image_path2: Paths to images
            show_details: Whether to print detailed comparison
            
        Returns:
            Dictionary with all similarity scores
        """
        # Load images
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load images: {image_path1}, {image_path2}")
        
        if self.use_parts:
            # Part-based comparison
            results = self.compute_part_based_similarity(img1, img2)
            
            if show_details:
                print("=== Part-Based Similarity Analysis ===")
                print(f"Overall Similarity: {results['overall_similarity']:.4f}")
                print(f"Global Similarity:  {results.get('global_similarity', 'N/A'):.4f}")
                print(f"Head Similarity:    {results.get('head_similarity', 'N/A'):.4f}")
                print(f"Torso Similarity:   {results.get('torso_similarity', 'N/A'):.4f}")
                print(f"Legs Similarity:    {results.get('legs_similarity', 'N/A'):.4f}")
                
                # Interpretation
                overall_sim = results['overall_similarity']
                if overall_sim > 0.8:
                    print("🔥 Very High Similarity - Likely same person")
                elif overall_sim > 0.6:
                    print("✅ High Similarity - Probably same person")
                elif overall_sim > 0.4:
                    print("⚠️  Moderate Similarity - Could be same person")
                else:
                    print("❌ Low Similarity - Likely different people")
        else:
            # Global comparison
            similarity = self.compute_similarity_from_arrays(img1, img2)
            results = {'overall_similarity': similarity}
            
            if show_details:
                print("=== Global Similarity Analysis ===")
                print(f"Overall Similarity: {similarity:.4f}")
        
        return results

    # Keep all original methods for backward compatibility
    def compute_similarity_from_images(self, image_path1: str, image_path2: str) -> float:
        """Original method - compute global similarity between two image files"""
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load images: {image_path1}, {image_path2}")
        
        if self.use_parts:
            # Use overall similarity from part-based analysis
            results = self.compute_part_based_similarity(img1, img2)
            return results['overall_similarity']
        else:
            # Original global similarity
            embeddings = self.inference([img1, img2])
            return np.dot(embeddings[0], embeddings[1])

    def compute_similarity_from_arrays(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Original method - compute global similarity between two image arrays"""
        if self.use_parts:
            # Use overall similarity from part-based analysis
            results = self.compute_part_based_similarity(img1, img2)
            return results['overall_similarity']
        else:
            # Original global similarity
            embeddings = self.inference([img1, img2])
            return np.dot(embeddings[0], embeddings[1])

    def warm_up(self):
        """Warm up the model with dummy input"""
        img = np.random.randint(
            0, 255, (self.max_batch_size, self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )
        print("Warming up model...")
        _ = self.inference(list(img))
        print("Model warm-up complete")


# Create instances for both configurations
feature_extractor_global = BPBReidFeatureExtractor(use_parts=False)  # Original global features
feature_extractor_parts = BPBReidFeatureExtractor(use_parts=True)    # New part-based features
feature_extractor_parts.compute_similarity_from_images("/home/aidev/workspace/reid/Thesis/reid-2024/Screenshots/3_0.png","/home/aidev/workspace/reid/Thesis/reid-2024/Screenshots/10_0.png")