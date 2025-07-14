import os
import sys

# # Add the local torchreid path to Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# sys.path.insert(0, project_root)

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import glob
from typing import List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import torchreid
from torchreid.utils import load_pretrained_weights, check_isfile
from default_config import get_default_config


class ReIDFeatureExtractor:
    def __init__(
        self,
        config_file_path: str,
        model_weights_path: str,
        num_classes: int = 702,
        max_batch_size: int = 32,
        device: Optional[str] = None,
        num_workers: int = 4,
    ):
        """
        Initialize ReID Feature Extractor
        
        Args:
            config_file_path: Path to config YAML file
            model_weights_path: Path to pretrained model weights
            num_classes: Number of classes in training dataset
            max_batch_size: Maximum batch size for inference
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
            num_workers: Number of workers for parallel processing
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        
        # Load model and config
        self.model, self.cfg = self._load_model_for_deployment(
            config_file_path, model_weights_path, num_classes
        )
        
        # Get preprocessing parameters from config
        self.normalize_mean = self.cfg.data.norm_mean
        self.normalize_std = self.cfg.data.norm_std
        self.image_height = self.cfg.data.height
        self.image_width = self.cfg.data.width
        
        # Precompute the transform for efficiency
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # Warm up the model
        self.warm_up()
    
    def _load_model_for_deployment(self, config_file, model_weights_path, num_classes):
        """Load model and configuration for deployment"""
        cfg = get_default_config()
        cfg.use_gpu = torch.cuda.is_available()
        
        cfg.merge_from_file(config_file)
        
        cfg.data.save_dir = 'log/deploy_hardcoded_run_temp' 
        cfg.test.evaluate = False 
        cfg.model.pretrained = False 
        cfg.freeze()

        model = torchreid.models.build_model(
            name=cfg.model.name,
            num_classes=num_classes, 
            loss=cfg.loss.name,      
            pretrained=False,        
            use_gpu=cfg.use_gpu
        )

        load_pretrained_weights(model, model_weights_path)

        if cfg.use_gpu:
            model = model.cuda()
        
        model.eval() 
        return model, cfg

    def lighting_robust_preprocessing(self, image: np.ndarray) -> torch.Tensor:
        """Enhanced preprocessing specifically for lighting robustness"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization for lighting normalization
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        
        # Apply CLAHE to L channel only (preserves color information)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        
        # Merge and convert back
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (self.image_width, self.image_height))
        
        # Apply transforms
        image_tensor = self.transform(image)
        return image_tensor

    def _preprocess_single_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image"""
        return self.lighting_robust_preprocessing(image)

    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess multiple images for model inference using parallel processing
        """
        # For small batches, process directly without threading
        if len(images) <= 4:
            processed_images = [self._preprocess_single_image(img) for img in images]
        else:
            # Use ThreadPoolExecutor for parallel preprocessing
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                processed_images = list(executor.map(self._preprocess_single_image, images))
        
        # Stack all images into a batch
        images_tensor = torch.stack(processed_images).to(self.device)
        return images_tensor

    def _create_batches(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Split images into batches of max_batch_size"""
        return [images[i:i + self.max_batch_size] for i in range(0, len(images), self.max_batch_size)]

    @torch.no_grad()
    def inference(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of images, efficiently processing in batches
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            np.ndarray: Feature matrix of shape (num_images, feature_dim)
        """
        # Handle empty input
        
            
        # If only one image or less than batch size, process directly
        if len(images) <= self.max_batch_size:
            preprocessed = self.preprocess(images)
            features = self.model(preprocessed)
            features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy()
        
        # For multiple batches, process each batch and concatenate results
        batches = self._create_batches(images)
        all_features = []
        
        for batch in batches:
            # Preprocess batch
            preprocessed = self.preprocess(batch)
            
            # Extract features
            features = self.model(preprocessed)
            
            features = F.normalize(features, p=2, dim=1)
            all_features.append(features.cpu().numpy())
        
        # Concatenate all batch results
        return np.concatenate(all_features, axis=0)

    def inference_async(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of images using async processing
        """
        # Handle empty input
        if not images:
            return np.array([])
            
        # If small batch, just use the synchronous method
        if len(images) <= self.max_batch_size:
            return self.inference(images)
            
        # For large inputs, process batches in parallel
        batches = self._create_batches(images)
        all_features = [None] * len(batches)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batch processing tasks
            future_to_idx = {}
            for idx, batch in enumerate(batches):
                future = executor.submit(self._process_batch, batch)
                future_to_idx[future] = idx
            
            # Collect results in order
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                all_features[idx] = result
        
        # Concatenate all batch results
        return np.concatenate(all_features, axis=0)

    def _process_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        """Process a single batch of images (for async processing)"""
        preprocessed = self.preprocess(batch)
        
        with torch.no_grad():
            features = self.model(preprocessed)
            features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy()

    def warm_up(self):
        """Warm up the model with a dummy input"""
        dummy_img = np.random.randint(
            0, 255, 
            (self.max_batch_size, self.image_height, self.image_width, 3), 
            dtype=np.uint8
        )
        
        print("Warming up model...")
        _ = self.inference(list(dummy_img))
        print("Model warm-up complete")

    def compare_features(self, query_feature: np.ndarray, gallery_features: np.ndarray, metric: str = 'cosine distance'):
        """
        Compare query feature with gallery features
        
        Args:
            query_feature: Query feature vector
            gallery_features: Gallery feature matrix
            metric: Distance metric ('euclidean', 'cosine', 'cosine distance')
            
        Returns:
            tuple: (distances, sorted_indices)
        """
        query_feature = torch.from_numpy(query_feature).squeeze()
        gallery_features = torch.from_numpy(gallery_features).squeeze()

        if gallery_features.ndim == 1 and gallery_features.numel() > 0:
            gallery_features = gallery_features.unsqueeze(0)
        elif gallery_features.numel() == 0:
            return np.array([]), np.array([])

        if metric == 'euclidean':
            distances = torch.cdist(query_feature.unsqueeze(0), gallery_features, p=2.0).squeeze(0)
            sorted_indices = torch.argsort(distances)
        elif metric == 'cosine':
            query_norm = F.normalize(query_feature.unsqueeze(0), p=2, dim=1)
            gallery_norm = F.normalize(gallery_features, p=2, dim=1)
            similarities = torch.mm(query_norm, gallery_norm.transpose(0, 1)).squeeze(0)
            distances = 1 - similarities 
            sorted_indices = torch.argsort(similarities, descending=True)
        elif metric == 'cosine distance':
            query_norm = F.normalize(query_feature.unsqueeze(0), p=2, dim=1)
            gallery_norm = F.normalize(gallery_features, p=2, dim=1)
            similarities = torch.mm(query_norm, gallery_norm.transpose(0, 1)).squeeze(0)
            distances = 1 - similarities
            sorted_indices = torch.argsort(distances, descending=False)
        else:
            raise ValueError(f"Metric not supported: {metric}. Choose 'euclidean', 'cosine', or 'cosine distance'.")
            
        return distances.numpy(), sorted_indices.numpy()

    def compute_similarity_from_images(self, image_path1: str, image_path2: str) -> float:
        """
        Compute similarity between two image files
        
        Args:
            image_path1: Path to the first image
            image_path2: Path to the second image
            
        Returns:
            float: Cosine similarity score between the two images (0-1 range)
        """
        # Load images using OpenCV
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        
        if img1 is None or img2 is None:
            raise ValueError(f"Failed to load one or both images: {image_path1}, {image_path2}")
        
        # Get embeddings by processing both images in a single batch
        embeddings = self.inference([img1, img2])
        
        # Compute cosine similarity between the first and second embedding
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return similarity

    def compute_similarity_from_arrays(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute similarity between two image arrays
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            float: Cosine similarity score between the two images (0-1 range)
        """
        # Get embeddings by processing both images in a single batch
        embeddings = self.inference([img1, img2])
        
        # Compute cosine similarity between the first and second embedding
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return similarity

    def extract_feature_from_path(self, image_path: str) -> np.ndarray:
        """
        Extract features from a single image path
        
        Args:
            image_path: Path to the image
            
        Returns:
            np.ndarray: Feature vector for the image
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Get embedding
        embedding = self.inference([img])
        
        return embedding.squeeze()  # Remove batch dimension

    def compute_cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, float]:
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            tuple: (similarity, distance) where similarity is cosine similarity
                  and distance is 1 - similarity
        """
        # Features should already be normalized from.inference(),
        # but normalize again to be sure
        if isinstance(feat1, np.ndarray):
            feat1 = torch.from_numpy(feat1)
        if isinstance(feat2, np.ndarray):
            feat2 = torch.from_numpy(feat2)
            
        feat1 = F.normalize(feat1, p=2, dim=0)
        feat2 = F.normalize(feat2, p=2, dim=0)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarity = torch.dot(feat1, feat2).item()
        
        # Convert to distance (1 - similarity)
        distance = 1 - similarity
        
        return similarity, distance

    def compute_similarity_batch(self, query_images: List[np.ndarray], gallery_images: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity matrix between query images and gallery images
        
        Args:
            query_images: List of query images as numpy arrays
            gallery_images: List of gallery images as numpy arrays
            
        Returns:
            np.ndarray: Similarity matrix of shape (len(query_images), len(gallery_images))
        """
        # Extract features for all images in batch
        query_features = self.inference(query_images)
        gallery_features = self.inference(gallery_images)
        
        # Compute similarity matrix efficiently using matrix multiplication
        similarity_matrix = np.dot(query_features, gallery_features.T)
        
        return similarity_matrix

    @staticmethod
    def get_image_files_from_directory(directory_path: str) -> List[str]:
        """Get all image files from a directory"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
        
        image_files = sorted(list(set([os.path.normpath(f) for f in image_files])))
        return image_files


def main(config_file_path: str, model_weights_path: str, query_image_path: str, 
         gallery_dir_path: str, num_classes: int = 702, metric: str = 'cosine distance'):
    """
    Main function to perform person re-identification
    
    Args:
        config_file_path: Path to config YAML file
        model_weights_path: Path to pretrained model weights
        query_image_path: Path to query image
        gallery_dir_path: Path to gallery directory
        num_classes: Number of classes in training dataset
        metric: Distance metric to use
    """
    # Initialize feature extractor
    extractor = ReIDFeatureExtractor(
        config_file_path=config_file_path,
        model_weights_path=model_weights_path,
        num_classes=num_classes
    )

    # Extract features for query image
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        raise ValueError(f"Failed to load query image: {query_image_path}")
    
    query_feature = extractor.inference([query_img])

    # Get gallery images and extract features
    gallery_image_paths = ReIDFeatureExtractor.get_image_files_from_directory(gallery_dir_path)
    
    if not gallery_image_paths:
        print("No gallery images found!")
        return

    # Load gallery images
    gallery_images = []
    valid_gallery_paths = []
    
    for gallery_path in gallery_image_paths:
        img = cv2.imread(gallery_path)
        if img is not None:
            gallery_images.append(img)
            valid_gallery_paths.append(gallery_path)

    if not gallery_images:
        print("No valid gallery images found!")
        return

    # Extract gallery features
    gallery_features = extractor.inference(gallery_images)

    # Compare and rank
    distances, sorted_indices = extractor.compare_features(query_feature[0], gallery_features, metric=metric)
    
    # Print results
    print(f"\nQuery: {os.path.basename(query_image_path)}")
    print(f"Gallery directory: {gallery_dir_path}")
    print(f"Metric: {metric}")
    print("-" * 60)
    
    for i in range(len(sorted_indices)):
        current_gallery_index = sorted_indices[i]
        gallery_path = valid_gallery_paths[current_gallery_index]
        dist = distances[current_gallery_index]
        print(f"Rank {i+1}: {os.path.basename(gallery_path)} - Distance: {dist:.4f}")
import matplotlib.pyplot as plt
import seaborn as sns
import os


def visualize_distance_matrix_from_folder(
    extractor: ReIDFeatureExtractor,
    folder_path: str,
    metric: str = 'cosine distance',
    output_path: str = 'distance_matrix.png',
    csv_output_path: str = 'distance_matrix.csv'
):
    """
    Compute and save the pairwise distance matrix heatmap of all images in the folder.
    Also exports the distance matrix to a CSV file.
    
    Args:
        extractor: Initialized ReIDFeatureExtractor
        folder_path: Path to image folder
        metric: Distance metric (default 'cosine distance')
        output_path: Path to save the .png file
        csv_output_path: Path to save the .csv file
    """
    # Load images
    image_paths = ReIDFeatureExtractor.get_image_files_from_directory(folder_path)
    images = []
    valid_paths = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            valid_paths.append(path)

    if len(images) < 2:
        print("Need at least 2 valid images.")
        return

    # Extract features
    features = extractor.inference(images)
    

    # Compute pairwise distances
    n = len(features)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            _, dist = extractor.compute_cosine_similarity(features[i], features[j])
            distance_matrix[i, j] = dist

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(distance_matrix,
                     xticklabels=[os.path.basename(p) for p in valid_paths],
                     yticklabels=[os.path.basename(p) for p in valid_paths],
                     annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Pairwise Cosine Distance Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path)
    plt.close()
    print(f"Distance matrix visualization saved to: {output_path}")

    # Export to CSV
    import pandas as pd
    df = pd.DataFrame(
        distance_matrix,
        index=[os.path.basename(p) for p in valid_paths],
        columns=[os.path.basename(p) for p in valid_paths]
    )
    df.to_csv(csv_output_path)
    print(f"Distance matrix data saved to: {csv_output_path}")

feature_extractor = ReIDFeatureExtractor(
        config_file_path='/home/aidev/workspace/reid/Thesis/Training/FPB/configs/cuhk_detected.yaml',
        model_weights_path='/home/aidev/workspace/reid/Thesis/Training/FPB/log/cuhk_detected/model.pth.tar-120',
        num_classes=702
    )
visualize_distance_matrix_from_folder(
    extractor=feature_extractor,
    folder_path='/home/aidev/workspace/reid/Thesis/reid-2024/Screenshot/vnpt',
    metric='cosine distance',
    output_path='check_distances.png',
    csv_output_path='check_distances.csv'
)
# if __name__ == '__main__':
#     main(
#         config_file_path='/home/aidev/workspace/reid/Thesis/Training/FPB/configs/cuhk_detected.yaml',
#         model_weights_path='/home/aidev/workspace/reid/Thesis/Training/FPB/log/cuhk_detected/model.pth.tar-120', 
#         query_image_path='Screenshots/check/4.png',
#         gallery_dir_path='Screenshots/check',
#         num_classes=702, 
#         metric='cosine distance'
#     )
    # extractor = ReIDFeatureExtractor(
    #     config_file_path='/home/aidev/workspace/reid/Thesis/Training/FPB/configs/cuhk_detected.yaml',
    #     model_weights_path='/home/aidev/workspace/reid/Thesis/Training/FPB/log/cuhk_detected/model.pth.tar-120',
    #     num_classes=702
    # )

    # # Extract features from single image
    # img = cv2.imread('/home/aidev/workspace/reid/Thesis/Screenshots/10_0.png')
    # features = extractor.inference([img])

    # # Compare two images
    # similarity = extractor.compute_similarity_from_images('/home/aidev/workspace/reid/Thesis/Screenshots/10_0.png', '/home/aidev/workspace/reid/Thesis/Screenshots/3_0.png')
    # print(similarity)
    