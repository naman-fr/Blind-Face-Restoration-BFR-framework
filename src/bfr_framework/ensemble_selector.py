import cv2
import numpy as np
import torch
import random
from typing import List, Tuple, Optional, Dict
from loguru import logger

class EnsembleSelector:
    """
    Handles multi-seed diffusion ensemble strategies.
    """
    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def compute_sharpness(image: np.ndarray) -> float:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def select_best(self, images: List[np.ndarray]) -> Tuple[np.ndarray, int, List[float]]:
        """
        Selects the sharpest image from a list.
        """
        scores = [self.compute_sharpness(img) for img in images]
        best_idx = int(np.argmax(scores))
        logger.info(f"Ensemble selection: Best index {best_idx} with sharpness {scores[best_idx]:.2f}")
        return images[best_idx], best_idx, scores

    def weighted_average(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Averages images weighted by their sharpness.
        """
        scores = [self.compute_sharpness(img) for img in images]
        total = sum(scores)
        if total == 0:
            weights = [1.0 / len(images)] * len(images)
        else:
            weights = [s / total for s in scores]
        
        avg = np.zeros_like(images[0], dtype=np.float64)
        for img, w in zip(images, weights):
            avg += img.astype(np.float64) * w
        
        return np.clip(avg, 0, 255).astype(np.uint8)
