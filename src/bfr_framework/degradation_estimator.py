import cv2
import numpy as np
from typing import Dict, Tuple, Union, Optional
from loguru import logger

class DegradationEstimator:
    """
    Estimates image degradation severity using Laplacian variance and DoG.
    """
    def __init__(
        self,
        lap_low: float = 30.0,
        lap_high: float = 800.0,
        noise_low: float = 2.0,
        noise_high: float = 18.0,
        blur_weight: float = 0.65
    ):
        self.lap_low = lap_low
        self.lap_high = lap_high
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.blur_weight = blur_weight

    def estimate_severity(self, image: Union[np.ndarray, str]) -> Tuple[float, Dict]:
        """
        Calculates severity score [0, 1].
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image at {image}")
        else:
            img = image.copy()

        if img.dtype in (np.float32, np.float64):
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        # Grayscale canonical resize
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

        # 1) Blur detection
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 1.0 - np.clip((lap_var - self.lap_low) / (self.lap_high - self.lap_low), 0.0, 1.0)

        # 2) Noise detection (DoG)
        gray_f = gray.astype(np.float64)
        dog = cv2.GaussianBlur(gray_f, (3, 3), 0.5) - cv2.GaussianBlur(gray_f, (7, 7), 1.5)
        noise_energy = np.std(dog)
        noise_score = np.clip((noise_energy - self.noise_low) / (self.noise_high - self.noise_low), 0.0, 1.0)

        # 3) Combine
        severity = float(np.clip(self.blur_weight * blur_score + (1 - self.blur_weight) * noise_score, 0.0, 1.0))

        info = {
            "laplacian_var": float(lap_var),
            "blur_score": float(blur_score),
            "noise_energy": float(noise_energy),
            "noise_score": float(noise_score),
            "severity": severity
        }
        
        logger.debug(f"Degradation analysis: severity={severity:.3f}, blur={blur_score:.3f}, noise={noise_score:.3f}")
        return severity, info

    def select_n_adaptive(
        self,
        image: np.ndarray,
        n_min: int = 250,
        n_max: int = 500,
        original_steps: int = 1000,
        respaced_steps: int = 250
    ) -> Tuple[int, float, Dict]:
        """
        Maps severity to a timestep N in the respaced domain.
        """
        severity, info = self.estimate_severity(image)
        n_orig = int(round(n_min + severity * (n_max - n_min)))
        n_respaced = int(round(n_orig * respaced_steps / original_steps))
        n_respaced = max(1, min(n_respaced, respaced_steps - 1))

        info["n_original"] = n_orig
        info["n_respaced"] = n_respaced
        
        return n_respaced, severity, info
