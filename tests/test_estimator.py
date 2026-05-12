import pytest
import numpy as np
from bfr_framework.degradation_estimator import DegradationEstimator

def test_degradation_estimator_output_range():
    estimator = DegradationEstimator()
    # Create a random noise image
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    severity, info = estimator.estimate_severity(img)
    
    assert 0.0 <= severity <= 1.0
    assert "laplacian_var" in info
    assert "blur_score" in info
    assert "noise_score" in info

def test_degradation_estimator_blur_detection():
    estimator = DegradationEstimator()
    # Smooth image (low variance)
    smooth_img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Sharp image (high variance)
    sharp_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    severity_smooth, info_smooth = estimator.estimate_severity(smooth_img)
    severity_sharp, info_sharp = estimator.estimate_severity(sharp_img)
    
    # Smooth image should have higher blur score (and thus higher severity if blur weighted)
    assert info_smooth["blur_score"] > info_sharp["blur_score"]

def test_n_selection():
    estimator = DegradationEstimator()
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    n_respaced, severity, info = estimator.select_n_adaptive(img, n_min=250, n_max=500)
    
    assert 1 <= n_respaced <= 250
    assert info["n_original"] >= 250
    assert info["n_original"] <= 500
