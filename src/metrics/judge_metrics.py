# ============================================================================
# File: src/metrics/judge_metrics.py
# ============================================================================
"""Judge reliability metrics: ESA, SC, JRS."""

from typing import List, Set
import numpy as np
from ..models.types import ErrorSet, JudgeMetrics
from .error_tracker import ErrorTracker


class JudgeReliabilityCalculator:
    """Computes multi-judge agreement and reliability metrics."""
    
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize with weights for JRS computation.
        
        Args:
            alpha: Weight for ESA (agreement)
            beta: Weight for z-score penalty
        """
        self.alpha = alpha
        self.beta = beta
    
    def compute_esa(self, error_set_1: ErrorSet, error_set_2: ErrorSet) -> float:
        """
        Error Set Agreement between two judges.
        
        ESA = |E_i ∩ E_j| / |E_i ∪ E_j|
        """
        tracker = ErrorTracker()
        e1 = tracker.get_all_errors(error_set_1)
        e2 = tracker.get_all_errors(error_set_2)
        
        if len(e1 | e2) == 0:
            return 1.0  # Both agree there are no errors
        
        return len(e1 & e2) / len(e1 | e2)
    
    def compute_pairwise_esa(self, error_sets: List[ErrorSet]) -> List[float]:
        """Compute all pairwise ESA scores."""
        n = len(error_sets)
        esa_scores = []
        
        for i in range(n):
            for j in range(i + 1, n):
                esa = self.compute_esa(error_sets[i], error_sets[j])
                esa_scores.append(esa)
        
        return esa_scores
    
    def compute_mean_esa_per_judge(self, error_sets: List[ErrorSet]) -> List[float]:
        """Compute mean ESA for each judge with all others."""
        n = len(error_sets)
        mean_esas = []
        
        for i in range(n):
            esas = []
            for j in range(n):
                if i != j:
                    esa = self.compute_esa(error_sets[i], error_sets[j])
                    esas.append(esa)
            mean_esas.append(np.mean(esas) if esas else 0.0)
        
        return mean_esas
    
    def compute_sc(self, scores: List[float]) -> float:
        """
        Score Consistency: variance of scores across judges.
        Lower is better.
        
        SC = Var(S_i)
        """
        if len(scores) <= 1:
            return 0.0
        return float(np.var(scores))
    
    def compute_z_scores(self, scores: List[float]) -> List[float]:
        """
        Compute standardized z-scores for outlier detection.
        
        z_i = (S_i - μ) / σ
        """
        if len(scores) <= 1:
            return [0.0] * len(scores)
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return [0.0] * len(scores)
        
        return [(s - mean) / std for s in scores]
    
    def compute_jrs(
        self,
        error_sets: List[ErrorSet],
        scores: List[float]
    ) -> List[JudgeMetrics]:
        """
        Compute Judge Reliability Score for each judge.
        
        JRS_i = alpha * ESA_i - beta * |z_i|
        """
        mean_esas = self.compute_mean_esa_per_judge(error_sets)
        z_scores = self.compute_z_scores(scores)
        sc = self.compute_sc(scores)
        
        judge_metrics = []
        for i in range(len(error_sets)):
            jrs = self.alpha * mean_esas[i] - self.beta * abs(z_scores[i])
            
            judge_metrics.append({
                "esa": mean_esas[i],
                "sc": sc,  # Same for all judges
                "z_score": z_scores[i],
                "jrs": jrs
            })
        
        return judge_metrics
    
    def detect_outliers(self, z_scores: List[float], threshold: float = 2.0) -> List[int]:
        """
        Detect outlier judges based on z-score threshold.
        
        Returns indices of outlier judges.
        """
        return [i for i, z in enumerate(z_scores) if abs(z) > threshold]
