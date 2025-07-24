"""
Core functionality for vibrational resonance analysis and failure prediction.

This module contains the main classes and data structures for analyzing
vibrational data and predicting catastrophic failure in mechanical systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
import warnings


@dataclass
class VibrationData:
    """
    Container for vibration measurement data.
    
    Attributes:
        time: Time array in seconds
        amplitude: Vibration amplitude array
        frequency: Sampling frequency in Hz
        units: Units of amplitude (e.g., 'm/s²', 'mm/s', 'g')
        metadata: Additional metadata about the measurement
    """
    time: np.ndarray
    amplitude: np.ndarray
    frequency: float
    units: str = "m/s²"
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate input data."""
        if len(self.time) != len(self.amplitude):
            raise ValueError("Time and amplitude arrays must have the same length")
        if self.frequency <= 0:
            raise ValueError("Sampling frequency must be positive")
        if len(self.time) < 2:
            raise ValueError("At least 2 data points are required")
    
    @property
    def duration(self) -> float:
        """Duration of the measurement in seconds."""
        return self.time[-1] - self.time[0]
    
    @property
    def rms_amplitude(self) -> float:
        """Root mean square amplitude."""
        return np.sqrt(np.mean(self.amplitude**2))
    
    @property
    def peak_amplitude(self) -> float:
        """Peak amplitude."""
        return np.max(np.abs(self.amplitude))


@dataclass
class ResonanceResult:
    """
    Results from resonance analysis.
    
    Attributes:
        resonant_frequencies: List of detected resonant frequencies in Hz
        resonance_peaks: Peak amplitudes at resonant frequencies
        quality_factors: Quality factors (Q) for each resonance
        resonance_risk: Risk level for each resonance (low, medium, high, critical)
        frequency_spectrum: Frequency spectrum data
        power_spectrum: Power spectral density
    """
    resonant_frequencies: List[float]
    resonance_peaks: List[float]
    quality_factors: List[float]
    resonance_risk: List[str]
    frequency_spectrum: np.ndarray
    power_spectrum: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate results."""
        if not (len(self.resonant_frequencies) == len(self.resonance_peaks) == 
                len(self.quality_factors) == len(self.resonance_risk)):
            raise ValueError("All result arrays must have the same length")


@dataclass
class FailurePrediction:
    """
    Prediction results for catastrophic failure.
    
    Attributes:
        failure_probability: Probability of failure (0-1)
        time_to_failure: Estimated time to failure in hours (if applicable)
        failure_mode: Predicted failure mode
        confidence: Confidence level in the prediction (0-1)
        risk_factors: List of contributing risk factors
        recommendations: List of recommendations to mitigate risk
    """
    failure_probability: float
    time_to_failure: Optional[float]
    failure_mode: str
    confidence: float
    risk_factors: List[str]
    recommendations: List[str]
    
    def __post_init__(self) -> None:
        """Validate prediction."""
        if not 0 <= self.failure_probability <= 1:
            raise ValueError("Failure probability must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.time_to_failure is not None and self.time_to_failure < 0:
            raise ValueError("Time to failure must be non-negative")


class ResonanceAnalyzer:
    """
    Analyzes vibrational data to detect resonance conditions.
    
    This class provides methods to identify resonant frequencies, calculate
    quality factors, and assess resonance risk levels.
    """
    
    def __init__(self, 
                 min_peak_height: float = 0.1,
                 min_peak_distance: float = 1.0,
                 prominence_threshold: float = 0.05):
        """
        Initialize the resonance analyzer.
        
        Args:
            min_peak_height: Minimum peak height relative to max amplitude
            min_peak_distance: Minimum distance between peaks in Hz
            prominence_threshold: Minimum peak prominence for detection
        """
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        self.prominence_threshold = prominence_threshold
    
    def analyze(self, data: VibrationData) -> ResonanceResult:
        """
        Analyze vibration data for resonance conditions.
        
        Args:
            data: Vibration data to analyze
            
        Returns:
            ResonanceResult containing analysis results
        """
        # Compute FFT
        n_samples = len(data.amplitude)
        fft_result = fft(data.amplitude)
        frequencies = fftfreq(n_samples, 1/data.frequency)
        
        # Compute power spectral density
        power_spectrum = np.abs(fft_result)**2 / n_samples
        
        # Find peaks in positive frequency range
        positive_mask = frequencies > 0
        positive_freqs = frequencies[positive_mask]
        positive_power = power_spectrum[positive_mask]
        
        # Find peaks using scipy
        peaks, properties = signal.find_peaks(
            positive_power,
            height=self.min_peak_height * np.max(positive_power),
            distance=int(self.min_peak_distance / (positive_freqs[1] - positive_freqs[0])),
            prominence=self.prominence_threshold * np.max(positive_power)
        )
        
        resonant_frequencies = positive_freqs[peaks].tolist()
        resonance_peaks = positive_power[peaks].tolist()
        
        # Calculate quality factors
        quality_factors = []
        for i, peak_idx in enumerate(peaks):
            freq = resonant_frequencies[i]
            peak_power = resonance_peaks[i]
            
            # Find -3dB points
            half_power = peak_power / 2
            left_idx = peak_idx
            right_idx = peak_idx
            
            # Search left
            while left_idx > 0 and positive_power[left_idx] > half_power:
                left_idx -= 1
            
            # Search right
            while right_idx < len(positive_power) - 1 and positive_power[right_idx] > half_power:
                right_idx += 1
            
            if left_idx < right_idx:
                bandwidth = positive_freqs[right_idx] - positive_freqs[left_idx]
                q_factor = freq / bandwidth if bandwidth > 0 else 1000
            else:
                q_factor = 1000  # Default for narrow peaks
            
            quality_factors.append(q_factor)
        
        # Assess risk levels
        resonance_risk = []
        for i, (freq, peak, q_factor) in enumerate(zip(resonant_frequencies, resonance_peaks, quality_factors)):
            risk = self._assess_risk(freq, peak, q_factor, data)
            resonance_risk.append(risk)
        
        return ResonanceResult(
            resonant_frequencies=resonant_frequencies,
            resonance_peaks=resonance_peaks,
            quality_factors=quality_factors,
            resonance_risk=resonance_risk,
            frequency_spectrum=frequencies,
            power_spectrum=power_spectrum
        )
    
    def _assess_risk(self, freq: float, peak: float, q_factor: float, data: VibrationData) -> str:
        """
        Assess risk level for a resonance.
        
        Args:
            freq: Resonant frequency
            peak: Peak amplitude
            q_factor: Quality factor
            data: Original vibration data
            
        Returns:
            Risk level string
        """
        # Normalize peak amplitude
        normalized_peak = peak / np.max(data.amplitude**2)
        
        # Risk assessment based on multiple factors
        risk_score = 0
        
        # High amplitude risk
        if normalized_peak > 0.8:
            risk_score += 3
        elif normalized_peak > 0.5:
            risk_score += 2
        elif normalized_peak > 0.2:
            risk_score += 1
        
        # High Q-factor risk (sharp resonance)
        if q_factor > 50:
            risk_score += 2
        elif q_factor > 20:
            risk_score += 1
        
        # Frequency-dependent risk (common problematic frequencies)
        if 50 <= freq <= 60:  # Power line frequencies
            risk_score += 1
        elif 100 <= freq <= 120:  # Motor frequencies
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 5:
            return "critical"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 1:
            return "medium"
        else:
            return "low"


class FailurePredictor:
    """
    Predicts catastrophic failure based on vibration analysis.
    
    This class uses machine learning and statistical methods to predict
    the probability and timing of catastrophic failure.
    """
    
    def __init__(self, 
                 historical_data: Optional[List[VibrationData]] = None,
                 failure_threshold: float = 0.8):
        """
        Initialize the failure predictor.
        
        Args:
            historical_data: Historical vibration data for training
            failure_threshold: Threshold for failure probability
        """
        self.historical_data = historical_data or []
        self.failure_threshold = failure_threshold
        self._trained = False
    
    def train(self, failure_events: List[Dict[str, Any]]) -> None:
        """
        Train the predictor with historical failure events.
        
        Args:
            failure_events: List of failure event data
        """
        if not failure_events:
            warnings.warn("No failure events provided for training")
            return
        
        # Simple training - in a real implementation, this would use
        # more sophisticated ML models
        self._trained = True
    
    def predict(self, data: VibrationData, resonance_result: ResonanceResult) -> FailurePrediction:
        """
        Predict failure probability and timing.
        
        Args:
            data: Current vibration data
            resonance_result: Results from resonance analysis
            
        Returns:
            FailurePrediction with results
        """
        # Calculate base failure probability
        failure_prob = self._calculate_failure_probability(data, resonance_result)
        
        # Estimate time to failure
        time_to_failure = self._estimate_time_to_failure(data, resonance_result)
        
        # Determine failure mode
        failure_mode = self._determine_failure_mode(resonance_result)
        
        # Calculate confidence
        confidence = self._calculate_confidence(data, resonance_result)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(data, resonance_result)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data, resonance_result)
        
        return FailurePrediction(
            failure_probability=failure_prob,
            time_to_failure=time_to_failure,
            failure_mode=failure_mode,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
    
    def _calculate_failure_probability(self, data: VibrationData, 
                                     resonance_result: ResonanceResult) -> float:
        """Calculate failure probability based on multiple factors."""
        prob = 0.0
        
        # RMS amplitude factor
        rms_factor = min(data.rms_amplitude / (data.peak_amplitude * 0.7), 1.0)
        prob += 0.3 * rms_factor
        
        # Critical resonance factor
        critical_resonances = sum(1 for risk in resonance_result.resonance_risk 
                                if risk in ["high", "critical"])
        prob += 0.2 * min(critical_resonances / 3, 1.0)
        
        # Peak amplitude factor
        peak_factor = min(data.peak_amplitude / (data.rms_amplitude * 5), 1.0)
        prob += 0.3 * peak_factor
        
        # Quality factor factor
        if resonance_result.quality_factors:
            avg_q = np.mean(resonance_result.quality_factors)
            q_factor = min(avg_q / 50, 1.0)
            prob += 0.2 * q_factor
        
        return min(prob, 1.0)
    
    def _estimate_time_to_failure(self, data: VibrationData, 
                                resonance_result: ResonanceResult) -> Optional[float]:
        """Estimate time to failure in hours."""
        if not self._trained:
            return None
        
        # Simple estimation based on resonance severity
        critical_count = sum(1 for risk in resonance_result.resonance_risk 
                           if risk == "critical")
        high_count = sum(1 for risk in resonance_result.resonance_risk 
                        if risk == "high")
        
        if critical_count > 0:
            return 24.0  # 24 hours for critical resonance
        elif high_count > 1:
            return 168.0  # 1 week for multiple high resonances
        elif high_count > 0:
            return 720.0  # 1 month for single high resonance
        else:
            return None  # No immediate failure expected
    
    def _determine_failure_mode(self, resonance_result: ResonanceResult) -> str:
        """Determine the most likely failure mode."""
        if not resonance_result.resonant_frequencies:
            return "unknown"
        
        # Analyze frequency patterns
        low_freq_count = sum(1 for f in resonance_result.resonant_frequencies if f < 50)
        mid_freq_count = sum(1 for f in resonance_result.resonant_frequencies if 50 <= f <= 200)
        high_freq_count = sum(1 for f in resonance_result.resonant_frequencies if f > 200)
        
        if low_freq_count > mid_freq_count and low_freq_count > high_freq_count:
            return "structural_failure"
        elif mid_freq_count > high_freq_count:
            return "bearing_failure"
        else:
            return "component_fatigue"
    
    def _calculate_confidence(self, data: VibrationData, 
                            resonance_result: ResonanceResult) -> float:
        """Calculate confidence in the prediction."""
        confidence = 0.5  # Base confidence
        
        # More data points increase confidence
        if len(data.time) > 1000:
            confidence += 0.2
        elif len(data.time) > 100:
            confidence += 0.1
        
        # Clear resonance peaks increase confidence
        if len(resonance_result.resonant_frequencies) > 0:
            confidence += 0.2
        
        # High quality factors increase confidence
        if resonance_result.quality_factors:
            avg_q = np.mean(resonance_result.quality_factors)
            if avg_q > 20:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _identify_risk_factors(self, data: VibrationData, 
                             resonance_result: ResonanceResult) -> List[str]:
        """Identify contributing risk factors."""
        risk_factors = []
        
        if data.rms_amplitude > data.peak_amplitude * 0.5:
            risk_factors.append("high_rms_amplitude")
        
        if any(risk == "critical" for risk in resonance_result.resonance_risk):
            risk_factors.append("critical_resonance_detected")
        
        if len(resonance_result.resonant_frequencies) > 3:
            risk_factors.append("multiple_resonances")
        
        if resonance_result.quality_factors:
            if np.mean(resonance_result.quality_factors) > 30:
                risk_factors.append("high_quality_factors")
        
        return risk_factors
    
    def _generate_recommendations(self, data: VibrationData, 
                                resonance_result: ResonanceResult) -> List[str]:
        """Generate recommendations to mitigate risk."""
        recommendations = []
        
        if any(risk == "critical" for risk in resonance_result.resonance_risk):
            recommendations.append("Immediate shutdown recommended due to critical resonance")
            recommendations.append("Inspect for structural damage")
        
        if len(resonance_result.resonant_frequencies) > 2:
            recommendations.append("Consider vibration isolation or damping")
        
        if data.rms_amplitude > data.peak_amplitude * 0.4:
            recommendations.append("Reduce operating speed or load")
        
        if resonance_result.quality_factors:
            if np.mean(resonance_result.quality_factors) > 25:
                recommendations.append("Add damping material to reduce resonance sharpness")
        
        if not recommendations:
            recommendations.append("Continue monitoring - current levels acceptable")
        
        return recommendations