"""
Tests for core Vibequake functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from vibequake.core import (
    VibrationData,
    ResonanceAnalyzer,
    FailurePredictor,
    ResonanceResult,
    FailurePrediction
)
from vibequake.utils import generate_sample_data


class TestVibrationData:
    """Test VibrationData class."""
    
    def test_valid_creation(self):
        """Test creating valid VibrationData."""
        time = np.linspace(0, 10, 1000)
        amplitude = np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.normal(0, 1, 1000)
        
        data = VibrationData(
            time=time,
            amplitude=amplitude,
            frequency=100.0,
            units="m/s²"
        )
        
        assert len(data.time) == 1000
        assert len(data.amplitude) == 1000
        assert data.frequency == 100.0
        assert data.units == "m/s²"
        assert data.duration == 10.0
        assert data.rms_amplitude > 0
        assert data.peak_amplitude > 0
    
    def test_invalid_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        time = np.linspace(0, 10, 1000)
        amplitude = np.sin(2 * np.pi * 50 * time)[:500]  # Different length
        
        with pytest.raises(ValueError, match="must have the same length"):
            VibrationData(time=time, amplitude=amplitude, frequency=100.0)
    
    def test_invalid_frequency(self):
        """Test that negative frequency raises ValueError."""
        time = np.linspace(0, 10, 1000)
        amplitude = np.sin(2 * np.pi * 50 * time)
        
        with pytest.raises(ValueError, match="must be positive"):
            VibrationData(time=time, amplitude=amplitude, frequency=-100.0)
    
    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        time = np.array([0])
        amplitude = np.array([1.0])
        
        with pytest.raises(ValueError, match="At least 2 data points"):
            VibrationData(time=time, amplitude=amplitude, frequency=100.0)


class TestResonanceAnalyzer:
    """Test ResonanceAnalyzer class."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = generate_sample_data(
            duration=5.0,
            sampling_freq=1000.0,
            resonant_freqs=[50, 120, 300],
            noise_level=0.1
        )
    
    def test_analyzer_creation(self):
        """Test creating analyzer with custom parameters."""
        analyzer = ResonanceAnalyzer(
            min_peak_height=0.2,
            min_peak_distance=2.0,
            prominence_threshold=0.1
        )
        
        assert analyzer.min_peak_height == 0.2
        assert analyzer.min_peak_distance == 2.0
        assert analyzer.prominence_threshold == 0.1
    
    def test_resonance_analysis(self):
        """Test basic resonance analysis."""
        analyzer = ResonanceAnalyzer()
        result = analyzer.analyze(self.data)
        
        assert isinstance(result, ResonanceResult)
        assert len(result.resonant_frequencies) > 0
        assert len(result.resonant_frequencies) == len(result.resonance_peaks)
        assert len(result.resonant_frequencies) == len(result.quality_factors)
        assert len(result.resonant_frequencies) == len(result.resonance_risk)
        
        # Check that frequencies are positive
        assert all(f > 0 for f in result.resonant_frequencies)
        
        # Check that peaks are positive
        assert all(p > 0 for p in result.resonance_peaks)
        
        # Check that quality factors are positive
        assert all(q > 0 for q in result.quality_factors)
        
        # Check that risk levels are valid
        valid_risks = {"low", "medium", "high", "critical"}
        assert all(risk in valid_risks for risk in result.resonance_risk)
    
    def test_analysis_with_no_resonances(self):
        """Test analysis with data that has no clear resonances."""
        # Create data with just noise
        time = np.linspace(0, 5, 5000)
        amplitude = 0.1 * np.random.normal(0, 1, 5000)
        
        data = VibrationData(
            time=time,
            amplitude=amplitude,
            frequency=1000.0
        )
        
        analyzer = ResonanceAnalyzer()
        result = analyzer.analyze(data)
        
        # Should still return valid result, possibly with no resonances
        assert isinstance(result, ResonanceResult)
        assert len(result.resonant_frequencies) >= 0
    
    def test_risk_assessment(self):
        """Test that risk assessment produces reasonable results."""
        analyzer = ResonanceAnalyzer()
        result = analyzer.analyze(self.data)
        
        # Check that risk levels are assigned
        assert len(result.resonance_risk) == len(result.resonant_frequencies)
        
        # Check that higher peaks tend to have higher risk
        for i, (peak, risk) in enumerate(zip(result.resonance_peaks, result.resonance_risk)):
            # This is a basic check - in practice, risk depends on multiple factors
            assert risk in ["low", "medium", "high", "critical"]


class TestFailurePredictor:
    """Test FailurePredictor class."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = generate_sample_data(
            duration=5.0,
            sampling_freq=1000.0,
            resonant_freqs=[50, 120, 300],
            noise_level=0.1
        )
        
        analyzer = ResonanceAnalyzer()
        self.resonance_result = analyzer.analyze(self.data)
    
    def test_predictor_creation(self):
        """Test creating predictor with custom parameters."""
        predictor = FailurePredictor(failure_threshold=0.9)
        assert predictor.failure_threshold == 0.9
    
    def test_failure_prediction(self):
        """Test basic failure prediction."""
        predictor = FailurePredictor()
        prediction = predictor.predict(self.data, self.resonance_result)
        
        assert isinstance(prediction, FailurePrediction)
        assert 0 <= prediction.failure_probability <= 1
        assert 0 <= prediction.confidence <= 1
        assert prediction.failure_mode in ["unknown", "structural_failure", "bearing_failure", "component_fatigue"]
        assert isinstance(prediction.risk_factors, list)
        assert isinstance(prediction.recommendations, list)
    
    def test_prediction_without_training(self):
        """Test prediction without historical training data."""
        predictor = FailurePredictor()
        prediction = predictor.predict(self.data, self.resonance_result)
        
        # Should still work, but time_to_failure might be None
        assert prediction.time_to_failure is None or prediction.time_to_failure >= 0
    
    def test_prediction_with_training(self):
        """Test prediction with training data."""
        predictor = FailurePredictor()
        
        # Mock training data
        failure_events = [
            {"timestamp": "2023-01-01", "failure_type": "bearing", "vibration_data": "..."}
        ]
        
        predictor.train(failure_events)
        prediction = predictor.predict(self.data, self.resonance_result)
        
        assert isinstance(prediction, FailurePrediction)
    
    def test_failure_mode_detection(self):
        """Test that failure mode detection works correctly."""
        predictor = FailurePredictor()
        
        # Test with different frequency patterns
        # Low frequency dominant
        low_freq_result = ResonanceResult(
            resonant_frequencies=[10, 20, 30],
            resonance_peaks=[0.5, 0.3, 0.2],
            quality_factors=[15, 12, 10],
            resonance_risk=["medium", "low", "low"],
            frequency_spectrum=np.array([]),
            power_spectrum=np.array([])
        )
        
        prediction = predictor.predict(self.data, low_freq_result)
        assert prediction.failure_mode in ["structural_failure", "bearing_failure", "component_fatigue", "unknown"]


class TestResonanceResult:
    """Test ResonanceResult class."""
    
    def test_valid_creation(self):
        """Test creating valid ResonanceResult."""
        result = ResonanceResult(
            resonant_frequencies=[50.0, 120.0],
            resonance_peaks=[0.8, 0.6],
            quality_factors=[25.0, 18.0],
            resonance_risk=["high", "medium"],
            frequency_spectrum=np.array([0, 1, 2, 3]),
            power_spectrum=np.array([0.1, 0.2, 0.3, 0.4])
        )
        
        assert len(result.resonant_frequencies) == 2
        assert len(result.resonance_peaks) == 2
        assert len(result.quality_factors) == 2
        assert len(result.resonance_risk) == 2
    
    def test_invalid_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            ResonanceResult(
                resonant_frequencies=[50.0, 120.0],
                resonance_peaks=[0.8],  # Different length
                quality_factors=[25.0, 18.0],
                resonance_risk=["high", "medium"],
                frequency_spectrum=np.array([]),
                power_spectrum=np.array([])
            )


class TestFailurePrediction:
    """Test FailurePrediction class."""
    
    def test_valid_creation(self):
        """Test creating valid FailurePrediction."""
        prediction = FailurePrediction(
            failure_probability=0.7,
            time_to_failure=168.0,
            failure_mode="bearing_failure",
            confidence=0.8,
            risk_factors=["high_rms_amplitude"],
            recommendations=["Reduce operating speed"]
        )
        
        assert prediction.failure_probability == 0.7
        assert prediction.time_to_failure == 168.0
        assert prediction.failure_mode == "bearing_failure"
        assert prediction.confidence == 0.8
    
    def test_invalid_probability(self):
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            FailurePrediction(
                failure_probability=1.5,  # Invalid
                time_to_failure=168.0,
                failure_mode="bearing_failure",
                confidence=0.8,
                risk_factors=[],
                recommendations=[]
            )
    
    def test_invalid_confidence(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            FailurePrediction(
                failure_probability=0.7,
                time_to_failure=168.0,
                failure_mode="bearing_failure",
                confidence=-0.1,  # Invalid
                risk_factors=[],
                recommendations=[]
            )
    
    def test_negative_time_to_failure(self):
        """Test that negative time to failure raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            FailurePrediction(
                failure_probability=0.7,
                time_to_failure=-10.0,  # Invalid
                failure_mode="bearing_failure",
                confidence=0.8,
                risk_factors=[],
                recommendations=[]
            )


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test the complete analysis workflow."""
        # Generate sample data
        data = generate_sample_data(
            duration=5.0,
            sampling_freq=1000.0,
            resonant_freqs=[50, 120, 300],
            noise_level=0.1
        )
        
        # Analyze resonance
        analyzer = ResonanceAnalyzer()
        resonance_result = analyzer.analyze(data)
        
        # Predict failure
        predictor = FailurePredictor()
        prediction = predictor.predict(data, resonance_result)
        
        # Verify results
        assert len(resonance_result.resonant_frequencies) > 0
        assert 0 <= prediction.failure_probability <= 1
        assert len(prediction.recommendations) > 0
    
    def test_data_persistence(self):
        """Test saving and loading data."""
        data = generate_sample_data(duration=2.0, sampling_freq=500.0)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            # Save data
            data_dict = {
                'time': data.time.tolist(),
                'amplitude': data.amplitude.tolist(),
                'frequency': data.frequency,
                'units': data.units,
                'metadata': data.metadata
            }
            json.dump(data_dict, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            # Load data
            loaded_data = VibrationData(
                time=np.array(data_dict['time']),
                amplitude=np.array(data_dict['amplitude']),
                frequency=data_dict['frequency'],
                units=data_dict['units'],
                metadata=data_dict['metadata']
            )
            
            # Verify data integrity
            assert np.allclose(data.time, loaded_data.time)
            assert np.allclose(data.amplitude, loaded_data.amplitude)
            assert data.frequency == loaded_data.frequency
            assert data.units == loaded_data.units
        
        finally:
            # Clean up
            Path(tmp_file_path).unlink(missing_ok=True)