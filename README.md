# Vibequake

When objects vibe too hard, they can undergo resonance and catastrophic failure.

[![PyPI version](https://badge.fury.io/py/vibequake.svg)](https://badge.fury.io/py/vibequake)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-vibequake.dev-blue.svg)](https://vibequake.dev)

A comprehensive Python library for analyzing vibrational data, detecting resonance conditions, and predicting catastrophic failure in mechanical systems.

## 🌟 Features

- **Resonance Analysis**: Detect resonant frequencies and assess risk levels
- **Failure Prediction**: Predict probability and timing of catastrophic failure
- **Data Visualization**: Generate comprehensive plots and visualizations
- **Multiple Formats**: Support for CSV, JSON, pickle, and numpy formats
- **REST API**: FastAPI web service with Swagger documentation
- **CLI Interface**: Command-line tools for batch processing
- **Real-time Analysis**: Process vibration data in real-time

## 🚀 Quick Start

### Installation

```bash
pip install vibequake
```

### Basic Usage

```python
import vibequake as vq
import numpy as np

# Generate sample data
data = vq.generate_sample_data(
    duration=10.0,
    sampling_freq=1000.0,
    resonant_freqs=[50, 120, 300],
    noise_level=0.1
)

# Analyze for resonance
analyzer = vq.ResonanceAnalyzer()
resonance_result = analyzer.analyze(data)

# Predict failure
predictor = vq.FailurePredictor()
prediction = predictor.predict(data, resonance_result)

print(f"Detected {len(resonance_result.resonant_frequencies)} resonant frequencies")
print(f"Failure probability: {prediction.failure_probability:.2%}")
print(f"Recommendations: {prediction.recommendations}")
```

### Command Line Interface

```bash
# Analyze a CSV file for resonance
vibequake analyze resonance data.csv --output results.json

# Predict failure from vibration data
vibequake predict failure data.csv --threshold 0.7

# Generate sample data for testing
vibequake generate sample --duration 10 --freqs 50,120,300 --output sample.csv

# Create visualization plots
vibequake visualize spectrum data.csv --output plot.png

# Complete analysis with all features
vibequake analyze complete data.csv --output analysis.json --plot
```

### Web API

Start the FastAPI server:

```bash
python -m vibequake.api
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

## 📊 Use Cases

- **Industrial Machinery Monitoring**: Detect resonance in rotating equipment
- **Structural Health Monitoring**: Identify structural vibrations and potential failures
- **Predictive Maintenance**: Predict equipment failure before it occurs
- **Quality Control**: Monitor manufacturing processes for vibration issues
- **Research & Development**: Analyze vibration patterns in experimental setups

## 🔧 API Reference

### Core Classes

#### `VibrationData`
Container for vibration measurement data.

```python
data = VibrationData(
    time=np.array([0, 0.001, 0.002, ...]),
    amplitude=np.array([0.1, 0.2, 0.15, ...]),
    frequency=1000.0,  # Hz
    units="m/s²"
)
```

#### `ResonanceAnalyzer`
Analyzes vibrational data to detect resonance conditions.

```python
analyzer = ResonanceAnalyzer(
    min_peak_height=0.1,
    min_peak_distance=1.0,
    prominence_threshold=0.05
)
result = analyzer.analyze(data)
```

#### `FailurePredictor`
Predicts catastrophic failure based on vibration analysis.

```python
predictor = FailurePredictor(failure_threshold=0.8)
prediction = predictor.predict(data, resonance_result)
```

### Utility Functions

#### Data I/O
```python
# Load data from various formats
data = vq.load_vibration_data("vibration.csv")
data = vq.load_vibration_data("vibration.json", format="json")

# Save data
vq.save_vibration_data(data, "output.csv")
vq.save_vibration_data(data, "output.json", format="json")
```

#### Visualization
```python
# Generate comprehensive plots
fig = vq.plot_vibration_spectrum(data, resonance_result)
fig = vq.plot_resonance_analysis(resonance_result)

# Save plots
fig.savefig("analysis.png", dpi=150, bbox_inches="tight")
```

#### Sample Data Generation
```python
# Generate synthetic data for testing
data = vq.generate_sample_data(
    duration=10.0,
    sampling_freq=1000.0,
    resonant_freqs=[50, 120, 300],
    noise_level=0.1
)
```

## 🌐 Web API Endpoints

### Analysis Endpoints

- `POST /analyze/resonance` - Analyze vibration data for resonance
- `POST /predict/failure` - Predict catastrophic failure
- `POST /analyze/complete` - Complete analysis with both resonance and failure prediction

### Visualization Endpoints

- `POST /visualize/spectrum` - Generate vibration spectrum plots
- `POST /visualize/resonance` - Generate detailed resonance analysis plots

### Utility Endpoints

- `POST /upload/analyze` - Upload and analyze vibration data files
- `GET /sample/generate` - Generate sample vibration data
- `GET /health` - Health check endpoint

### Example API Request

```python
import requests
import json

# Sample data
data = {
    "time": [0, 0.001, 0.002, ...],
    "amplitude": [0.1, 0.2, 0.15, ...],
    "frequency": 1000.0,
    "units": "m/s²"
}

# Analyze resonance
response = requests.post(
    "http://localhost:8000/analyze/resonance",
    json={"data": data}
)
result = response.json()

print(f"Detected frequencies: {result['resonant_frequencies']}")
print(f"Risk levels: {result['resonance_risk']}")
```

## 📈 Output Examples

### Resonance Analysis Results

```json
{
  "resonant_frequencies": [50.2, 119.8, 300.1],
  "resonance_peaks": [0.85, 0.72, 0.45],
  "quality_factors": [25.3, 18.7, 12.1],
  "resonance_risk": ["high", "medium", "low"],
  "statistics": {
    "total_resonances": 3,
    "critical_resonances": 0,
    "high_resonances": 1,
    "avg_quality_factor": 18.7
  }
}
```

### Failure Prediction Results

```json
{
  "failure_probability": 0.65,
  "time_to_failure": 168.0,
  "failure_mode": "bearing_failure",
  "confidence": 0.78,
  "risk_factors": ["high_rms_amplitude", "critical_resonance_detected"],
  "recommendations": [
    "Immediate shutdown recommended due to critical resonance",
    "Inspect for structural damage",
    "Reduce operating speed or load"
  ],
  "severity_level": "high"
}
```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install vibequake[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=vibequake --cov-report=html
```

## 📚 Documentation

- **API Documentation**: [https://vibequake.dev](https://vibequake.dev)
- **Interactive API**: [http://localhost:8000/docs](http://localhost:8000/docs) (when running locally)
- **Examples**: See the `examples/` directory for detailed usage examples

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/vibequake/vibequake.git
cd vibequake

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for scientific computing
- Visualization powered by [Matplotlib](https://matplotlib.org/)
- Web API built with [FastAPI](https://fastapi.tiangolo.com/)
- Inspired by real-world vibration analysis challenges in mechanical engineering

## 📞 Support

- **Documentation**: [https://vibequake.dev](https://vibequake.dev)
- **Issues**: [GitHub Issues](https://github.com/vibequake/vibequake/issues)
- **Email**: team@vibequake.dev

---

**Vibequake**: When objects vibe too hard, we help you predict when they'll break! 🌊💥
