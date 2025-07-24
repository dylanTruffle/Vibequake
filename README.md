# Vibequake Documentation

Welcome to the Vibequake documentation site! This is the GitHub Pages branch that serves the documentation for the Vibequake library.

## 🌊 About Vibequake

Vibequake is a comprehensive Python library for analyzing vibrational data, detecting resonance conditions, and predicting catastrophic failure in mechanical systems.

**When objects vibe too hard, they can undergo resonance and catastrophic failure.**

## 📚 Documentation

The main documentation is available at: [https://vibequake.dev](https://vibequake.dev)

### Quick Links

- **Installation**: `pip install vibequake`
- **GitHub Repository**: [https://github.com/vibequake/vibequake](https://github.com/vibequake/vibequake)
- **PyPI Package**: [https://pypi.org/project/vibequake/](https://pypi.org/project/vibequake/)
- **Issues**: [https://github.com/vibequake/vibequake/issues](https://github.com/vibequake/vibequake/issues)

## 🚀 Quick Start

```python
import vibequake as vq

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
```

## 📖 Full Documentation

For complete documentation, API reference, and examples, please visit the main documentation site or check out the GitHub repository.

---

**Vibequake**: When objects vibe too hard, we help you predict when they'll break! 🌊💥
