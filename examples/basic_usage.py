#!/usr/bin/env python3
"""
Basic usage example for Vibequake library.

This example demonstrates how to:
1. Generate sample vibration data
2. Analyze for resonance conditions
3. Predict catastrophic failure
4. Create visualizations
"""

import vibequake as vq
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Run the basic usage example."""
    print("🌊 Vibequake: Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Generate sample vibration data
    print("\n1. Generating sample vibration data...")
    data = vq.generate_sample_data(
        duration=10.0,
        sampling_freq=1000.0,
        resonant_freqs=[50, 120, 300],
        noise_level=0.1
    )
    
    print(f"   Generated {len(data.time)} data points")
    print(f"   Duration: {data.duration:.2f} seconds")
    print(f"   Sampling frequency: {data.frequency:.0f} Hz")
    print(f"   RMS amplitude: {data.rms_amplitude:.3f} {data.units}")
    print(f"   Peak amplitude: {data.peak_amplitude:.3f} {data.units}")
    
    # Step 2: Analyze for resonance
    print("\n2. Analyzing for resonance conditions...")
    analyzer = vq.ResonanceAnalyzer()
    resonance_result = analyzer.analyze(data)
    
    print(f"   Detected {len(resonance_result.resonant_frequencies)} resonant frequencies:")
    for i, (freq, peak, q_factor, risk) in enumerate(zip(
        resonance_result.resonant_frequencies,
        resonance_result.resonance_peaks,
        resonance_result.quality_factors,
        resonance_result.resonance_risk
    )):
        print(f"     {i+1}. {freq:.1f} Hz (Q={q_factor:.1f}, {risk} risk)")
    
    # Step 3: Predict failure
    print("\n3. Predicting catastrophic failure...")
    predictor = vq.FailurePredictor()
    prediction = predictor.predict(data, resonance_result)
    
    print(f"   Failure probability: {prediction.failure_probability:.1%}")
    print(f"   Confidence: {prediction.confidence:.1%}")
    print(f"   Failure mode: {prediction.failure_mode}")
    if prediction.time_to_failure:
        print(f"   Estimated time to failure: {prediction.time_to_failure:.1f} hours")
    else:
        print("   No immediate failure expected")
    
    print(f"   Risk factors: {', '.join(prediction.risk_factors)}")
    print("   Recommendations:")
    for rec in prediction.recommendations:
        print(f"     - {rec}")
    
    # Step 4: Create visualizations
    print("\n4. Creating visualizations...")
    
    # Spectrum plot
    fig1 = vq.plot_vibration_spectrum(data, resonance_result)
    fig1.savefig("vibration_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("   Saved: vibration_spectrum.png")
    
    # Resonance analysis plot
    fig2 = vq.plot_resonance_analysis(resonance_result)
    fig2.savefig("resonance_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("   Saved: resonance_analysis.png")
    
    # Step 5: Save results
    print("\n5. Saving analysis results...")
    
    # Save data
    vq.save_vibration_data(data, "sample_vibration.csv")
    print("   Saved: sample_vibration.csv")
    
    # Save analysis results
    import json
    results = {
        "resonance_analysis": {
            "resonant_frequencies": resonance_result.resonant_frequencies,
            "resonance_peaks": resonance_result.resonance_peaks,
            "quality_factors": resonance_result.quality_factors,
            "resonance_risk": resonance_result.resonance_risk
        },
        "failure_prediction": {
            "failure_probability": prediction.failure_probability,
            "time_to_failure": prediction.time_to_failure,
            "failure_mode": prediction.failure_mode,
            "confidence": prediction.confidence,
            "risk_factors": prediction.risk_factors,
            "recommendations": prediction.recommendations
        }
    }
    
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("   Saved: analysis_results.json")
    
    print("\n✅ Analysis complete! Check the generated files:")
    print("   - vibration_spectrum.png")
    print("   - resonance_analysis.png")
    print("   - sample_vibration.csv")
    print("   - analysis_results.json")


if __name__ == "__main__":
    main()