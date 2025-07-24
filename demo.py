#!/usr/bin/env python3
"""
Vibequake Library Demonstration

This script demonstrates the key features of the Vibequake library
for vibrational resonance analysis and failure prediction.
"""

import sys
import os

# Add the src directory to the path so we can import vibequake
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from vibequake import (
    VibrationData,
    ResonanceAnalyzer,
    FailurePredictor,
    generate_sample_data,
    plot_vibration_spectrum,
    plot_resonance_analysis
)

def main():
    """Run the Vibequake demonstration."""
    print("🌊 Vibequake Library Demonstration")
    print("=" * 50)
    print()
    
    # Step 1: Generate sample vibration data
    print("1. Generating sample vibration data...")
    data = generate_sample_data(
        duration=10.0,
        sampling_freq=1000.0,
        resonant_freqs=[50, 120, 300],
        noise_level=0.1
    )
    
    print(f"   ✅ Generated {len(data.time):,} data points")
    print(f"   📊 Duration: {data.duration:.2f} seconds")
    print(f"   🔄 Sampling frequency: {data.frequency:.0f} Hz")
    print(f"   📈 RMS amplitude: {data.rms_amplitude:.3f} {data.units}")
    print(f"   📊 Peak amplitude: {data.peak_amplitude:.3f} {data.units}")
    print()
    
    # Step 2: Analyze for resonance
    print("2. Analyzing for resonance conditions...")
    analyzer = ResonanceAnalyzer(
        min_peak_height=0.1,
        min_peak_distance=1.0,
        prominence_threshold=0.05
    )
    resonance_result = analyzer.analyze(data)
    
    print(f"   🔍 Detected {len(resonance_result.resonant_frequencies)} resonant frequencies:")
    for i, (freq, peak, q_factor, risk) in enumerate(zip(
        resonance_result.resonant_frequencies,
        resonance_result.resonance_peaks,
        resonance_result.quality_factors,
        resonance_result.resonance_risk
    )):
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}[risk]
        print(f"      {i+1}. {freq:.1f} Hz (Q={q_factor:.1f}, {risk_emoji} {risk} risk)")
    print()
    
    # Step 3: Predict failure
    print("3. Predicting catastrophic failure...")
    predictor = FailurePredictor(failure_threshold=0.8)
    prediction = predictor.predict(data, resonance_result)
    
    # Determine severity emoji
    if prediction.failure_probability >= 0.8:
        severity_emoji = "🔴"
    elif prediction.failure_probability >= 0.6:
        severity_emoji = "🟠"
    elif prediction.failure_probability >= 0.4:
        severity_emoji = "🟡"
    else:
        severity_emoji = "🟢"
    
    print(f"   {severity_emoji} Failure probability: {prediction.failure_probability:.1%}")
    print(f"   🎯 Confidence: {prediction.confidence:.1%}")
    print(f"   🔧 Failure mode: {prediction.failure_mode}")
    
    if prediction.time_to_failure:
        print(f"   ⏰ Estimated time to failure: {prediction.time_to_failure:.1f} hours")
    else:
        print("   ✅ No immediate failure expected")
    
    print(f"   ⚠️  Risk factors: {', '.join(prediction.risk_factors)}")
    print("   💡 Recommendations:")
    for rec in prediction.recommendations:
        print(f"      • {rec}")
    print()
    
    # Step 4: Create visualizations
    print("4. Creating visualizations...")
    
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Vibequake Analysis Results', fontsize=16, fontweight='bold')
    
    # Time domain plot
    ax1.plot(data.time, data.amplitude, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Amplitude ({data.units})')
    ax1.set_title('Time Domain Signal')
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain plot
    positive_mask = resonance_result.frequency_spectrum > 0
    freqs = resonance_result.frequency_spectrum[positive_mask]
    power = resonance_result.power_spectrum[positive_mask]
    
    ax2.semilogy(freqs, power, 'g-', linewidth=1, alpha=0.8)
    
    # Mark resonant frequencies
    colors = {'low': 'green', 'medium': 'orange', 'high': 'red', 'critical': 'purple'}
    for freq, peak, risk in zip(
        resonance_result.resonant_frequencies,
        resonance_result.resonance_peaks,
        resonance_result.resonance_risk
    ):
        color = colors[risk]
        ax2.axvline(freq, color=color, linestyle='--', alpha=0.7)
        ax2.annotate(f'{freq:.1f} Hz\n({risk})', 
                    xy=(freq, peak), xytext=(10, 10),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Frequency Domain')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(1000, freqs[-1] if len(freqs) > 0 else 100))
    
    # Statistics
    stats_text = f"""
Duration: {data.duration:.2f} s
RMS Amplitude: {data.rms_amplitude:.3f} {data.units}
Peak Amplitude: {data.peak_amplitude:.3f} {data.units}
Sampling Frequency: {data.frequency:.0f} Hz

Resonant Frequencies: {len(resonance_result.resonant_frequencies)}
Critical Resonances: {sum(1 for r in resonance_result.resonance_risk if r == 'critical')}

Failure Probability: {prediction.failure_probability:.1%}
Confidence: {prediction.confidence:.1%}
Failure Mode: {prediction.failure_mode}
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Analysis Statistics')
    
    # Risk distribution
    risk_counts = {}
    for risk in resonance_result.resonance_risk:
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    if risk_counts:
        risks = list(risk_counts.keys())
        counts = list(risk_counts.values())
        colors_list = [colors[risk] for risk in risks]
        
        wedges, texts, autotexts = ax4.pie(counts, labels=risks, colors=colors_list,
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Resonance Risk Distribution')
    
    plt.tight_layout()
    plt.savefig('vibequake_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   📊 Saved: vibequake_demo.png")
    print()
    
    # Step 5: Summary
    print("5. Analysis Summary")
    print("   " + "=" * 30)
    
    # Count risk levels
    risk_summary = {}
    for risk in resonance_result.resonance_risk:
        risk_summary[risk] = risk_summary.get(risk, 0) + 1
    
    print(f"   📊 Total resonances detected: {len(resonance_result.resonant_frequencies)}")
    for risk, count in risk_summary.items():
        emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}[risk]
        print(f"      {emoji} {risk.capitalize()}: {count}")
    
    print(f"   🎯 Overall failure probability: {prediction.failure_probability:.1%}")
    print(f"   🎯 Prediction confidence: {prediction.confidence:.1%}")
    
    # Overall assessment
    if prediction.failure_probability >= 0.8:
        assessment = "🔴 CRITICAL - Immediate action required"
    elif prediction.failure_probability >= 0.6:
        assessment = "🟠 HIGH - Monitor closely and plan maintenance"
    elif prediction.failure_probability >= 0.4:
        assessment = "🟡 MEDIUM - Continue monitoring"
    else:
        assessment = "🟢 LOW - Normal operation"
    
    print(f"   📋 Overall assessment: {assessment}")
    print()
    
    print("✅ Demonstration complete!")
    print("📁 Generated files:")
    print("   - vibequake_demo.png (analysis visualization)")
    print()
    print("🌊 Vibequake: When objects vibe too hard, we help you predict when they'll break! 💥")

if __name__ == "__main__":
    main()