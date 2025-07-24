"""
Command-line interface for Vibequake.

This module provides a CLI for analyzing vibration data and predicting
catastrophic failure from the command line.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .core import ResonanceAnalyzer, FailurePredictor, VibrationData
from .utils import (
    load_vibration_data, 
    save_vibration_data, 
    plot_vibration_spectrum, 
    plot_resonance_analysis,
    generate_sample_data
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vibequake: Analyze vibrational resonance and predict catastrophic failure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a CSV file for resonance
  vibequake analyze resonance data.csv --output results.json

  # Predict failure from vibration data
  vibequake predict failure data.csv --threshold 0.7

  # Generate sample data for testing
  vibequake generate sample --duration 10 --freqs 50,120,300

  # Create visualization plots
  vibequake visualize spectrum data.csv --output plot.png

  # Complete analysis with all features
  vibequake analyze complete data.csv --output analysis.json --plot
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze vibration data')
    analyze_subparsers = analyze_parser.add_subparsers(dest='analysis_type', help='Analysis type')
    
    # Resonance analysis
    resonance_parser = analyze_subparsers.add_parser('resonance', help='Resonance analysis')
    resonance_parser.add_argument('input', help='Input vibration data file')
    resonance_parser.add_argument('--output', '-o', help='Output file for results')
    resonance_parser.add_argument('--min-peak-height', type=float, default=0.1,
                                 help='Minimum peak height (default: 0.1)')
    resonance_parser.add_argument('--min-peak-distance', type=float, default=1.0,
                                 help='Minimum peak distance in Hz (default: 1.0)')
    resonance_parser.add_argument('--prominence', type=float, default=0.05,
                                 help='Prominence threshold (default: 0.05)')
    resonance_parser.add_argument('--plot', action='store_true', help='Generate plot')
    
    # Complete analysis
    complete_parser = analyze_subparsers.add_parser('complete', help='Complete analysis')
    complete_parser.add_argument('input', help='Input vibration data file')
    complete_parser.add_argument('--output', '-o', help='Output file for results')
    complete_parser.add_argument('--plot', action='store_true', help='Generate plot')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict failure')
    predict_subparsers = predict_parser.add_subparsers(dest='prediction_type', help='Prediction type')
    
    failure_parser = predict_subparsers.add_parser('failure', help='Failure prediction')
    failure_parser.add_argument('input', help='Input vibration data file')
    failure_parser.add_argument('--output', '-o', help='Output file for results')
    failure_parser.add_argument('--threshold', type=float, default=0.8,
                               help='Failure threshold (default: 0.8)')
    failure_parser.add_argument('--resonance-file', help='Pre-computed resonance results file')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate sample data')
    generate_subparsers = generate_parser.add_subparsers(dest='generate_type', help='Generate type')
    
    sample_parser = generate_subparsers.add_parser('sample', help='Generate sample vibration data')
    sample_parser.add_argument('--output', '-o', required=True, help='Output file')
    sample_parser.add_argument('--duration', type=float, default=10.0, help='Duration in seconds')
    sample_parser.add_argument('--sampling-freq', type=float, default=1000.0, help='Sampling frequency')
    sample_parser.add_argument('--freqs', default='50,120,300', help='Resonant frequencies (comma-separated)')
    sample_parser.add_argument('--noise', type=float, default=0.1, help='Noise level')
    sample_parser.add_argument('--format', choices=['csv', 'json', 'pickle'], default='csv',
                              help='Output format')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations')
    visualize_subparsers = visualize_parser.add_subparsers(dest='plot_type', help='Plot type')
    
    spectrum_parser = visualize_subparsers.add_parser('spectrum', help='Vibration spectrum plot')
    spectrum_parser.add_argument('input', help='Input vibration data file')
    spectrum_parser.add_argument('--output', '-o', required=True, help='Output plot file')
    spectrum_parser.add_argument('--dpi', type=int, default=150, help='Plot DPI')
    
    resonance_plot_parser = visualize_subparsers.add_parser('resonance', help='Resonance analysis plot')
    resonance_plot_parser.add_argument('input', help='Input vibration data file')
    resonance_plot_parser.add_argument('--output', '-o', required=True, help='Output plot file')
    resonance_plot_parser.add_argument('--dpi', type=int, default=150, help='Plot DPI')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between file formats')
    convert_parser.add_argument('input', help='Input file')
    convert_parser.add_argument('output', help='Output file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'analyze':
            handle_analyze(args)
        elif args.command == 'predict':
            handle_predict(args)
        elif args.command == 'generate':
            handle_generate(args)
        elif args.command == 'visualize':
            handle_visualize(args)
        elif args.command == 'convert':
            handle_convert(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_analyze(args):
    """Handle analyze command."""
    # Load data
    data = load_vibration_data(args.input)
    
    if args.analysis_type == 'resonance':
        # Perform resonance analysis
        analyzer = ResonanceAnalyzer(
            min_peak_height=args.min_peak_height,
            min_peak_distance=args.min_peak_distance,
            prominence_threshold=args.prominence
        )
        result = analyzer.analyze(data)
        
        # Prepare output
        output_data = {
            "resonant_frequencies": result.resonant_frequencies,
            "resonance_peaks": result.resonance_peaks,
            "quality_factors": result.quality_factors,
            "resonance_risk": result.resonance_risk,
            "statistics": {
                "total_resonances": len(result.resonant_frequencies),
                "critical_resonances": sum(1 for r in result.resonance_risk if r == "critical"),
                "high_resonances": sum(1 for r in result.resonance_risk if r == "high"),
                "medium_resonances": sum(1 for r in result.resonance_risk if r == "medium"),
                "low_resonances": sum(1 for r in result.resonance_risk if r == "low"),
                "avg_quality_factor": np.mean(result.quality_factors) if result.quality_factors else 0,
                "data_duration": data.duration,
                "rms_amplitude": data.rms_amplitude,
                "peak_amplitude": data.peak_amplitude
            }
        }
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))
        
        # Generate plot if requested
        if args.plot:
            plot_file = args.output.replace('.json', '_plot.png') if args.output else 'resonance_plot.png'
            fig = plot_resonance_analysis(result)
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved to {plot_file}")
    
    elif args.analysis_type == 'complete':
        # Perform complete analysis
        analyzer = ResonanceAnalyzer()
        resonance_result = analyzer.analyze(data)
        
        predictor = FailurePredictor()
        failure_prediction = predictor.predict(data, resonance_result)
        
        # Prepare output
        output_data = {
            "resonance_analysis": {
                "resonant_frequencies": resonance_result.resonant_frequencies,
                "resonance_peaks": resonance_result.resonance_peaks,
                "quality_factors": resonance_result.quality_factors,
                "resonance_risk": resonance_result.resonance_risk
            },
            "failure_prediction": {
                "failure_probability": failure_prediction.failure_probability,
                "time_to_failure": failure_prediction.time_to_failure,
                "failure_mode": failure_prediction.failure_mode,
                "confidence": failure_prediction.confidence,
                "risk_factors": failure_prediction.risk_factors,
                "recommendations": failure_prediction.recommendations
            },
            "data_statistics": {
                "duration": data.duration,
                "rms_amplitude": data.rms_amplitude,
                "peak_amplitude": data.peak_amplitude,
                "sampling_frequency": data.frequency
            }
        }
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Complete analysis saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))
        
        # Generate plot if requested
        if args.plot:
            plot_file = args.output.replace('.json', '_plot.png') if args.output else 'complete_analysis_plot.png'
            fig = plot_vibration_spectrum(data, resonance_result)
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved to {plot_file}")


def handle_predict(args):
    """Handle predict command."""
    # Load data
    data = load_vibration_data(args.input)
    
    if args.prediction_type == 'failure':
        # Load or compute resonance results
        if args.resonance_file:
            with open(args.resonance_file, 'r') as f:
                resonance_data = json.load(f)
            resonance_result = ResonanceResult(
                resonant_frequencies=resonance_data["resonant_frequencies"],
                resonance_peaks=resonance_data["resonance_peaks"],
                quality_factors=resonance_data["quality_factors"],
                resonance_risk=resonance_data["resonance_risk"],
                frequency_spectrum=np.array([]),  # Not needed for prediction
                power_spectrum=np.array([])
            )
        else:
            analyzer = ResonanceAnalyzer()
            resonance_result = analyzer.analyze(data)
        
        # Perform failure prediction
        predictor = FailurePredictor(failure_threshold=args.threshold)
        prediction = predictor.predict(data, resonance_result)
        
        # Prepare output
        output_data = {
            "failure_probability": prediction.failure_probability,
            "time_to_failure": prediction.time_to_failure,
            "failure_mode": prediction.failure_mode,
            "confidence": prediction.confidence,
            "risk_factors": prediction.risk_factors,
            "recommendations": prediction.recommendations,
            "severity_level": "critical" if prediction.failure_probability >= 0.8 else
                            "high" if prediction.failure_probability >= 0.6 else
                            "medium" if prediction.failure_probability >= 0.4 else "low"
        }
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Failure prediction saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))


def handle_generate(args):
    """Handle generate command."""
    if args.generate_type == 'sample':
        # Parse resonant frequencies
        freqs = [float(f.strip()) for f in args.freqs.split(",")]
        
        # Generate sample data
        data = generate_sample_data(
            duration=args.duration,
            sampling_freq=args.sampling_freq,
            resonant_freqs=freqs,
            noise_level=args.noise
        )
        
        # Save data
        save_vibration_data(data, args.output, format=args.format)
        print(f"Sample data saved to {args.output}")
        print(f"Generated {len(data.time)} data points over {data.duration:.2f} seconds")
        print(f"Resonant frequencies: {freqs}")


def handle_visualize(args):
    """Handle visualize command."""
    # Load data
    data = load_vibration_data(args.input)
    
    if args.plot_type == 'spectrum':
        # Generate spectrum plot
        analyzer = ResonanceAnalyzer()
        resonance_result = analyzer.analyze(data)
        fig = plot_vibration_spectrum(data, resonance_result)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Spectrum plot saved to {args.output}")
    
    elif args.plot_type == 'resonance':
        # Generate resonance plot
        analyzer = ResonanceAnalyzer()
        resonance_result = analyzer.analyze(data)
        fig = plot_resonance_analysis(resonance_result)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Resonance plot saved to {args.output}")


def handle_convert(args):
    """Handle convert command."""
    # Load data
    data = load_vibration_data(args.input)
    
    # Save in new format
    save_vibration_data(data, args.output)
    print(f"Converted {args.input} to {args.output}")


if __name__ == "__main__":
    main()