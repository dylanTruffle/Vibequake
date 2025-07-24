"""
Utility functions for data handling and visualization.

This module provides functions for loading/saving vibration data and
creating visualizations for analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Union, Dict, Any
import json
import pickle
from pathlib import Path

from .core import VibrationData, ResonanceResult


def load_vibration_data(file_path: Union[str, Path], 
                       format: str = "auto") -> VibrationData:
    """
    Load vibration data from various file formats.
    
    Args:
        file_path: Path to the data file
        format: File format ('csv', 'json', 'pickle', 'numpy', 'auto')
        
    Returns:
        VibrationData object
        
    Raises:
        ValueError: If format is not supported or file cannot be loaded
    """
    file_path = Path(file_path)
    
    if format == "auto":
        format = file_path.suffix.lower().lstrip(".")
    
    if format == "csv":
        return _load_csv(file_path)
    elif format == "json":
        return _load_json(file_path)
    elif format == "pickle":
        return _load_pickle(file_path)
    elif format == "numpy":
        return _load_numpy(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_vibration_data(data: VibrationData, 
                       file_path: Union[str, Path], 
                       format: str = "auto") -> None:
    """
    Save vibration data to various file formats.
    
    Args:
        data: VibrationData object to save
        file_path: Path to save the file
        format: File format ('csv', 'json', 'pickle', 'numpy', 'auto')
        
    Raises:
        ValueError: If format is not supported
    """
    file_path = Path(file_path)
    
    if format == "auto":
        format = file_path.suffix.lower().lstrip(".")
    
    if format == "csv":
        _save_csv(data, file_path)
    elif format == "json":
        _save_json(data, file_path)
    elif format == "pickle":
        _save_pickle(data, file_path)
    elif format == "numpy":
        _save_numpy(data, file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def plot_vibration_spectrum(data: VibrationData, 
                           resonance_result: Optional[ResonanceResult] = None,
                           figsize: tuple = (12, 8)) -> Figure:
    """
    Create a comprehensive vibration spectrum plot.
    
    Args:
        data: Vibration data to plot
        resonance_result: Optional resonance analysis results to overlay
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Time domain plot
    ax1.plot(data.time, data.amplitude, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Amplitude ({data.units})')
    ax1.set_title('Time Domain Signal')
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain plot
    if resonance_result is not None:
        positive_mask = resonance_result.frequency_spectrum > 0
        freqs = resonance_result.frequency_spectrum[positive_mask]
        power = resonance_result.power_spectrum[positive_mask]
        
        ax2.semilogy(freqs, power, 'g-', linewidth=1, alpha=0.8)
        
        # Mark resonant frequencies
        for i, (freq, peak, risk) in enumerate(zip(
            resonance_result.resonant_frequencies,
            resonance_result.resonance_peaks,
            resonance_result.resonance_risk
        )):
            color = {'low': 'green', 'medium': 'orange', 
                    'high': 'red', 'critical': 'purple'}[risk]
            ax2.axvline(freq, color=color, linestyle='--', alpha=0.7)
            ax2.annotate(f'{freq:.1f} Hz\n({risk})', 
                        xy=(freq, peak), xytext=(10, 10),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Frequency Domain')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(1000, freqs[-1] if resonance_result else 100))
    
    # Statistics
    stats_text = f"""
    Duration: {data.duration:.2f} s
    RMS Amplitude: {data.rms_amplitude:.3f} {data.units}
    Peak Amplitude: {data.peak_amplitude:.3f} {data.units}
    Sampling Frequency: {data.frequency:.0f} Hz
    """
    
    if resonance_result:
        stats_text += f"""
        Resonant Frequencies: {len(resonance_result.resonant_frequencies)}
        Critical Resonances: {sum(1 for r in resonance_result.resonance_risk if r == 'critical')}
        """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Statistics')
    
    # Risk assessment
    if resonance_result:
        risk_counts = {}
        for risk in resonance_result.resonance_risk:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        risks = list(risk_counts.keys())
        counts = list(risk_counts.values())
        colors = ['green', 'orange', 'red', 'purple']
        
        ax4.bar(risks, counts, color=colors[:len(risks)], alpha=0.7)
        ax4.set_xlabel('Risk Level')
        ax4.set_ylabel('Number of Resonances')
        ax4.set_title('Resonance Risk Distribution')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No resonance analysis\navailable', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Risk Assessment')
    
    plt.tight_layout()
    return fig


def plot_resonance_analysis(resonance_result: ResonanceResult,
                           figsize: tuple = (12, 8)) -> Figure:
    """
    Create detailed resonance analysis plots.
    
    Args:
        resonance_result: Resonance analysis results
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Power spectrum with peaks
    positive_mask = resonance_result.frequency_spectrum > 0
    freqs = resonance_result.frequency_spectrum[positive_mask]
    power = resonance_result.power_spectrum[positive_mask]
    
    ax1.semilogy(freqs, power, 'b-', linewidth=1, alpha=0.8)
    
    # Mark peaks with colors based on risk
    colors = {'low': 'green', 'medium': 'orange', 'high': 'red', 'critical': 'purple'}
    for freq, peak, risk in zip(resonance_result.resonant_frequencies,
                               resonance_result.resonance_peaks,
                               resonance_result.resonance_risk):
        color = colors[risk]
        ax1.plot(freq, peak, 'o', color=color, markersize=8)
        ax1.annotate(f'{freq:.1f} Hz\nQ={resonance_result.quality_factors[resonance_result.resonant_frequencies.index(freq)]:.0f}',
                    xy=(freq, peak), xytext=(10, 10),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title('Resonance Peaks')
    ax1.grid(True, alpha=0.3)
    
    # Quality factors
    if resonance_result.quality_factors:
        ax2.bar(range(len(resonance_result.quality_factors)), 
                resonance_result.quality_factors,
                color=[colors[risk] for risk in resonance_result.resonance_risk],
                alpha=0.7)
        ax2.set_xlabel('Resonance Index')
        ax2.set_ylabel('Quality Factor (Q)')
        ax2.set_title('Quality Factors')
        ax2.grid(True, alpha=0.3)
    
    # Frequency vs Quality Factor
    if resonance_result.quality_factors:
        scatter = ax3.scatter(resonance_result.resonant_frequencies,
                             resonance_result.quality_factors,
                             c=[colors[risk] for risk in resonance_result.resonance_risk],
                             s=100, alpha=0.7)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Quality Factor (Q)')
        ax3.set_title('Frequency vs Quality Factor')
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, label=risk)
                          for risk, color in colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right')
    
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
        ax4.set_title('Risk Distribution')
    
    plt.tight_layout()
    return fig


def generate_sample_data(duration: float = 10.0,
                        sampling_freq: float = 1000.0,
                        resonant_freqs: Optional[list] = None,
                        noise_level: float = 0.1) -> VibrationData:
    """
    Generate sample vibration data for testing and demonstration.
    
    Args:
        duration: Duration of the signal in seconds
        sampling_freq: Sampling frequency in Hz
        resonant_freqs: List of resonant frequencies to include
        noise_level: Level of noise to add
        
    Returns:
        VibrationData object with sample data
    """
    if resonant_freqs is None:
        resonant_freqs = [50, 120, 300]
    
    # Time array
    time = np.linspace(0, duration, int(duration * sampling_freq))
    
    # Generate signal with resonant frequencies
    signal_components = []
    for freq in resonant_freqs:
        # Add some amplitude variation and phase
        amplitude = 1.0 + 0.5 * np.sin(2 * np.pi * 0.1 * time)
        phase = np.random.uniform(0, 2 * np.pi)
        component = amplitude * np.sin(2 * np.pi * freq * time + phase)
        signal_components.append(component)
    
    # Combine components
    amplitude = np.sum(signal_components, axis=0)
    
    # Add noise
    noise = noise_level * np.random.normal(0, 1, len(time))
    amplitude += noise
    
    # Add some non-stationary behavior
    amplitude *= (1 + 0.2 * np.sin(2 * np.pi * 0.05 * time))
    
    return VibrationData(
        time=time,
        amplitude=amplitude,
        frequency=sampling_freq,
        units="m/s²",
        metadata={
            "type": "sample_data",
            "resonant_frequencies": resonant_freqs,
            "noise_level": noise_level
        }
    )


# Private helper functions for file I/O

def _load_csv(file_path: Path) -> VibrationData:
    """Load data from CSV file."""
    df = pd.read_csv(file_path)
    
    required_cols = ['time', 'amplitude']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Try to get frequency from metadata or estimate
    frequency = df.get('frequency', None)
    if frequency is None:
        # Estimate from time differences
        time_diffs = np.diff(df['time'])
        frequency = 1.0 / np.mean(time_diffs)
    
    units = df.get('units', 'm/s²').iloc[0] if 'units' in df.columns else 'm/s²'
    
    return VibrationData(
        time=df['time'].values,
        amplitude=df['amplitude'].values,
        frequency=frequency,
        units=units
    )


def _save_csv(data: VibrationData, file_path: Path) -> None:
    """Save data to CSV file."""
    df = pd.DataFrame({
        'time': data.time,
        'amplitude': data.amplitude,
        'frequency': [data.frequency] * len(data.time),
        'units': [data.units] * len(data.time)
    })
    df.to_csv(file_path, index=False)


def _load_json(file_path: Path) -> VibrationData:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    
    return VibrationData(
        time=np.array(data_dict['time']),
        amplitude=np.array(data_dict['amplitude']),
        frequency=data_dict['frequency'],
        units=data_dict.get('units', 'm/s²'),
        metadata=data_dict.get('metadata')
    )


def _save_json(data: VibrationData, file_path: Path) -> None:
    """Save data to JSON file."""
    data_dict = {
        'time': data.time.tolist(),
        'amplitude': data.amplitude.tolist(),
        'frequency': data.frequency,
        'units': data.units,
        'metadata': data.metadata
    }
    
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=2)


def _load_pickle(file_path: Path) -> VibrationData:
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _save_pickle(data: VibrationData, file_path: Path) -> None:
    """Save data to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def _load_numpy(file_path: Path) -> VibrationData:
    """Load data from numpy file."""
    data = np.load(file_path, allow_pickle=True).item()
    return VibrationData(
        time=data['time'],
        amplitude=data['amplitude'],
        frequency=data['frequency'],
        units=data.get('units', 'm/s²'),
        metadata=data.get('metadata')
    )


def _save_numpy(data: VibrationData, file_path: Path) -> None:
    """Save data to numpy file."""
    data_dict = {
        'time': data.time,
        'amplitude': data.amplitude,
        'frequency': data.frequency,
        'units': data.units,
        'metadata': data.metadata
    }
    np.save(file_path, data_dict)