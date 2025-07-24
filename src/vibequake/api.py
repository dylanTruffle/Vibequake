"""
FastAPI web service for vibrational resonance analysis.

This module provides a REST API for analyzing vibration data and predicting
catastrophic failure, with comprehensive Swagger documentation.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import json

from .core import (
    VibrationData, 
    ResonanceAnalyzer, 
    FailurePredictor, 
    ResonanceResult, 
    FailurePrediction
)
from .utils import (
    load_vibration_data, 
    save_vibration_data, 
    plot_vibration_spectrum, 
    plot_resonance_analysis,
    generate_sample_data
)


# Pydantic models for API requests/responses
class VibrationDataRequest(BaseModel):
    """Request model for vibration data analysis."""
    time: List[float] = Field(..., description="Time array in seconds")
    amplitude: List[float] = Field(..., description="Vibration amplitude array")
    frequency: float = Field(..., description="Sampling frequency in Hz")
    units: str = Field(default="m/s²", description="Units of amplitude")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ResonanceAnalysisRequest(BaseModel):
    """Request model for resonance analysis."""
    data: VibrationDataRequest = Field(..., description="Vibration data to analyze")
    min_peak_height: float = Field(default=0.1, description="Minimum peak height relative to max amplitude")
    min_peak_distance: float = Field(default=1.0, description="Minimum distance between peaks in Hz")
    prominence_threshold: float = Field(default=0.05, description="Minimum peak prominence for detection")


class FailurePredictionRequest(BaseModel):
    """Request model for failure prediction."""
    data: VibrationDataRequest = Field(..., description="Vibration data to analyze")
    resonance_result: Optional[Dict[str, Any]] = Field(default=None, description="Pre-computed resonance results")
    failure_threshold: float = Field(default=0.8, description="Threshold for failure probability")


class ResonanceAnalysisResponse(BaseModel):
    """Response model for resonance analysis."""
    resonant_frequencies: List[float] = Field(..., description="Detected resonant frequencies in Hz")
    resonance_peaks: List[float] = Field(..., description="Peak amplitudes at resonant frequencies")
    quality_factors: List[float] = Field(..., description="Quality factors (Q) for each resonance")
    resonance_risk: List[str] = Field(..., description="Risk level for each resonance")
    statistics: Dict[str, Any] = Field(..., description="Analysis statistics")
    risk_summary: Dict[str, int] = Field(..., description="Summary of risk levels")


class FailurePredictionResponse(BaseModel):
    """Response model for failure prediction."""
    failure_probability: float = Field(..., description="Probability of failure (0-1)")
    time_to_failure: Optional[float] = Field(None, description="Estimated time to failure in hours")
    failure_mode: str = Field(..., description="Predicted failure mode")
    confidence: float = Field(..., description="Confidence level in the prediction (0-1)")
    risk_factors: List[str] = Field(..., description="Contributing risk factors")
    recommendations: List[str] = Field(..., description="Recommendations to mitigate risk")
    severity_level: str = Field(..., description="Overall severity level")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Vibequake API",
        description="""
        # Vibequake: Vibrational Resonance Analysis and Failure Prediction API
        
        A comprehensive API for analyzing vibrational data, detecting resonance conditions,
        and predicting catastrophic failure in mechanical systems.
        
        ## Features
        
        - **Resonance Analysis**: Detect resonant frequencies and assess risk levels
        - **Failure Prediction**: Predict probability and timing of catastrophic failure
        - **Data Visualization**: Generate plots and visualizations
        - **Multiple Formats**: Support for CSV, JSON, and other data formats
        - **Real-time Analysis**: Process vibration data in real-time
        
        ## Quick Start
        
        1. Upload your vibration data or use the sample data generator
        2. Run resonance analysis to detect problematic frequencies
        3. Get failure predictions and recommendations
        4. Download visualizations and reports
        
        ## Use Cases
        
        - Industrial machinery monitoring
        - Structural health monitoring
        - Predictive maintenance
        - Quality control in manufacturing
        - Research and development
        
        For more information, visit [https://vibequake.dev](https://vibequake.dev)
        """,
        version="0.1.0",
        contact={
            "name": "Vibequake Team",
            "email": "team@vibequake.dev",
            "url": "https://vibequake.dev"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Basic API information and links
    """
    return {
        "message": "Welcome to Vibequake API",
        "version": "0.1.0",
        "description": "Vibrational resonance analysis and failure prediction",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    from datetime import datetime
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/analyze/resonance", response_model=ResonanceAnalysisResponse)
async def analyze_resonance(request: ResonanceAnalysisRequest):
    """
    Analyze vibration data for resonance conditions.
    
    This endpoint performs a comprehensive resonance analysis on the provided
    vibration data, detecting resonant frequencies, calculating quality factors,
    and assessing risk levels.
    
    Args:
        request: Resonance analysis request containing vibration data and parameters
        
    Returns:
        Resonance analysis results with detected frequencies and risk assessment
        
    Raises:
        HTTPException: If data is invalid or analysis fails
    """
    try:
        # Convert request to VibrationData
        data = VibrationData(
            time=np.array(request.data.time),
            amplitude=np.array(request.data.amplitude),
            frequency=request.data.frequency,
            units=request.data.units,
            metadata=request.data.metadata
        )
        
        # Create analyzer with custom parameters
        analyzer = ResonanceAnalyzer(
            min_peak_height=request.min_peak_height,
            min_peak_distance=request.min_peak_distance,
            prominence_threshold=request.prominence_threshold
        )
        
        # Perform analysis
        result = analyzer.analyze(data)
        
        # Calculate statistics
        statistics = {
            "total_resonances": len(result.resonant_frequencies),
            "critical_resonances": sum(1 for r in result.resonance_risk if r == "critical"),
            "high_resonances": sum(1 for r in result.resonance_risk if r == "high"),
            "medium_resonances": sum(1 for r in result.resonance_risk if r == "medium"),
            "low_resonances": sum(1 for r in result.resonance_risk if r == "low"),
            "avg_quality_factor": np.mean(result.quality_factors) if result.quality_factors else 0,
            "max_quality_factor": np.max(result.quality_factors) if result.quality_factors else 0,
            "data_duration": data.duration,
            "rms_amplitude": data.rms_amplitude,
            "peak_amplitude": data.peak_amplitude
        }
        
        # Risk summary
        risk_summary = {}
        for risk in result.resonance_risk:
            risk_summary[risk] = risk_summary.get(risk, 0) + 1
        
        return ResonanceAnalysisResponse(
            resonant_frequencies=result.resonant_frequencies,
            resonance_peaks=result.resonance_peaks,
            quality_factors=result.quality_factors,
            resonance_risk=result.resonance_risk,
            statistics=statistics,
            risk_summary=risk_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/failure", response_model=FailurePredictionResponse)
async def predict_failure(request: FailurePredictionRequest):
    """
    Predict catastrophic failure based on vibration analysis.
    
    This endpoint analyzes vibration data and resonance results to predict
    the probability and timing of catastrophic failure, providing actionable
    recommendations for risk mitigation.
    
    Args:
        request: Failure prediction request containing vibration data and optional resonance results
        
    Returns:
        Failure prediction with probability, timing, and recommendations
        
    Raises:
        HTTPException: If data is invalid or prediction fails
    """
    try:
        # Convert request to VibrationData
        data = VibrationData(
            time=np.array(request.data.time),
            amplitude=np.array(request.data.amplitude),
            frequency=request.data.frequency,
            units=request.data.units,
            metadata=request.data.metadata
        )
        
        # Get or compute resonance results
        if request.resonance_result:
            # Use provided resonance results
            resonance_result = ResonanceResult(
                resonant_frequencies=request.resonance_result["resonant_frequencies"],
                resonance_peaks=request.resonance_result["resonance_peaks"],
                quality_factors=request.resonance_result["quality_factors"],
                resonance_risk=request.resonance_result["resonance_risk"],
                frequency_spectrum=np.array(request.resonance_result["frequency_spectrum"]),
                power_spectrum=np.array(request.resonance_result["power_spectrum"])
            )
        else:
            # Compute resonance results
            analyzer = ResonanceAnalyzer()
            resonance_result = analyzer.analyze(data)
        
        # Create predictor
        predictor = FailurePredictor(failure_threshold=request.failure_threshold)
        
        # Make prediction
        prediction = predictor.predict(data, resonance_result)
        
        # Determine severity level
        if prediction.failure_probability >= 0.8:
            severity = "critical"
        elif prediction.failure_probability >= 0.6:
            severity = "high"
        elif prediction.failure_probability >= 0.4:
            severity = "medium"
        else:
            severity = "low"
        
        return FailurePredictionResponse(
            failure_probability=prediction.failure_probability,
            time_to_failure=prediction.time_to_failure,
            failure_mode=prediction.failure_mode,
            confidence=prediction.confidence,
            risk_factors=prediction.risk_factors,
            recommendations=prediction.recommendations,
            severity_level=severity
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze/complete")
async def complete_analysis(request: VibrationDataRequest):
    """
    Perform complete analysis including resonance detection and failure prediction.
    
    This endpoint combines resonance analysis and failure prediction in a single
    request, providing comprehensive results for vibration data analysis.
    
    Args:
        request: Vibration data for complete analysis
        
    Returns:
        Complete analysis results including resonance and failure prediction
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Convert request to VibrationData
        data = VibrationData(
            time=np.array(request.time),
            amplitude=np.array(request.amplitude),
            frequency=request.frequency,
            units=request.units,
            metadata=request.metadata
        )
        
        # Perform resonance analysis
        analyzer = ResonanceAnalyzer()
        resonance_result = analyzer.analyze(data)
        
        # Perform failure prediction
        predictor = FailurePredictor()
        failure_prediction = predictor.predict(data, resonance_result)
        
        return {
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
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/visualize/spectrum")
async def generate_spectrum_plot(request: VibrationDataRequest):
    """
    Generate vibration spectrum visualization.
    
    This endpoint creates a comprehensive visualization of the vibration data
    including time domain, frequency domain, statistics, and risk assessment.
    
    Args:
        request: Vibration data for visualization
        
    Returns:
        Base64 encoded PNG image of the spectrum plot
        
    Raises:
        HTTPException: If visualization fails
    """
    try:
        # Convert request to VibrationData
        data = VibrationData(
            time=np.array(request.time),
            amplitude=np.array(request.amplitude),
            frequency=request.frequency,
            units=request.units,
            metadata=request.metadata
        )
        
        # Perform resonance analysis for overlay
        analyzer = ResonanceAnalyzer()
        resonance_result = analyzer.analyze(data)
        
        # Generate plot
        fig = plot_vibration_spectrum(data, resonance_result)
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "image": image_base64,
            "format": "png",
            "description": "Vibration spectrum analysis plot"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/visualize/resonance")
async def generate_resonance_plot(request: ResonanceAnalysisRequest):
    """
    Generate detailed resonance analysis visualization.
    
    This endpoint creates specialized plots for resonance analysis including
    power spectrum with peaks, quality factors, and risk distribution.
    
    Args:
        request: Resonance analysis request
        
    Returns:
        Base64 encoded PNG image of the resonance analysis plot
        
    Raises:
        HTTPException: If visualization fails
    """
    try:
        # Convert request to VibrationData
        data = VibrationData(
            time=np.array(request.data.time),
            amplitude=np.array(request.data.amplitude),
            frequency=request.data.frequency,
            units=request.data.units,
            metadata=request.data.metadata
        )
        
        # Perform analysis
        analyzer = ResonanceAnalyzer(
            min_peak_height=request.min_peak_height,
            min_peak_distance=request.min_peak_distance,
            prominence_threshold=request.prominence_threshold
        )
        resonance_result = analyzer.analyze(data)
        
        # Generate plot
        fig = plot_resonance_analysis(resonance_result)
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "image": image_base64,
            "format": "png",
            "description": "Resonance analysis detailed plot"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload/analyze")
async def upload_and_analyze(
    file: UploadFile = File(..., description="Vibration data file (CSV, JSON, etc.)"),
    analysis_type: str = Form("complete", description="Type of analysis: 'resonance', 'failure', or 'complete'")
):
    """
    Upload vibration data file and perform analysis.
    
    This endpoint accepts file uploads in various formats (CSV, JSON, etc.)
    and performs the requested type of analysis.
    
    Args:
        file: Uploaded vibration data file
        analysis_type: Type of analysis to perform
        
    Returns:
        Analysis results based on the requested type
        
    Raises:
        HTTPException: If file upload or analysis fails
    """
    try:
        # Read file content
        content = await file.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load data
            data = load_vibration_data(tmp_file_path)
            
            # Perform requested analysis
            if analysis_type == "resonance":
                analyzer = ResonanceAnalyzer()
                result = analyzer.analyze(data)
                return {
                    "resonant_frequencies": result.resonant_frequencies,
                    "resonance_peaks": result.resonance_peaks,
                    "quality_factors": result.quality_factors,
                    "resonance_risk": result.resonance_risk
                }
            elif analysis_type == "failure":
                analyzer = ResonanceAnalyzer()
                resonance_result = analyzer.analyze(data)
                predictor = FailurePredictor()
                prediction = predictor.predict(data, resonance_result)
                return {
                    "failure_probability": prediction.failure_probability,
                    "time_to_failure": prediction.time_to_failure,
                    "failure_mode": prediction.failure_mode,
                    "confidence": prediction.confidence,
                    "risk_factors": prediction.risk_factors,
                    "recommendations": prediction.recommendations
                }
            else:  # complete
                analyzer = ResonanceAnalyzer()
                resonance_result = analyzer.analyze(data)
                predictor = FailurePredictor()
                failure_prediction = predictor.predict(data, resonance_result)
                return {
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
                    }
                }
                
        finally:
            # Clean up temporary file
            Path(tmp_file_path).unlink(missing_ok=True)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sample/generate")
async def generate_sample_data_endpoint(
    duration: float = 10.0,
    sampling_freq: float = 1000.0,
    resonant_freqs: str = "50,120,300",
    noise_level: float = 0.1
):
    """
    Generate sample vibration data for testing and demonstration.
    
    This endpoint creates synthetic vibration data with specified parameters
    for testing and demonstration purposes.
    
    Args:
        duration: Duration of the signal in seconds
        sampling_freq: Sampling frequency in Hz
        resonant_freqs: Comma-separated list of resonant frequencies
        noise_level: Level of noise to add
        
    Returns:
        Sample vibration data in JSON format
    """
    try:
        # Parse resonant frequencies
        freqs = [float(f.strip()) for f in resonant_freqs.split(",")]
        
        # Generate sample data
        data = generate_sample_data(
            duration=duration,
            sampling_freq=sampling_freq,
            resonant_freqs=freqs,
            noise_level=noise_level
        )
        
        return {
            "time": data.time.tolist(),
            "amplitude": data.amplitude.tolist(),
            "frequency": data.frequency,
            "units": data.units,
            "metadata": data.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)