"""
Visualization utilities for time series forecasting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def plot_forecast(context: np.ndarray,
                 predictions: np.ndarray,
                 ground_truth: Optional[np.ndarray] = None,
                 quantiles: Optional[Dict[str, np.ndarray]] = None,
                 title: str = "Time Series Forecast",
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot time series forecast with context, predictions, and optional ground truth.
    
    Args:
        context: Historical time series data
        predictions: Forecast predictions
        ground_truth: True future values (optional)
        quantiles: Prediction quantiles (optional)
        title: Plot title
        save_path: Path to save plot (optional)
        show_plot: Whether to display the plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create time indices
    context_time = np.arange(len(context))
    forecast_time = np.arange(len(context), len(context) + len(predictions))
    
    # Plot historical data
    ax.plot(context_time, context, 'b-', linewidth=2, label='Historical Data', alpha=0.8)
    
    # Plot predictions
    ax.plot(forecast_time, predictions, 'r-', linewidth=2, label='Predictions')
    
    # Plot ground truth if available
    if ground_truth is not None:
        gt_time = forecast_time[:len(ground_truth)]
        ax.plot(gt_time, ground_truth, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    
    # Plot quantiles if available
    if quantiles:
        # Plot confidence intervals
        if '0.1' in quantiles and '0.9' in quantiles:
            ax.fill_between(forecast_time, quantiles['0.1'], quantiles['0.9'], 
                           alpha=0.2, color='red', label='80% Confidence')
        
        if '0.25' in quantiles and '0.75' in quantiles:
            ax.fill_between(forecast_time, quantiles['0.25'], quantiles['0.75'], 
                           alpha=0.3, color='red', label='50% Confidence')
    
    # Add vertical line at forecast start
    ax.axvline(x=len(context), color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def create_forecast_report(context: np.ndarray,
                          predictions: np.ndarray,
                          ground_truth: np.ndarray,
                          metrics: Dict[str, float],
                          model_name: str,
                          quantiles: Optional[Dict[str, np.ndarray]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive 4-panel forecast report.
    
    Args:
        context: Historical time series data
        predictions: Forecast predictions
        ground_truth: True future values
        metrics: Dictionary of calculated metrics
        model_name: Name of the forecasting model
        quantiles: Prediction quantiles (optional)
        save_path: Path to save report (optional)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} Forecast Report', fontsize=16, fontweight='bold')
    
    # Panel 1: Full time series view
    context_time = np.arange(len(context))
    forecast_time = np.arange(len(context), len(context) + len(predictions))
    
    # Show recent context for better visualization
    recent_context_len = min(200, len(context))
    recent_context = context[-recent_context_len:]
    recent_time = context_time[-recent_context_len:]
    
    ax1.plot(recent_time, recent_context, 'b-', linewidth=2, label='Historical Data', alpha=0.8)
    ax1.plot(forecast_time, ground_truth, 'g-', linewidth=3, label='Ground Truth')
    ax1.plot(forecast_time, predictions, 'r--', linewidth=2, label='Predictions')
    
    if quantiles and '0.1' in quantiles and '0.9' in quantiles:
        ax1.fill_between(forecast_time, quantiles['0.1'], quantiles['0.9'], 
                        alpha=0.2, color='red')
    
    ax1.axvline(x=len(context), color='black', linestyle=':', alpha=0.7)
    ax1.set_title('Panel 1: Full Time Series View')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Forecast detail comparison
    ax2.plot(ground_truth, 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax2.plot(predictions, 'r--', linewidth=2, label='Predictions')
    
    if quantiles:
        if '0.1' in quantiles and '0.9' in quantiles:
            ax2.fill_between(range(len(predictions)), quantiles['0.1'], quantiles['0.9'], 
                            alpha=0.2, color='red')
    
    ax2.set_title('Panel 2: Forecast Detail Comparison')
    ax2.set_xlabel('Forecast Steps')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Residuals
    residuals = predictions - ground_truth
    ax3.plot(residuals, 'purple', linewidth=2, label='Prediction Error')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(range(len(residuals)), residuals, 0, alpha=0.3, color='purple')
    ax3.set_title('Panel 3: Prediction Residuals')
    ax3.set_xlabel('Forecast Steps')
    ax3.set_ylabel('Error (Predicted - Actual)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Metrics and information
    ax4.axis('off')
    
    # Create metrics text
    metrics_text = f"""
PERFORMANCE METRICS
{'='*20}
MAE:  {metrics.get('mae', 0):.4f}
RMSE: {metrics.get('rmse', 0):.4f}
MAPE: {metrics.get('mape', 0):.2f}%

ADDITIONAL METRICS
{'='*20}
Correlation: {metrics.get('correlation', 0):.4f}
R²: {metrics.get('r2_score', 0):.4f}
Dir. Accuracy: {metrics.get('directional_accuracy', 0):.1f}%

MODEL INFO
{'='*20}
Model: {model_name}
Context Length: {len(context)}
Forecast Horizon: {len(predictions)}
Has Quantiles: {quantiles is not None}

DATA SUMMARY
{'='*20}
Pred Range: [{predictions.min():.2f}, {predictions.max():.2f}]
Truth Range: [{ground_truth.min():.2f}, {ground_truth.max():.2f}]
    """
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    ax4.set_title('Panel 4: Metrics & Information')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Forecast report saved to {save_path}")
    
    return fig


def plot_multiple_forecasts(context: np.ndarray,
                           forecasts: Dict[str, np.ndarray],
                           ground_truth: Optional[np.ndarray] = None,
                           title: str = "Model Comparison",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot multiple model forecasts for comparison.
    
    Args:
        context: Historical time series data
        forecasts: Dictionary of model_name -> predictions
        ground_truth: True future values (optional)
        title: Plot title
        save_path: Path to save plot (optional)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create time indices
    context_time = np.arange(len(context))
    
    # Plot historical data
    ax.plot(context_time, context, 'k-', linewidth=2, label='Historical Data', alpha=0.8)
    
    # Colors for different models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot each model's forecast
    for i, (model_name, predictions) in enumerate(forecasts.items()):
        forecast_time = np.arange(len(context), len(context) + len(predictions))
        color = colors[i % len(colors)]
        ax.plot(forecast_time, predictions, color=color, linewidth=2, 
               label=f'{model_name}', linestyle='--')
    
    # Plot ground truth if available
    if ground_truth is not None:
        forecast_time = np.arange(len(context), len(context) + len(ground_truth))
        ax.plot(forecast_time, ground_truth, 'g-', linewidth=3, 
               label='Ground Truth', alpha=0.7)
    
    # Add vertical line at forecast start
    ax.axvline(x=len(context), color='black', linestyle=':', alpha=0.7, 
              label='Forecast Start')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Multi-model plot saved to {save_path}")
    
    return fig


def create_metrics_comparison(metrics_results: Dict[str, Dict[str, float]],
                             title: str = "Model Performance Comparison",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart comparing metrics across models.
    
    Args:
        metrics_results: Dictionary of model_name -> metrics_dict
        title: Plot title
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
    """
    # Get all unique metrics
    all_metrics = set()
    for metrics in metrics_results.values():
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(list(all_metrics))
    models = list(metrics_results.keys())
    
    # Create subplots for each metric
    n_metrics = len(all_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(all_metrics):
        ax = axes[i]
        
        values = []
        for model in models:
            values.append(metrics_results[model].get(metric, 0))
        
        bars = ax.bar(models, values)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(models, key=len)) > 8:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(all_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison saved to {save_path}")
    
    return fig


# Additional functions needed by tests

def plot_forecast(context_true: np.ndarray,
                 forecast_true: np.ndarray,
                 forecast_pred: np.ndarray,
                 context_dates: Optional[np.ndarray] = None,
                 forecast_dates: Optional[np.ndarray] = None,
                 confidence_intervals: Optional[np.ndarray] = None,
                 title: str = "Time Series Forecast") -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot forecast with context and prediction data.
    
    Args:
        context_true: Historical true values
        forecast_true: True forecast values
        forecast_pred: Predicted forecast values
        context_dates: Dates for context data (optional)
        forecast_dates: Dates for forecast data (optional)
        confidence_intervals: Confidence intervals (optional)
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    # Input validation
    if len(context_true) == 0 or len(forecast_true) == 0 or len(forecast_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if context_dates is not None and len(context_dates) != len(context_true):
        raise ValueError("Context dates length must match context data length")
    
    if forecast_dates is not None and len(forecast_dates) != len(forecast_true):
        raise ValueError("Forecast dates length must match forecast data length")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create time indices if dates not provided
    if context_dates is None:
        context_time = np.arange(len(context_true))
    else:
        context_time = context_dates
    
    if forecast_dates is None:
        forecast_time = np.arange(len(context_true), len(context_true) + len(forecast_true))
    else:
        forecast_time = forecast_dates
    
    # Plot context data
    ax.plot(context_time, context_true, 'b-', linewidth=2, label='Historical Data', alpha=0.8)
    
    # Plot true forecast
    ax.plot(forecast_time, forecast_true, 'g-', linewidth=2, label='True Forecast', alpha=0.8)
    
    # Plot predicted forecast
    ax.plot(forecast_time, forecast_pred, 'r--', linewidth=2, label='Predicted Forecast')
    
    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        lower_bound = confidence_intervals[:, 0]
        upper_bound = confidence_intervals[:, 1]
        ax.fill_between(forecast_time, lower_bound, upper_bound, 
                       alpha=0.3, color='red', label='Confidence Interval')
    
    # Add vertical line at forecast start
    if context_dates is None and forecast_dates is None:
        ax.axvline(x=len(context_true), color='black', linestyle=':', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_multiple_forecasts(context_true: np.ndarray,
                           forecast_true: np.ndarray,
                           forecasts: Dict[str, np.ndarray],
                           context_dates: Optional[np.ndarray] = None,
                           forecast_dates: Optional[np.ndarray] = None,
                           metrics: Optional[Dict[str, Dict]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple model forecasts for comparison.
    
    Args:
        context_true: Historical true values
        forecast_true: True forecast values
        forecasts: Dictionary of model_name -> predictions
        context_dates: Dates for context data (optional)
        forecast_dates: Dates for forecast data (optional)
        metrics: Dictionary of model_name -> metrics (optional)
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create time indices if dates not provided
    if context_dates is None:
        context_time = np.arange(len(context_true))
    else:
        context_time = context_dates
    
    if forecast_dates is None:
        forecast_time = np.arange(len(context_true), len(context_true) + len(forecast_true))
    else:
        forecast_time = forecast_dates
    
    # Plot context data
    ax.plot(context_time, context_true, 'k-', linewidth=2, label='Historical Data', alpha=0.8)
    
    # Plot true forecast
    ax.plot(forecast_time, forecast_true, 'g-', linewidth=3, label='True Forecast', alpha=0.8)
    
    # Colors for different models
    colors = ['red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Plot each model's forecast
    for i, (model_name, predictions) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        label = model_name
        
        # Add metrics to label if available
        if metrics and model_name in metrics:
            model_metrics = metrics[model_name]
            mae = model_metrics.get('MAE', model_metrics.get('mae'))
            if mae is not None:
                label += f" (MAE: {mae:.3f})"
        
        ax.plot(forecast_time[:len(predictions)], predictions, 
               color=color, linewidth=2, label=label, linestyle='--')
    
    # Add vertical line at forecast start
    if context_dates is None and forecast_dates is None:
        ax.axvline(x=len(context_true), color='black', linestyle=':', alpha=0.7)
    
    ax.set_title('Model Comparison')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_metrics_comparison(metrics_data: Dict[str, Dict[str, float]], 
                           metrics_to_plot: Optional[List[str]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a bar chart comparing metrics across models.
    
    Args:
        metrics_data: Dictionary of model_name -> metrics_dict
        metrics_to_plot: List of specific metrics to plot (optional)
        
    Returns:
        Tuple of (figure, axes)
    """
    if not metrics_data:
        raise ValueError("Metrics data cannot be empty")
    
    # Get all unique metrics
    all_metrics = set()
    for metrics in metrics_data.values():
        all_metrics.update(metrics.keys())
    
    if metrics_to_plot:
        all_metrics = [m for m in metrics_to_plot if m in all_metrics]
    else:
        all_metrics = sorted(list(all_metrics))
    
    models = list(metrics_data.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(all_metrics))
    width = 0.8 / len(models)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        values = []
        for metric in all_metrics:
            values.append(metrics_data[model].get(metric, 0))
        
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_residuals(residuals: np.ndarray, 
                  dates: Optional[np.ndarray] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot residuals analysis with time series, histogram, and Q-Q plot.
    
    Args:
        residuals: Residuals array
        dates: Optional dates array
        
    Returns:
        Tuple of (figure, list of axes)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Time series plot
    if dates is not None:
        axes[0].plot(dates, residuals, 'b-', alpha=0.7)
    else:
        axes[0].plot(residuals, 'b-', alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title('Residuals Over Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residuals')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1].set_title('Residuals Distribution')
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normal)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def save_plot(fig: plt.Figure, filepath: str, dpi: int = 150, bbox_inches: str = 'tight'):
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save the file
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box specification
    """
    # Create directory if it doesn't exist
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Plot saved to {filepath}")


def create_forecast_report(context_true: np.ndarray,
                          forecast_true: np.ndarray,
                          forecast_pred: np.ndarray,
                          metrics: Dict[str, float],
                          model_name: str,
                          save_dir: Optional[Union[str, Path]] = None,
                          context_dates: Optional[np.ndarray] = None,
                          forecast_dates: Optional[np.ndarray] = None,
                          confidence_intervals: Optional[np.ndarray] = None) -> str:
    """
    Create a comprehensive forecast report and save it as an image.
    
    Args:
        context_true: Historical true values
        forecast_true: True forecast values  
        forecast_pred: Predicted forecast values
        metrics: Dictionary of calculated metrics
        model_name: Name of the forecasting model
        save_dir: Directory to save the report (optional)
        context_dates: Dates for context data (optional)
        forecast_dates: Dates for forecast data (optional)
        confidence_intervals: Confidence intervals (optional)
        
    Returns:
        Path to the saved report file
    """
    # Set default save directory
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "test" / "reports"
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the plot
    fig, ax = plot_forecast(
        context_true=context_true,
        forecast_true=forecast_true,
        forecast_pred=forecast_pred,
        context_dates=context_dates,
        forecast_dates=forecast_dates,
        confidence_intervals=confidence_intervals,
        title=f'{model_name} Forecast Report'
    )
    
    # Add metrics text box
    metrics_text = f"""
Metrics:
MAE: {metrics.get('mae', 'N/A')}
RMSE: {metrics.get('rmse', 'N/A')}
Correlation: {metrics.get('correlation', 'N/A')}
    """
    
    ax.text(0.02, 0.98, metrics_text.strip(), transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Save the report
    report_filename = f"{model_name.replace(' ', '_')}_forecast_report.png"
    report_path = save_dir / report_filename
    
    save_plot(fig, str(report_path))
    plt.close(fig)
    
    return str(report_path)