"""
Enhanced visualization utilities for time series forecasting with proper scaling and quantiles.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from chronotick.tsfm.datasets.preprocessing import denormalize_data

logger = logging.getLogger(__name__)

def create_enhanced_forecast_report(
    context_true: np.ndarray,
    forecast_true: np.ndarray,
    forecast_pred: np.ndarray,
    metrics: Dict[str, float],
    model_name: str,
    save_dir: Optional[Union[str, Path]] = None,
    quantiles: Optional[Dict[str, np.ndarray]] = None,
    normalization_stats: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Create enhanced forecast report with proper scaling and quantiles.
    
    Args:
        context_true: Historical true values
        forecast_true: True forecast values  
        forecast_pred: Predicted forecast values
        metrics: Dictionary of calculated metrics
        model_name: Name of the forecasting model
        save_dir: Directory to save the report
        quantiles: Quantile predictions from model
        normalization_stats: Stats for denormalizing data
        metadata: Additional model metadata
        
    Returns:
        Path to the saved report file
    """
    # Set default save directory
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent / "tests" / "integration" / "reports"
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize data if normalization stats provided
    if normalization_stats is not None:
        context_denorm = denormalize_data(context_true, normalization_stats)
        forecast_true_denorm = denormalize_data(forecast_true, normalization_stats)
        forecast_pred_denorm = denormalize_data(forecast_pred, normalization_stats)
        
        # Denormalize quantiles too
        quantiles_denorm = None
        if quantiles is not None:
            quantiles_denorm = {}
            for q_level, q_values in quantiles.items():
                quantiles_denorm[q_level] = denormalize_data(q_values, normalization_stats)
    else:
        context_denorm = context_true
        forecast_true_denorm = forecast_true  
        forecast_pred_denorm = forecast_pred
        quantiles_denorm = quantiles
    
    # Create the enhanced plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} Enhanced Forecast Report', fontsize=16, fontweight='bold')
    
    # Panel 1: Full time series with quantiles
    context_time = np.arange(len(context_denorm))
    forecast_time = np.arange(len(context_denorm), len(context_denorm) + len(forecast_pred_denorm))
    
    # Show recent context for better visualization
    recent_context_len = min(200, len(context_denorm))
    recent_context = context_denorm[-recent_context_len:]
    recent_time = context_time[-recent_context_len:]
    
    ax1.plot(recent_time, recent_context, 'b-', linewidth=2, label='Historical Data', alpha=0.8)
    ax1.plot(forecast_time, forecast_true_denorm, 'g-', linewidth=3, label='True Forecast')
    ax1.plot(forecast_time, forecast_pred_denorm, 'r--', linewidth=2, label='Predicted Forecast')
    
    # Add quantiles if available (handle length mismatches)
    if quantiles_denorm:
        # Check quantile lengths and adjust forecast_time if needed
        quantile_length = None
        for q_level, q_values in quantiles_denorm.items():
            if quantile_length is None:
                quantile_length = len(q_values)
            elif len(q_values) != quantile_length:
                logger.warning(f"Inconsistent quantile lengths: {q_level} has {len(q_values)}, expected {quantile_length}")
        
        if quantile_length is not None and quantile_length < len(forecast_pred_denorm):
            # Quantiles are shorter than predictions, use subset of forecast_time
            quantile_time = forecast_time[:quantile_length]
            logger.info(f"Using quantiles for first {quantile_length} of {len(forecast_pred_denorm)} predictions")
        else:
            quantile_time = forecast_time
        
        # 80% confidence interval
        if '0.1' in quantiles_denorm and '0.9' in quantiles_denorm:
            q_low = quantiles_denorm['0.1'][:len(quantile_time)]
            q_high = quantiles_denorm['0.9'][:len(quantile_time)]
            ax1.fill_between(quantile_time, q_low, q_high, 
                            alpha=0.2, color='red', label='80% Confidence')
        
        # 50% confidence interval  
        if '0.25' in quantiles_denorm and '0.75' in quantiles_denorm:
            q_low = quantiles_denorm['0.25'][:len(quantile_time)]
            q_high = quantiles_denorm['0.75'][:len(quantile_time)]
            ax1.fill_between(quantile_time, q_low, q_high, 
                            alpha=0.3, color='red', label='50% Confidence')
    
    ax1.axvline(x=len(context_denorm), color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    ax1.set_title('Panel 1: Full Time Series with Uncertainty')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Forecast detail comparison with quantiles
    ax2.plot(forecast_true_denorm, 'g-', linewidth=3, label='True Forecast', alpha=0.8)
    ax2.plot(forecast_pred_denorm, 'r--', linewidth=2, label='Predicted Forecast')
    
    if quantiles_denorm:
        # Use same quantile_length as Panel 1
        if quantile_length is not None and quantile_length < len(forecast_pred_denorm):
            quantile_range = range(quantile_length)
        else:
            quantile_range = range(len(forecast_pred_denorm))
        
        # Plot multiple quantile levels
        quantile_levels = sorted([float(q) for q in quantiles_denorm.keys()])
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(quantile_levels)))
        
        for i, q_level in enumerate(quantile_levels):
            if q_level != 0.5:  # Skip median as it's close to main prediction
                q_values = quantiles_denorm[str(q_level)][:len(quantile_range)]
                ax2.plot(quantile_range, q_values, '--', alpha=0.6, 
                        color=colors[i], label=f'{q_level:.1f} quantile')
        
        # Confidence bands
        if '0.1' in quantiles_denorm and '0.9' in quantiles_denorm:
            q_low = quantiles_denorm['0.1'][:len(quantile_range)]
            q_high = quantiles_denorm['0.9'][:len(quantile_range)]
            ax2.fill_between(quantile_range, q_low, q_high, 
                            alpha=0.2, color='red')
    
    ax2.set_title('Panel 2: Forecast Detail with Quantiles')
    ax2.set_xlabel('Forecast Steps')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Residuals analysis
    residuals = forecast_pred_denorm - forecast_true_denorm
    ax3.plot(residuals, 'purple', linewidth=2, label='Prediction Error')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(range(len(residuals)), residuals, 0, alpha=0.3, color='purple')
    
    # Add confidence bands for residuals if quantiles available
    if quantiles_denorm and '0.1' in quantiles_denorm and '0.9' in quantiles_denorm:
        if quantile_length is not None and quantile_length < len(forecast_pred_denorm):
            # Use only the portion with quantiles
            residual_lower = quantiles_denorm['0.1'] - forecast_true_denorm[:quantile_length]
            residual_upper = quantiles_denorm['0.9'] - forecast_true_denorm[:quantile_length]
            quantile_residual_range = range(quantile_length)
        else:
            residual_lower = quantiles_denorm['0.1'] - forecast_true_denorm
            residual_upper = quantiles_denorm['0.9'] - forecast_true_denorm
            quantile_residual_range = range(len(residuals))
        
        ax3.fill_between(quantile_residual_range, residual_lower, residual_upper, 
                        alpha=0.1, color='orange', label='Quantile Range')
    
    ax3.set_title('Panel 3: Prediction Residuals')
    ax3.set_xlabel('Forecast Steps')
    ax3.set_ylabel('Error (Predicted - Actual)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Enhanced metrics and model info
    ax4.axis('off')
    
    # Enhanced metrics text
    metrics_text = f"""
PERFORMANCE METRICS
{'='*25}
MAE:  {metrics.get('mae', 0):.6f}
RMSE: {metrics.get('rmse', 0):.6f}
MAPE: {metrics.get('mape', 0):.2f}%
Correlation: {metrics.get('correlation', 0):.4f}

DATA SCALING
{'='*25}
Original Scale: {'Yes' if normalization_stats else 'No'}
Context Range: [{context_denorm.min():.2f}, {context_denorm.max():.2f}]
Pred Range: [{forecast_pred_denorm.min():.2f}, {forecast_pred_denorm.max():.2f}]
Truth Range: [{forecast_true_denorm.min():.2f}, {forecast_true_denorm.max():.2f}]

MODEL CAPABILITIES
{'='*25}
Model: {model_name}
Context Length: {len(context_denorm)}
Forecast Horizon: {len(forecast_pred_denorm)}
Has Quantiles: {quantiles_denorm is not None}
Quantile Levels: {list(quantiles_denorm.keys()) if quantiles_denorm else 'None'}

ENHANCED FEATURES
{'='*25}"""

    if metadata:
        if metadata.get('multivariate_support'):
            metrics_text += f"\nMultivariate: {metadata.get('multivariate_support')}"
        if metadata.get('covariates_support'):
            metrics_text += f"\nCovariates: {metadata.get('covariates_support')}"
        if 'model_repo' in metadata:
            metrics_text += f"\nModel Repo: {metadata['model_repo']}"
    
    ax4.text(0.02, 0.98, metrics_text.strip(), transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    ax4.set_title('Panel 4: Enhanced Metrics & Model Info')
    
    plt.tight_layout()
    
    # Save the report
    report_filename = f"{model_name.replace(' ', '_')}_Integration_forecast_report.png"
    report_path = save_dir / report_filename
    
    fig.savefig(str(report_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Enhanced forecast report saved to {report_path}")
    return str(report_path)


def create_multivariate_forecast_report(
    multivariate_results: Dict[str, Dict],
    model_name: str,
    save_dir: Optional[Union[str, Path]] = None,
    normalization_stats: Optional[Dict] = None
) -> str:
    """
    Create multivariate forecast visualization report.
    
    Args:
        multivariate_results: Dict of {var_name: {context, forecast_true, forecast_pred, quantiles}}
        model_name: Name of the forecasting model
        save_dir: Directory to save the report
        normalization_stats: Stats for denormalizing data
        
    Returns:
        Path to the saved report file
    """
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent / "tests" / "integration" / "reports"
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_vars = len(multivariate_results)
    n_cols = min(2, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if n_vars == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'{model_name} Multivariate Forecast Report', fontsize=16, fontweight='bold')
    
    for i, (var_name, var_data) in enumerate(multivariate_results.items()):
        ax = axes[i]
        
        context = var_data['context']
        forecast_true = var_data['forecast_true']  
        forecast_pred = var_data['forecast_pred']
        quantiles = var_data.get('quantiles')
        
        # Denormalize if needed
        if normalization_stats:
            context = denormalize_data(context, normalization_stats)
            forecast_true = denormalize_data(forecast_true, normalization_stats)
            forecast_pred = denormalize_data(forecast_pred, normalization_stats)
            if quantiles:
                quantiles = {q: denormalize_data(vals, normalization_stats) 
                           for q, vals in quantiles.items()}
        
        # Plot data
        context_time = np.arange(len(context))
        forecast_time = np.arange(len(context), len(context) + len(forecast_pred))
        
        # Show recent context
        recent_len = min(100, len(context))
        ax.plot(context_time[-recent_len:], context[-recent_len:], 'b-', linewidth=2, alpha=0.8)
        ax.plot(forecast_time, forecast_true, 'g-', linewidth=2, label='True')
        ax.plot(forecast_time, forecast_pred, 'r--', linewidth=2, label='Predicted')
        
        # Add quantiles
        if quantiles and '0.1' in quantiles and '0.9' in quantiles:
            ax.fill_between(forecast_time, quantiles['0.1'], quantiles['0.9'], 
                           alpha=0.2, color='red', label='80% CI')
        
        ax.axvline(x=len(context), color='black', linestyle=':', alpha=0.7)
        ax.set_title(f'Variable: {var_name}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the report
    report_filename = f"{model_name.replace(' ', '_')}_Multivariate_forecast_report.png"
    report_path = save_dir / report_filename
    
    fig.savefig(str(report_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Multivariate forecast report saved to {report_path}")
    return str(report_path)


def create_covariates_forecast_report(
    context: np.ndarray,
    forecast_true: np.ndarray,
    forecast_pred: np.ndarray,
    covariates_data: Dict[str, np.ndarray],
    model_name: str,
    save_dir: Optional[Union[str, Path]] = None,
    quantiles: Optional[Dict[str, np.ndarray]] = None,
    normalization_stats: Optional[Dict] = None
) -> str:
    """
    Create covariates forecast visualization report.
    
    Args:
        context: Historical target data
        forecast_true: True forecast values
        forecast_pred: Predicted forecast values
        covariates_data: Dict of covariate name -> values
        model_name: Name of the forecasting model
        save_dir: Directory to save the report
        quantiles: Quantile predictions
        normalization_stats: Stats for denormalizing data
        
    Returns:
        Path to the saved report file
    """
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent / "tests" / "integration" / "reports"
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 2x2 layout: main forecast + 3 covariate plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} Covariates Forecast Report', fontsize=16, fontweight='bold')
    
    # Denormalize main forecast data
    if normalization_stats:
        context_denorm = denormalize_data(context, normalization_stats)
        forecast_true_denorm = denormalize_data(forecast_true, normalization_stats)
        forecast_pred_denorm = denormalize_data(forecast_pred, normalization_stats)
        quantiles_denorm = {q: denormalize_data(vals, normalization_stats) 
                           for q, vals in quantiles.items()} if quantiles else None
    else:
        context_denorm = context
        forecast_true_denorm = forecast_true
        forecast_pred_denorm = forecast_pred
        quantiles_denorm = quantiles
    
    # Panel 1: Main forecast with covariates influence
    context_time = np.arange(len(context_denorm))
    forecast_time = np.arange(len(context_denorm), len(context_denorm) + len(forecast_pred_denorm))
    
    recent_len = min(100, len(context_denorm))
    ax1.plot(context_time[-recent_len:], context_denorm[-recent_len:], 'b-', linewidth=2, label='Historical')
    ax1.plot(forecast_time, forecast_true_denorm, 'g-', linewidth=2, label='True Forecast')
    ax1.plot(forecast_time, forecast_pred_denorm, 'r--', linewidth=2, label='Predicted (with Covariates)')
    
    if quantiles_denorm and '0.1' in quantiles_denorm and '0.9' in quantiles_denorm:
        ax1.fill_between(forecast_time, quantiles_denorm['0.1'], quantiles_denorm['0.9'], 
                        alpha=0.2, color='red', label='80% CI')
    
    ax1.axvline(x=len(context_denorm), color='black', linestyle=':', alpha=0.7)
    ax1.set_title('Panel 1: Target Forecast with Covariates')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Target Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panels 2-4: Covariate time series
    covariate_axes = [ax2, ax3, ax4]
    covariate_names = list(covariates_data.keys())[:3]  # Show first 3 covariates
    
    for i, (ax, cov_name) in enumerate(zip(covariate_axes, covariate_names)):
        cov_data = covariates_data[cov_name]
        cov_time = np.arange(len(cov_data))
        
        ax.plot(cov_time, cov_data, 'orange', linewidth=2, label=f'{cov_name}')
        ax.axvline(x=len(context_denorm), color='black', linestyle=':', alpha=0.7, label='Forecast Start')
        
        ax.set_title(f'Panel {i+2}: Covariate - {cov_name}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Covariate Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # If fewer than 3 covariates, add info to remaining panels
    if len(covariate_names) < 3:
        remaining_covs = list(covariates_data.keys())[3:]
        for i in range(len(covariate_names), 3):
            ax = covariate_axes[i]
            ax.axis('off')
            if remaining_covs:
                info_text = f"Additional Covariates:\n" + "\n".join(remaining_covs[:5])
                ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
            else:
                ax.text(0.5, 0.5, f"Total Covariates: {len(covariates_data)}", 
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    # Save the report
    report_filename = f"{model_name.replace(' ', '_')}_Covariates_forecast_report.png"
    report_path = save_dir / report_filename
    
    fig.savefig(str(report_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Covariates forecast report saved to {report_path}")
    return str(report_path)