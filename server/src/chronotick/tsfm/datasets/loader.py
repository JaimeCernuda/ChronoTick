"""
Dataset loading utilities for TSFM library.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Utility class for loading and managing time series datasets."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing datasets. If None, uses package datasets.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
        
        logger.info(f"DatasetLoader initialized with data_dir: {self.data_dir}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets."""
        datasets = []
        for file_path in self.data_dir.glob("*.csv"):
            datasets.append(file_path.stem)
        return sorted(datasets)
    
    def load_dataset(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset (without extension)
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            Loaded DataFrame
        """
        file_path = self.data_dir / f"{dataset_name}.csv"
        
        if not file_path.exists():
            available = self.list_available_datasets()
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found. Available datasets: {available}"
            )
        
        logger.info(f"Loading dataset: {dataset_name}")
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        return df
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset."""
        df = self.load_dataset(dataset_name)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        info = {
            "name": dataset_name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "date_columns": []
        }
        
        # Try to identify date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head())
                    info["date_columns"].append(col)
                except:
                    pass
        
        return info


def load_ett_data(variant: str = "ETTh1", 
                  column: str = "OT",
                  data_dir: Optional[str] = None) -> np.ndarray:
    """
    Load ETT (Electricity Transforming Temperature) dataset.
    
    Args:
        variant: ETT dataset variant (ETTh1, ETTh2, ETTm1, ETTm2)
        column: Column to extract (OT, HUFL, HULL, MUFL, MULL, LUFL, LULL)
        data_dir: Directory containing ETT data
        
    Returns:
        Time series data as numpy array
    """
    loader = DatasetLoader(data_dir)
    
    try:
        df = loader.load_dataset(variant)
        
        if column not in df.columns:
            available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            raise ValueError(f"Column '{column}' not found. Available: {available_cols}")
        
        data = df[column].values.astype(np.float32)
        
        # Remove any NaN values
        data = data[~np.isnan(data)]
        
        logger.info(f"Loaded ETT data: {variant}[{column}], length: {len(data)}")
        return data
        
    except FileNotFoundError:
        logger.warning(f"ETT dataset {variant} not found, creating synthetic data")
        return create_synthetic_data(length=17420, pattern="ett")


def create_synthetic_data(length: int = 2000, 
                         pattern: str = "mixed",
                         noise_level: float = 0.1,
                         seed: Optional[int] = None) -> np.ndarray:
    """
    Create synthetic time series data for testing.
    
    Args:
        length: Length of the time series
        pattern: Pattern type ('linear', 'seasonal', 'mixed', 'ett', 'random')
        noise_level: Amount of noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Synthetic time series data
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(length)
    
    if pattern == "linear":
        data = 0.01 * t + 50
    elif pattern == "seasonal":
        data = 10 * np.sin(2 * np.pi * t / 24) + 50
    elif pattern == "ett":
        # ETT-like pattern
        trend = 0.001 * t
        daily_season = 2.0 * np.sin(2 * np.pi * t / 24)
        weekly_season = 1.0 * np.sin(2 * np.pi * t / (24 * 7))
        data = 50 + trend + daily_season + weekly_season
    elif pattern == "mixed":
        # Complex pattern with multiple components
        trend = 0.005 * t
        daily = 5 * np.sin(2 * np.pi * t / 24)
        weekly = 3 * np.sin(2 * np.pi * t / (24 * 7))
        monthly = 2 * np.sin(2 * np.pi * t / (24 * 30))
        data = 100 + trend + daily + weekly + monthly
    else:  # random
        data = np.cumsum(np.random.randn(length)) + 50
    
    # Add noise
    noise = noise_level * np.random.randn(length) * np.std(data)
    data += noise
    
    logger.info(f"Created synthetic data: pattern={pattern}, length={length}")
    return data.astype(np.float32)


def load_electricity_data(data_dir: Optional[str] = None) -> np.ndarray:
    """Load electricity consumption data if available."""
    loader = DatasetLoader(data_dir)
    
    try:
        df = loader.load_dataset("electricity")
        # Assume first numeric column contains the main time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data = df[numeric_cols[0]].values.astype(np.float32)
            return data[~np.isnan(data)]
    except FileNotFoundError:
        pass
    
    logger.warning("Electricity data not found, creating synthetic data")
    return create_synthetic_data(length=26304, pattern="mixed")  # ~3 years hourly


def load_weather_data(data_dir: Optional[str] = None) -> np.ndarray:
    """Load weather data if available."""
    loader = DatasetLoader(data_dir)
    
    try:
        df = loader.load_dataset("weather")
        # Look for temperature or similar column
        for col in ["temperature", "temp", "T"]:
            if col in df.columns:
                data = df[col].values.astype(np.float32)
                return data[~np.isnan(data)]
        
        # Fallback to first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data = df[numeric_cols[0]].values.astype(np.float32)
            return data[~np.isnan(data)]
            
    except FileNotFoundError:
        pass
    
    logger.warning("Weather data not found, creating synthetic data")
    return create_synthetic_data(length=52696, pattern="seasonal")  # ~6 years hourly