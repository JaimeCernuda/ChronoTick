# ChronoTick Real Data Implementation Plan

## Executive Summary

This document outlines the plan to replace the current synthetic data system with a real clock measurement infrastructure that uses NTP as reference time, implements proper error bounds, and provides accurate data fusion between CPU and GPU models following the design specifications.

## Current State Analysis

### Critical Issues to Address

1. **Synthetic Data Problem**: Current system uses `ClockDataGenerator.generate_offset_sequence()` which produces fake clock offsets - this is misleading and unusable in production
2. **Missing Reference Clock**: No actual time reference for measuring real clock drift
3. **Incomplete Error Bounds**: Error bounds don't account for multiple uncertainty sources
4. **Inconsistent Data Fusion**: CPU/GPU model combination needs proper temporal weighting

## Implementation Plan

### Phase 1: Real Clock Measurement Infrastructure

#### 1.1 NTP Client Implementation
```python
class NTPClient:
    """High-precision NTP client for reference time measurements"""
    
    def __init__(self, servers: List[str], timeout: float = 2.0):
        self.ntp_servers = servers  # ["pool.ntp.org", "time.google.com", "time.cloudflare.com"]
        self.timeout = timeout
        self.measurement_history = []
    
    def measure_offset(self) -> NTPMeasurement:
        """
        Measure clock offset vs NTP reference
        Returns: NTPMeasurement with offset, delay, stratum, precision
        """
        
    def get_best_measurement(self) -> Tuple[float, float, float]:
        """
        Query multiple NTP servers and return best measurement
        Returns: (offset, uncertainty, stratum)
        """
```

#### 1.2 Clock Measurement Collector
```python
class ClockMeasurementCollector:
    """Collects real clock offset measurements with configurable timing"""
    
    def __init__(self, config: ClockMeasurementConfig):
        self.ntp_client = NTPClient(config.ntp_servers)
        self.warm_up_duration = config.warm_up.duration_seconds  # 180s
        self.warm_up_interval = config.warm_up.measurement_interval  # 1.0s
        self.normal_interval = config.normal_operation.measurement_interval  # 300.0s
        
        # Real measurement storage
        self.offset_measurements = []  # (timestamp, local_time - ntp_time)
        self.drift_measurements = []   # (timestamp, drift_rate)
        
    def collect_real_offset(self) -> ClockMeasurement:
        """
        Get real clock offset measurement vs NTP
        Returns: ClockMeasurement with offset, uncertainty, measurement_time
        """
        
    def start_collection(self):
        """Start collecting with warm-up then normal intervals"""
```

#### 1.3 Configuration Extensions
```yaml
# Add to config.yaml
clock_measurement:
  warm_up:
    duration_seconds: 180        # 3 minutes warm-up
    measurement_interval: 1.0    # Every second during warm-up
  normal_operation:
    measurement_interval: 300.0  # Every 5 minutes during normal operation
  ntp_servers:
    - "pool.ntp.org"
    - "time.google.com" 
    - "time.cloudflare.com"
    - "time.nist.gov"
  max_ntp_uncertainty: 0.010     # 10ms max NTP uncertainty
  min_stratum: 3                 # Minimum acceptable NTP stratum

# Model prediction intervals aligned with measurement intervals
prediction:
  cpu_model:
    prediction_interval: 30.0    # CPU predicts every 30 seconds
    prediction_horizon: 30       # 30 seconds ahead
  gpu_model:
    prediction_interval: 300.0   # GPU predicts every 5 minutes (matches NTP)
    prediction_horizon: 300      # 5 minutes ahead
```

### Phase 2: Error Bound Framework

#### 2.1 Multi-Source Uncertainty Quantification
```python
@dataclass
class UncertaintyComponents:
    """Complete uncertainty breakdown for clock corrections"""
    ml_uncertainty: float         # From TSFM model prediction intervals
    ntp_measurement_error: float  # NTP network delay and server precision
    time_since_ntp: float        # Seconds since last NTP measurement
    system_state_uncertainty: float  # Temperature/voltage/frequency changes
    drift_extrapolation_error: float  # Error from extrapolating drift
    
    def total_uncertainty(self) -> float:
        """Combine uncertainties using root-sum-square"""
        return math.sqrt(
            self.ml_uncertainty**2 +
            self.ntp_measurement_error**2 + 
            (self.time_since_ntp * self.drift_extrapolation_error)**2 +
            self.system_state_uncertainty**2
        )

class ErrorBoundCalculator:
    """Calculate realistic error bounds for clock corrections"""
    
    def calculate_bounds(self, prediction: ModelPrediction, 
                        time_since_ntp: float,
                        system_state: SystemMetrics) -> Tuple[float, float]:
        """
        Calculate conservative but realistic error bounds
        Returns: (lower_bound, upper_bound)
        """
        uncertainty = UncertaintyComponents(
            ml_uncertainty=prediction.uncertainty,
            ntp_measurement_error=self.estimate_ntp_error(),
            time_since_ntp=time_since_ntp,
            system_state_uncertainty=self.estimate_system_uncertainty(system_state),
            drift_extrapolation_error=self.estimate_drift_error()
        )
        
        total_error = uncertainty.total_uncertainty()
        return (prediction.value - total_error, prediction.value + total_error)
```

#### 2.2 Separate Offset and Drift Bounds
```python
@dataclass 
class CorrectionWithBounds:
    """Clock correction with separate offset and drift error bounds"""
    
    # Primary corrections
    offset_correction: float      # Immediate offset correction
    drift_rate: float            # Current drift rate estimate
    
    # Offset uncertainty
    offset_uncertainty: float    # ± uncertainty in offset correction
    offset_lower_bound: float    # offset - offset_uncertainty  
    offset_upper_bound: float    # offset + offset_uncertainty
    
    # Drift uncertainty  
    drift_uncertainty: float     # ± uncertainty in drift rate
    drift_lower_bound: float     # drift - drift_uncertainty
    drift_upper_bound: float     # drift + drift_uncertainty
    
    # Combined time bounds
    corrected_time_lower: float  # Most conservative lower bound
    corrected_time_upper: float  # Most conservative upper bound
    
    # Metadata
    confidence: float            # Overall confidence [0,1]
    time_since_ntp: float       # Staleness indicator
    measurement_source: str      # "cpu_model", "gpu_model", "fusion"
```

### Phase 3: Operational Data Fusion

#### 3.1 Temporal Model Coordination
```python
class TemporalModelCoordinator:
    """Coordinates CPU and GPU model predictions with proper timing"""
    
    def __init__(self, config: PredictionConfig):
        self.cpu_interval = config.cpu_model.prediction_interval  # 30s
        self.cpu_horizon = config.cpu_model.prediction_horizon    # 30s  
        self.gpu_interval = config.gpu_model.prediction_interval  # 300s
        self.gpu_horizon = config.gpu_model.prediction_horizon    # 300s
        
        # Prediction storage
        self.active_cpu_prediction = None
        self.active_gpu_prediction = None
        self.last_cpu_prediction_time = 0
        self.last_gpu_prediction_time = 0
        
    def should_update_cpu_prediction(self, current_time: float) -> bool:
        """Check if CPU model needs new prediction (every 30 seconds)"""
        return (current_time - self.last_cpu_prediction_time) >= self.cpu_interval
        
    def should_update_gpu_prediction(self, current_time: float) -> bool:
        """Check if GPU model needs new prediction (every 5 minutes)"""
        return (current_time - self.last_gpu_prediction_time) >= self.gpu_interval
        
    def get_current_correction(self, current_time: float) -> CorrectionWithBounds:
        """Get current correction using progressive CPU->GPU weighting"""
        
        # Time progress within current CPU prediction window
        time_in_cpu_window = current_time - self.last_cpu_prediction_time
        cpu_progress = min(time_in_cpu_window / self.cpu_interval, 1.0)
        
        # Progressive weighting: start with CPU, gradually trust GPU more
        cpu_weight = 1.0 - cpu_progress  # 1.0 -> 0.0 over 30 seconds
        gpu_weight = cpu_progress        # 0.0 -> 1.0 over 30 seconds
        
        return self.fuse_predictions(cpu_weight, gpu_weight, current_time)
```

#### 3.2 Data Fusion Algorithm  
```python
class PredictionFusionEngine:
    """Implements design.md inverse-variance weighting for model fusion"""
    
    def fuse_predictions(self, cpu_pred: ModelPrediction, gpu_pred: ModelPrediction,
                        cpu_weight: float, gpu_weight: float) -> CorrectionWithBounds:
        """
        Fuse CPU and GPU predictions using design.md equation 53-54:
        σ ≈ (Q₉₀ - Q₁₀) / 2.56
        ŷ(t) = Σᵢ wᵢ(t) · ŷᵢ(t), with wᵢ(t) = (1/σᵢ²) / (Σⱼ 1/σⱼ²)
        """
        
        # Extract uncertainties from prediction intervals (design.md equation 49-50)
        cpu_uncertainty = self.extract_uncertainty(cpu_pred)
        gpu_uncertainty = self.extract_uncertainty(gpu_pred)
        
        # Inverse-variance weights (design.md equation 54)
        cpu_inv_var = 1.0 / (cpu_uncertainty**2) if cpu_uncertainty > 0 else 0
        gpu_inv_var = 1.0 / (gpu_uncertainty**2) if gpu_uncertainty > 0 else 0
        
        # Apply temporal weighting on top of uncertainty weighting
        weighted_cpu_inv_var = cpu_inv_var * cpu_weight
        weighted_gpu_inv_var = gpu_inv_var * gpu_weight
        
        total_inv_var = weighted_cpu_inv_var + weighted_gpu_inv_var
        
        if total_inv_var > 0:
            final_cpu_weight = weighted_cpu_inv_var / total_inv_var
            final_gpu_weight = weighted_gpu_inv_var / total_inv_var
        else:
            # Fallback to temporal weights if uncertainties unavailable
            final_cpu_weight = cpu_weight
            final_gpu_weight = gpu_weight
            
        # Fused prediction (design.md equation 53)
        fused_offset = (final_cpu_weight * cpu_pred.offset + 
                       final_gpu_weight * gpu_pred.offset)
        fused_drift = (final_cpu_weight * cpu_pred.drift + 
                      final_gpu_weight * gpu_pred.drift)
        
        return self.create_fused_correction(fused_offset, fused_drift, ...)
```

### Phase 4: Dataset Consistency and Retrospective Correction

#### 4.1 Measurement Gap Filling
```python
class DatasetManager:
    """Maintains consistent 1-second measurement frequency for TSFM models"""
    
    def __init__(self):
        self.measurement_dataset = []  # Every-second measurements for models
        self.prediction_history = []   # Predictions between NTP measurements
        
    def fill_measurement_gaps(self, start_time: float, end_time: float,
                             predictions: List[ModelPrediction]):
        """
        Fill gaps between NTP measurements with model predictions
        Maintains 1-second frequency required by TSFM models
        """
        for t in range(int(start_time), int(end_time)):
            if t not in self.measurement_dataset:
                # Use prediction for this timestamp
                pred_offset = self.interpolate_prediction(predictions, t)
                self.measurement_dataset[t] = {
                    'timestamp': t,
                    'offset': pred_offset,
                    'source': 'prediction',
                    'uncertainty': predictions.uncertainty
                }
    
    def apply_retrospective_correction(self, ntp_measurement: NTPMeasurement,
                                     interval_start: float):
        """
        Apply design.md Algorithm 1: Retrospective Bias Correction
        
        δ ← o_t - ô_t
        For i ← 0 to n:
            α ← t_i - t + 1
            ô_t_i'' ← ô_t_i + α · δ
        """
        prediction_error = ntp_measurement.offset - self.get_prediction_at_time(ntp_measurement.timestamp)
        
        # Apply linear weighting correction (design.md lines 94-96)
        for timestamp in range(int(interval_start), int(ntp_measurement.timestamp)):
            alpha = (timestamp - interval_start) / (ntp_measurement.timestamp - interval_start)
            
            # Correct the stored measurement
            self.measurement_dataset[timestamp]['offset'] += alpha * prediction_error
            self.measurement_dataset[timestamp]['corrected'] = True
            
        # Dataset now maintains realistic every-second measurements
```

#### 4.2 Real Data Pipeline
```python
class RealDataPipeline:
    """Complete pipeline replacing synthetic ClockDataGenerator"""
    
    def __init__(self, config: ClockMeasurementConfig):
        self.ntp_collector = ClockMeasurementCollector(config)
        self.model_coordinator = TemporalModelCoordinator(config)
        self.fusion_engine = PredictionFusionEngine()
        self.dataset_manager = DatasetManager()
        self.error_calculator = ErrorBoundCalculator()
        
    def get_real_clock_correction(self, current_time: float) -> CorrectionWithBounds:
        """
        Main function replacing synthetic data generation
        Returns real clock correction with proper error bounds
        """
        
        # 1. Check if new predictions needed
        if self.model_coordinator.should_update_cpu_prediction(current_time):
            self.update_cpu_prediction()
            
        if self.model_coordinator.should_update_gpu_prediction(current_time):
            self.update_gpu_prediction()
            
        # 2. Get fused correction
        correction = self.model_coordinator.get_current_correction(current_time)
        
        # 3. Calculate realistic error bounds
        time_since_ntp = current_time - self.ntp_collector.last_measurement_time
        bounds = self.error_calculator.calculate_bounds(correction, time_since_ntp)
        
        # 4. Check for new NTP measurements
        if self.ntp_collector.has_new_measurement():
            new_ntp = self.ntp_collector.get_latest_measurement()
            self.dataset_manager.apply_retrospective_correction(new_ntp, ...)
            
        return correction
```

### Phase 5: Integration and Testing

#### 5.1 Replace Synthetic Components
```python
# In daemon.py:403-407, REPLACE:
# current_offset = data_generator.generate_offset_sequence(...)  # SYNTHETIC

# WITH:
current_correction = self.real_data_pipeline.get_real_clock_correction(current_time)
current_offset = current_correction.offset_correction
```

#### 5.2 Validation Framework
```python
class RealDataValidator:
    """Validate real data pipeline against known good references"""
    
    def validate_ntp_measurements(self):
        """Verify NTP measurements are reasonable"""
        
    def validate_prediction_accuracy(self):
        """Test prediction accuracy against held-out real measurements"""
        
    def validate_error_bounds(self):
        """Verify error bounds are realistic and conservative"""
```

## Implementation Timeline

### Week 1: Infrastructure
- [ ] Implement NTP client with multiple server support
- [ ] Create clock measurement collector with configurable timing
- [ ] Add configuration extensions for real measurements

### Week 2: Error Bounds
- [ ] Implement multi-source uncertainty quantification
- [ ] Create error bound calculator with realistic estimates  
- [ ] Separate offset and drift uncertainty tracking

### Week 3: Data Fusion
- [ ] Implement temporal model coordinator
- [ ] Create prediction fusion engine with inverse-variance weighting
- [ ] Progressive CPU->GPU weighting system

### Week 4: Dataset Management
- [ ] Implement measurement gap filling system
- [ ] Create retrospective correction algorithm (design.md Algorithm 1)
- [ ] Maintain 1-second measurement frequency for TSFM models

### Week 5: Integration
- [ ] Replace all synthetic data components
- [ ] Integrate real data pipeline into daemon
- [ ] Comprehensive testing and validation

## Success Criteria

1. **No Synthetic Data**: All clock measurements come from real NTP references
2. **Realistic Error Bounds**: Error bounds accurately reflect multiple uncertainty sources
3. **Model Frequency Consistency**: TSFM models receive consistent 1-second measurements
4. **Accurate Data Fusion**: CPU and GPU predictions properly weighted and combined
5. **Production Ready**: System provides reliable microsecond-level corrections with quantified uncertainty

## Risks and Mitigations

1. **NTP Network Issues**: Use multiple NTP servers with fallback and quality filtering
2. **Model Accuracy**: Extensive validation against held-out real measurements
3. **Error Bound Calibration**: Conservative bounds initially, tune based on real performance
4. **Performance Impact**: Optimize NTP measurements and model coordination for minimal overhead

This plan transforms ChronoTick from a synthetic demonstration system into a production-ready clock correction service using real measurements, proper uncertainty quantification, and accurate model fusion.