```py
"""
Chart Pattern Recognition Module
================================

Professional-grade pattern detection for cryptocurrency trading.
Implements: double top/bottom, head and shoulders, flags, wedges,
triangles, and divergence detection (hidden/regular).

Author: Trading Bot Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy.signal import argrelextrema
from scipy.stats import linregress
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Enumeration of supported chart patterns."""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BULL_PENNANT = "bull_pennant"
    BEAR_PENNANT = "bear_pennant"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    REGULAR_BULL_DIVERGENCE = "regular_bull_divergence"
    REGULAR_BEAR_DIVERGENCE = "regular_bear_divergence"
    HIDDEN_BULL_DIVERGENCE = "hidden_bull_divergence"
    HIDDEN_BEAR_DIVERGENCE = "hidden_bear_divergence"


@dataclass
class PatternResult:
    """Data class for pattern detection results."""
    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    start_index: int
    end_index: int
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern result to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "confidence": self.confidence,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class PatternDetector:
    """
    Advanced chart pattern detection engine.
    
    Detects multiple chart patterns using price action analysis,
    local extrema detection, and statistical methods.
    """

    def __init__(
        self,
        lookback_period: int = 100,
        min_pattern_size: int = 10,
        max_pattern_size: int = 80,
        confidence_threshold: float = 0.6,
        extremum_order: int = 5
    ):
        """
        Initialize pattern detector.
        
        Args:
            lookback_period: Number of candles to analyze
            min_pattern_size: Minimum candles for a valid pattern
            max_pattern_size: Maximum candles for a valid pattern
            confidence_threshold: Minimum confidence to report pattern
            extremum_order: Order for local extrema detection
        """
        self.lookback_period = lookback_period
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.confidence_threshold = confidence_threshold
        self.extremum_order = extremum_order
        
        # Validate parameters
        if lookback_period < max_pattern_size:
            raise ValueError("lookback_period must be >= max_pattern_size")
        if min_pattern_size < 5:
            raise ValueError("min_pattern_size must be >= 5")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

    def detect_all_patterns(
        self,
        ohlcv_data: pd.DataFrame
    ) -> List[PatternResult]:
        """
        Detect all supported chart patterns.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            List of detected patterns with confidence scores
        """
        self._validate_data(ohlcv_data)
        
        patterns = []
        data = ohlcv_data.tail(self.lookback_period).copy()
        
        # Detect reversal patterns
        patterns.extend(self._detect_double_top_bottom(data))
        patterns.extend(self._detect_head_and_shoulders(data))
        
        # Detect continuation patterns
        patterns.extend(self._detect_flags_pennants(data))
        patterns.extend(self._detect_wedges(data))
        patterns.extend(self._detect_triangles(data))
        
        # Detect divergences
        patterns.extend(self._detect_divergences(data))
        
        # Filter by confidence threshold
        patterns = [
            p for p in patterns 
            if p.confidence >= self.confidence_threshold
        ]
        
        return patterns

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        if len(data) < self.min_pattern_size:
            raise ValueError(f"Data length {len(data)} < min_pattern_size {self.min_pattern_size}")

    def _find_local_extrema(
        self,
        prices: np.ndarray,
        order: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find local maxima and minima in price series.
        
        Args:
            prices: Array of prices
            order: Number of points on each side to compare
            
        Returns:
            Tuple of (maxima_indices, minima_indices)
        """
        if order is None:
            order = self.extremum_order
            
        max_indices = argrelextrema(prices, np.greater, order=order)[0]
        min_indices = argrelextrema(prices, np.less, order=order)[0]
        
        return max_indices, min_indices

    def _detect_double_top_bottom(
        self,
        data: pd.DataFrame
    ) -> List[PatternResult]:
        """
        Detect double top and double bottom patterns.
        
        Double Top: Two peaks at roughly same level with a valley between
        Double Bottom: Two troughs at roughly same level with a peak between
        """
        patterns = []
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        max_indices, min_indices = self._find_local_extrema(close)
        
        # Detect double tops
        for i in range(len(max_indices) - 1):
            idx1, idx2 = max_indices[i], max_indices[i + 1]
            pattern_size = idx2 - idx1
            
            if not (self.min_pattern_size <= pattern_size <= self.max_pattern_size):
                continue
                
            # Check if peaks are at similar levels
            peak1, peak2 = high[idx1], high[idx2]
            peak_diff = abs(peak1 - peak2) / max(peak1, peak2)
            
            if peak_diff > 0.05:  # Peaks should be within 5%
                continue
                
            # Find valley between peaks
            valley_idx = np.argmin(close[idx1:idx2 + 1]) + idx1
            valley = low[valley_idx]
            
            # Calculate confidence based on pattern quality
            avg_peak = (peak1 + peak2) / 2
            valley_depth = (avg_peak - valley) / avg_peak
            
            if valley_depth > 0.02:  # Minimum valley depth
                confidence = min(1.0, valley_depth * 5)
                price_target = valley - (avg_peak - valley)  # Measured move
                stop_loss = max(peak1, peak2) * 1.02
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.DOUBLE_TOP,
                    confidence=confidence,
                    start_index=idx1,
                    end_index=idx2,
                    price_target=price_target,
                    stop_loss=stop_loss,
                    metadata={
                        'peak1': float(peak1),
                        'peak2': float(peak2),
                        'valley': float(valley),
                        'valley_depth': float(valley_depth)
                    }
                ))
        
        # Detect double bottoms (inverse logic)
        for i in range(len(min_indices) - 1):
            idx1, idx2 = min_indices[i], min_indices[i + 1]
            pattern_size = idx2 - idx1
            
            if not (self.min_pattern_size <= pattern_size <= self.max_pattern_size):
                continue
                
            trough1, trough2 = low[idx1], low[idx2]
            trough_diff = abs(trough1 - trough2) / max(trough1, trough2)
            
            if trough_diff > 0.05:
                continue
                
            peak_idx = np.argmax(close[idx1:idx2 + 1]) + idx1
            peak = high[peak_idx]
            
            avg_trough = (trough1 + trough2) / 2
            peak_height = (peak - avg_trough) / avg_trough
            
            if peak_height > 0.02:
                confidence = min(1.0, peak_height * 5)
                price_target = peak + (peak - avg_trough)
                stop_loss = min(trough1, trough2) * 0.98
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    confidence=confidence,
                    start_index=idx1,
                    end_index=idx2,
                    price_target=price_target,
                    stop_loss=stop_loss,
                    metadata={
                        'trough1': float(trough1),
                        'trough2': float(trough2),
                        'peak': float(peak),
                        'peak_height': float(peak_height)
                    }
                ))
        
        return patterns

    def _detect_head_and_shoulders(
        self,
        data: pd.DataFrame
    ) -> List[PatternResult]:
        """
        Detect head and shoulders and inverse head and shoulders patterns.
        
        Head and Shoulders: Three peaks with middle being highest
        Inverse H&S: Three troughs with middle being lowest
        """
        patterns = []
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        max_indices, min_indices = self._find_local_extrema(close, order=3)
        
        # Detect head and shoulders (top)
        for i in range(len(max_indices) - 2):
            left_shoulder_idx = max_indices[i]
            head_idx = max_indices[i + 1]
            right_shoulder_idx = max_indices[i + 2]
            
            pattern_size = right_shoulder_idx - left_shoulder_idx
            if not (self.min_pattern_size <= pattern_size <= self.max_pattern_size):
                continue
            
            left_shoulder = high[left_shoulder_idx]
            head = high[head_idx]
            right_shoulder = high[right_shoulder_idx]
            
            # Head should be highest
            if not (head > left_shoulder and head > right_shoulder):
                continue
            
            # Shoulders should be at similar levels
            shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
            if shoulder_diff > 0.05:
                continue
            
            # Find neckline (valleys between peaks)
            left_valley_idx = np.argmin(close[left_shoulder_idx:head_idx + 1]) + left_shoulder_idx
            right_valley_idx = np.argmin(close[head_idx:right_shoulder_idx + 1]) + head_idx
            
            neckline_slope = (close[right_valley_idx] - close[left_valley_idx]) / (right_valley_idx - left_valley_idx)
            
            # Calculate confidence
            head_height = (head - max(left_shoulder, right_shoulder)) / max(left_shoulder, right_shoulder)
            confidence = min(1.0, head_height * 10)
            
            if confidence >= self.confidence_threshold:
                neckline_value = close[left_valley_idx] + neckline_slope * (right_shoulder_idx - left_valley_idx)
                price_target = neckline_value - (head - neckline_value)
                stop_loss = head * 1.02
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.HEAD_AND_SHOULDERS,
                    confidence=confidence,
                    start_index=left_shoulder_idx,
                    end_index=right_shoulder_idx,
                    price_target=price_target,
                    stop_loss=stop_loss,
                    metadata={
                        'left_shoulder': float(left_shoulder),
                        'head': float(head),
                        'right_shoulder': float(right_shoulder),
                        'neckline_slope': float(neckline_slope)
                    }
                ))
        
        # Detect inverse head and shoulders
        for i in range(len(min_indices) - 2):
            left_shoulder_idx = min_indices[i]
            head_idx = min_indices[i + 1]
            right_shoulder_idx = min_indices[i + 2]
            
            pattern_size = right_shoulder_idx - left_shoulder_idx
            if not (self.min_pattern_size <= pattern_size <= self.max_pattern_size):
                continue
            
            left_shoulder = low[left_shoulder_idx]
            head = low[head_idx]
            right_shoulder = low[right_shoulder_idx]
            
            if not (head < left_shoulder and head < right_shoulder):
                continue
            
            shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
            if shoulder_diff > 0.05:
                continue
            
            left_peak_idx = np.argmax(close[left_shoulder_idx:head_idx + 1]) + left_shoulder_idx
            right_peak_idx = np.argmax(close[head_idx:right_shoulder_idx + 1]) + head_idx
            
            neckline_slope = (close[right_peak_idx] - close[left_peak_idx]) / (right_peak_idx - left_peak_idx)
            
            head_depth = (max(left_shoulder, right_shoulder) - head) / max(left_shoulder, right_shoulder)
            confidence = min(1.0, head_depth * 10)
            
            if confidence >= self.confidence_threshold:
                neckline_value = close[left_peak_idx] + neckline_slope * (right_shoulder_idx - left_peak_idx)
                price_target = neckline_value + (neckline_value - head)
                stop_loss = head * 0.98
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                    confidence=confidence,
                    start_index=left_shoulder_idx,
                    end_index=right_shoulder_idx,
                    price_target=price_target,
                    stop_loss=stop_loss,
                    metadata={
                        'left_shoulder': float(left_shoulder),
                        'head': float(head),
                        'right_shoulder': float(right_shoulder),
                        'neckline_slope': float(neckline_slope)
                    }
                ))
        
        return patterns

    def _detect_flags_pennants(
        self,
        data: pd.DataFrame
    ) -> List[PatternResult]:
        """
        Detect flag and pennant patterns.
        
        Bull Flag: Sharp upward move followed by downward sloping consolidation
        Bear Flag: Sharp downward move followed by upward sloping consolidation
        Pennants: Similar but with converging trendlines
        """
        patterns = []
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        for i in range(self.min_pattern_size, len(data) - self.min_pattern_size):
            # Look for sharp move (flagpole)
            pole_start = i - self.min_pattern_size
            pole_end = i
            
            pole_move = (close[pole_end] - close[pole_start]) / close[pole_start]
            
            if abs(pole_move) < 0.05:  # Need at least 5% move
                continue
            
            # Analyze consolidation period
            consolidation = close[i:i + self.min_pattern_size]
            consolidation_slope = np.polyfit(
                range(len(consolidation)), 
                consolidation, 
                1
            )[0]
            
            # Detect flags (parallel channels)
            if abs(consolidation_slope) < 0.001:  # Sideways
                continue
            
            # Calculate channel boundaries
            consolidation_high = high[i:i + self.min_pattern_size]
            consolidation_low = low[i:i + self.min_pattern_size]
            
            high_slope = np.polyfit(
                range(len(consolidation_high)),
                consolidation_high,
                1
            )[0]
            low_slope = np.polyfit(
                range(len(consolidation_low)),
                consolidation_low,
                1
            )[0]
            
            # Check if channel is parallel (flag) or converging (pennant)
            slope_diff = abs(high_slope - low_slope)
            is_pennant = slope_diff > 0.001
            
            if pole_move > 0:  # Bullish
                if consolidation_slope < 0:  # Downward sloping consolidation
                    confidence = min(1.0, abs(pole_move) * 3)
                    pattern_type = PatternType.BULL_PENNANT if is_pennant else PatternType.BULL_FLAG
                    price_target = close[pole_end] + (close[pole_end] - close[pole_start])
                    stop_loss = min(consolidation_low) * 0.98
                    
                    patterns.append(PatternResult(
                        pattern_type=pattern_type,
                        confidence=confidence,
                        start_index=pole_start,
                        end_index=i + self.min_pattern_size,
                        price_target=price_target,
                        stop_loss=stop_loss,
                        metadata={
                            'pole_move': float(pole_move),
                            'consolidation_slope': float(consolidation_slope),
                            'is_pennant': is_pennant
                        }
                    ))
            else:  # Bearish
                if consolidation_slope > 0:  # Upward sloping consolidation
                    confidence = min(1.0, abs(pole_move) * 3)
                    pattern_type = PatternType.BEAR_PENNANT if is_pennant else PatternType.BEAR_FLAG
                    price_target = close[pole_end] - (close[pole_start] - close[pole_end])
                    stop_loss = max(consolidation_high) * 1.02
                    
                    patterns.append(PatternResult(
                        pattern_type=pattern_type,
                        confidence=confidence,
                        start_index=pole_start,
                        end_index=i + self.min_pattern_size,
                        price_target=price_target,
                        stop_loss=stop_loss,
                        metadata={
                            'pole_move': float(pole_move),
                            'consolidation_slope': float(consolidation_slope),
                            'is_pennant': is_pennant
                        }
                    ))
        
        return patterns

    def _detect_wedges(
        self,
        data: pd.DataFrame
    ) -> List[PatternResult]:
        """
        Detect rising and falling wedge patterns.
        
        Rising Wedge: Higher highs and higher lows with converging trendlines
        Falling Wedge: Lower highs and lower lows with converging trendlines
        """
        patterns = []
        high = data['high'].values
        low = data['low'].values
        
        for i in range(self.min_pattern_size, len(data) - self.min_pattern_size):
            segment_high = high[i - self.min_pattern_size:i + self.min_pattern_size]
            segment_low = low[i - self.min_pattern_size:i + self.min_pattern_size]
            
            x = np.arange(len(segment_high))
            
            # Fit trendlines
            high_slope, high_intercept, _, _, _ = linregress(x, segment_high)
            low_slope, low_intercept, _, _, _ = linregress(x, segment_low)
            
            # Check if trendlines are converging
            if high_slope <= low_slope:  # Not converging
                continue
            
            # Calculate convergence
            convergence = abs(high_slope - low_slope)
            
            if high_slope > 0 and low_slope > 0:  # Rising wedge
                confidence = min(1.0, convergence * 50)
                if confidence >= self.confidence_threshold:
                    price_target = segment_low[-1] - (segment_high[-1] - segment_low[-1])
                    stop_loss = segment_high[-1] * 1.02
                    
                    patterns.append(PatternResult(
                        pattern_type=PatternType.RISING_WEDGE,
                        confidence=confidence,
                        start_index=i - self.min_pattern_size,
                        end_index=i + self.min_pattern_size,
                        price_target=price_target,
                        stop_loss=stop_loss,
                        metadata={
                            'high_slope': float(high_slope),
                            'low_slope': float(low_slope),
                            'convergence': float(convergence)
                        }
                    ))
            elif high_slope < 0 and low_slope < 0:  # Falling wedge
                confidence = min(1.0, convergence * 50)
                if confidence >= self.confidence_threshold:
                    price_target = segment_high[-1] + (segment_high[-1] - segment_low[-1])
                    stop_loss = segment_low[-1] * 0.98
                    
                    patterns.append(PatternResult(
                        pattern_type=PatternType.FALLING_WEDGE,
                        confidence=confidence,
                        start_index=i - self.min_pattern_size,
                        end_index=i + self.min_pattern_size,
                        price_target=price_target,
                        stop_loss=stop_loss,
                        metadata={
                            'high_slope': float(high_slope),
                            'low_slope': float(low_slope),
                            'convergence': float(convergence)
                        }
                    ))
        
        return patterns

    def _detect_triangles(
        self,
        data: pd.DataFrame
    ) -> List[PatternResult]:
        """
        Detect triangle patterns (ascending, descending, symmetrical).
        
        Ascending Triangle: Flat top, rising bottom
        Descending Triangle: Falling top, flat bottom
        Symmetrical Triangle: Converging trendlines
        """
        patterns = []
        high = data['high'].values
        low = data['low'].values
        
        for i in range(self.min_pattern_size, len(data) - self.min_pattern_size):
            segment_high = high[i - self.min_pattern_size:i + self.min_pattern_size]
            segment_low = low[i - self.min_pattern_size:i + self.min_pattern_size]
            
            x = np.arange(len(segment_high))
            
            high_slope, high_intercept, _, _, _ = linregress(x, segment_high)
            low_slope, low_intercept, _, _, _ = linregress(x, segment_low)
            
            # Check convergence
            if high_slope >= low_slope:
                continue
            
            # Classify triangle type
            if abs(high_slope) < 0.001 and low_slope > 0:  # Ascending
                confidence = min(1.0, low_slope * 100)
                pattern_type = PatternType.ASCENDING_TRIANGLE
                price_target = segment_high[-1] + (segment_high[-1] - segment_low[-1])
                stop_loss = segment_low[-1] * 0.98
                
            elif high_slope < 0 and abs(low_slope) < 0.001:  # Descending
                confidence = min(1.0, abs(high_slope) * 100)
                pattern_type = PatternType.DESCENDING_TRIANGLE
                price_target = segment_low[-1] - (segment_high[-1] - segment_low[-1])
                stop_loss = segment_high[-1] * 1.02
                
            else:  # Symmetrical
                convergence = abs(high_slope - low_slope)
                confidence = min(1.0, convergence * 50)
                pattern_type = PatternType.SYMMETRICAL_TRIANGLE
                
                # Breakout direction based on prior trend
                prior_trend = (segment_high[-1] - segment_high[0]) / segment_high[0]
                if prior_trend > 0:  # Bullish bias
                    price_target = segment_high[-1] + (segment_high[-1] - segment_low[-1])
                    stop_loss = segment_low[-1] * 0.98
                else:
                    price_target = segment_low[-1] - (segment_high[-1] - segment_low[-1])
                    stop_loss = segment_high[-1] * 1.02
            
            if confidence >= self.confidence_threshold:
                patterns.append(PatternResult(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    start_index=i - self.min_pattern_size,
                    end_index=i + self.min_pattern_size,
                    price_target=price_target,
                    stop_loss=stop_loss,
                    metadata={
                        'high_slope': float(high_slope),
                        'low_slope': float(low_slope),
                        'prior_trend': float(prior_trend) if 'prior_trend' in locals() else 0.0
                    }
                ))
        
        return patterns

    def _detect_divergences(
        self,
        data: pd.DataFrame
    ) -> List[PatternResult]:
        """
        Detect regular and hidden divergences between price and RSI.
        
        Regular Bull Divergence: Price makes lower low, RSI makes higher low
        Regular Bear Divergence: Price makes higher high, RSI makes lower high
        Hidden Bull Divergence: Price makes higher low, RSI makes lower low
        Hidden Bear Divergence: Price makes lower high, RSI makes higher high
        """
        patterns = []
        close = data['close'].values
        
        # Calculate RSI
        rsi = self._calculate_rsi(close, period=14)
        
        # Find price and RSI extrema
        price_max, price_min = self._find_local_extrema(close)
        rsi_max, rsi_min = self._find_local_extrema(rsi)
        
        # Detect regular bear divergence
        for i in range(len(price_max) - 1):
            idx1, idx2 = price_max[i], price_max[i + 1]
            
            if idx2 - idx1 < self.min_pattern_size:
                continue
            
            # Price makes higher high
            if close[idx2] <= close[idx1]:
                continue
            
            # Find corresponding RSI peaks
            rsi_peaks = [p for p in rsi_max if idx1 <= p <= idx2]
            if len(rsi_peaks) < 2:
                continue
            
            rsi_peak1 = rsi[rsi_peaks[-2]]
            rsi_peak2 = rsi[rsi_peaks[-1]]
            
            # RSI makes lower high (regular bear divergence)
            if rsi_peak2 < rsi_peak1:
                confidence = min(1.0, (close[idx2] - close[idx1]) / close[idx1] * 10)
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.REGULAR_BEAR_DIVERGENCE,
                    confidence=confidence,
                    start_index=idx1,
                    end_index=idx2,
                    price_target=close[idx1] - (close[idx2] - close[idx1]),
                    stop_loss=close[idx2] * 1.02,
                    metadata={
                        'price_high1': float(close[idx1]),
                        'price_high2': float(close[idx2]),
                        'rsi_high1': float(rsi_peak1),
                        'rsi_high2': float(rsi_peak2)
                    }
                ))
        
        # Detect regular bull divergence
        for i in range(len(price_min) - 1):
            idx1, idx2 = price_min[i], price_min[i + 1]
            
            if idx2 - idx1 < self.min_pattern_size:
                continue
            
            # Price makes lower low
            if close[idx2] >= close[idx1]:
                continue
            
            # Find corresponding RSI troughs
            rsi_troughs = [p for p in rsi_min if idx1 <= p <= idx2]
            if len(rsi_troughs) < 2:
                continue
            
            rsi_trough1 = rsi[rsi_troughs[-2]]
            rsi_trough2 = rsi[rsi_troughs[-1]]
            
            # RSI makes higher low (regular bull divergence)
            if rsi_trough2 > rsi_trough1:
                confidence = min(1.0, (close[idx1] - close[idx2]) / close[idx1] * 10)
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.REGULAR_BULL_DIVERGENCE,
                    confidence=confidence,
                    start_index=idx1,
                    end_index=idx2,
                    price_target=close[idx2] + (close[idx1] - close[idx2]),
                    stop_loss=close[idx2] * 0.98,
                    metadata={
                        'price_low1': float(close[idx1]),
                        'price_low2': float(close[idx2]),
                        'rsi_low1': float(rsi_trough1),
                        'rsi_low2': float(rsi_trough2)
                    }
                ))
        
        # Detect hidden divergences
        # Hidden bear divergence: Lower high in price, higher high in RSI
        for i in range(len(price_max) - 1):
            idx1, idx2 = price_max[i], price_max[i + 1]
            
            if idx2 - idx1 < self.min_pattern_size:
                continue
            
            if close[idx2] >= close[idx1]:  # Not a lower high
                continue
            
            rsi_peaks = [p for p in rsi_max if idx1 <= p <= idx2]
            if len(rsi_peaks) < 2:
                continue
            
            rsi_peak1 = rsi[rsi_peaks[-2]]
            rsi_peak2 = rsi[rsi_peaks[-1]]
            
            if rsi_peak2 > rsi_peak1:  # Hidden bear divergence
                confidence = min(1.0, (close[idx1] - close[idx2]) / close[idx1] * 8)
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.HIDDEN_BEAR_DIVERGENCE,
                    confidence=confidence,
                    start_index=idx1,
                    end_index=idx2,
                    price_target=close[idx2] - (close[idx1] - close[idx2]),
                    stop_loss=close[idx1] * 1.02,
                    metadata={
                        'price_high1': float(close[idx1]),
                        'price_high2': float(close[idx2]),
                        'rsi_high1': float(rsi_peak1),
                        'rsi_high2': float(rsi_peak2)
                    }
                ))
        
        # Hidden bull divergence: Higher low in price, lower low in RSI
        for i in range(len(price_min) - 1):
            idx1, idx2 = price_min[i], price_min[i + 1]
            
            if idx2 - idx1 < self.min_pattern_size:
                continue
            
            if close[idx2] <= close[idx1]:  # Not a higher low
                continue
            
            rsi_troughs = [p for p in rsi_min if idx1 <= p <= idx2]
            if len(rsi_troughs) < 2:
                continue
            
            rsi_trough1 = rsi[rsi_troughs[-2]]
            rsi_trough2 = rsi[rsi_troughs[-1]]
            
            if rsi_trough2 < rsi_trough1:  # Hidden bull divergence
                confidence = min(1.0, (close[idx2] - close[idx1]) / close[idx1] * 8)
                
                patterns.append(PatternResult(
                    pattern_type=PatternType.HIDDEN_BULL_DIVERGENCE,
                    confidence=confidence,
                    start_index=idx1,
                    end_index=idx2,
                    price_target=close[idx2] + (close[idx2] - close[idx1]),
                    stop_loss=close[idx1] * 0.98,
                    metadata={
                        'price_low1': float(close[idx1]),
                        'price_low2': float(close[idx2]),
                        'rsi_low1': float(rsi_trough1),
                        'rsi_low2': float(rsi_trough2)
                    }
                ))
        
        return patterns

    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Array of price values
            period: RSI period (default 14)
            
        Returns:
            Array of RSI values
        """
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            rs = float('inf')
        else:
            rs = up / down
            
        rsi = np.zeros_like(prices)
        rsi[:period] = 100 - (100 / (1 + rs))
        
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rs = float('inf')
            else:
                rs = up / down
                
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi


def detect_patterns(
    ohlcv_data: pd.DataFrame,
    lookback_period: int = 100,
    min_confidence: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Convenience function to detect all chart patterns.
    
    Args:
        ohlcv_data: DataFrame with OHLCV data
        lookback_period: Number of candles to analyze
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detected patterns as dictionaries
    """
    detector = PatternDetector(
        lookback_period=lookback_period,
        confidence_threshold=min_confidence
    )
    
    patterns = detector.detect_all_patterns(ohlcv_data)
    return [p.to_dict() for p in patterns]


def get_pattern_summary(
    ohlcv_data: pd.DataFrame,
    lookback_period: int = 100
) -> Dict[str, Any]:
    """
    Generate summary of detected patterns with statistics.
    
    Args:
        ohlcv_data: DataFrame with OHLCV data
        lookback_period: Number of candles to analyze
        
    Returns:
        Dictionary with pattern summary statistics
    """
    patterns = detect_patterns(ohlcv_data, lookback_period)