```py
"""
Technical Analysis Indicators Module
====================================

Professional-grade technical analysis indicator calculations for crypto trading.
Supports: RSI, MACD, EMA, Bollinger Bands, ATR, ADX, Ichimoku, Stochastic RSI,
OBV, Volume Profile, Fibonacci levels.

All functions accept pandas DataFrame or numpy arrays and return consistent,
validated results with proper error handling.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
from decimal import Decimal, ROUND_HALF_UP
import warnings
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ValidationError(Exception):
    """Custom exception for indicator validation errors."""
    pass


class IndicatorType(Enum):
    """Enumeration of supported indicator types."""
    RSI = "rsi"
    MACD = "macd"
    EMA = "ema"
    BOLLINGER = "bollinger"
    ATR = "atr"
    ADX = "adx"
    ICHIMOKU = "ichimoku"
    STOCH_RSI = "stoch_rsi"
    OBV = "obv"
    VOLUME_PROFILE = "volume_profile"
    FIBONACCI = "fibonacci"


@dataclass
class IndicatorResult:
    """Standardized result container for indicator calculations."""
    values: Union[np.ndarray, pd.Series, Dict]
    metadata: Dict
    success: bool = True
    error: Optional[str] = None


def validate_input(data: Union[pd.DataFrame, np.ndarray, pd.Series, List],
                   required_columns: Optional[List[str]] = None,
                   min_periods: int = 1) -> pd.DataFrame:
    """
    Validate and standardize input data for indicator calculations.

    Args:
        data: Input data (DataFrame, array, or list)
        required_columns: List of required column names for DataFrame input
        min_periods: Minimum number of data points required

    Returns:
        Standardized pandas DataFrame

    Raises:
        ValidationError: If input validation fails
    """
    if data is None or (isinstance(data, (list, np.ndarray)) and len(data) == 0):
        raise ValidationError("Input data cannot be None or empty")

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
    elif isinstance(data, (pd.Series, np.ndarray, list)):
        df = pd.DataFrame({'close': data}) if not isinstance(data, pd.DataFrame) else data
    else:
        raise ValidationError(f"Unsupported data type: {type(data)}")

    if len(df) < min_periods:
        raise ValidationError(
            f"Insufficient data points. Required: {min_periods}, Got: {len(df)}"
        )

    # Check for NaN/Inf values
    if df.isnull().values.any() or np.isinf(df.select_dtypes(include=[np.number]).values).any():
        raise ValidationError("Input data contains NaN or Inf values")

    return df


def validate_parameters(**kwargs) -> None:
    """
    Validate indicator parameters.

    Args:
        **kwargs: Parameter name-value pairs to validate

    Raises:
        ValidationError: If any parameter is invalid
    """
    validations = {
        'period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'fast_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'slow_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'signal_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'std_dev': (lambda x: isinstance(x, (int, float)) and x > 0, "must be positive number"),
        'multiplier': (lambda x: isinstance(x, (int, float)) and x > 0, "must be positive number"),
        'atr_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'di_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'adx_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'tenkan_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'kijun_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'senkou_span_b_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'displacement': (lambda x: isinstance(x, int) and x >= 0, "must be non-negative integer"),
        'rsi_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'stoch_k_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'stoch_d_period': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'smooth_k': (lambda x: isinstance(x, int) and x >= 1, "must be integer >= 1"),
        'smooth_d': (lambda x: isinstance(x, int) and x >= 1, "must be integer >= 1"),
        'lookback': (lambda x: isinstance(x, int) and x > 1, "must be integer > 1"),
        'num_std': (lambda x: isinstance(x, (int, float)) and x > 0, "must be positive number"),
    }

    for param_name, param_value in kwargs.items():
        if param_name in validations:
            validator, error_msg = validations[param_name]
            if not validator(param_value):
                raise ValidationError(f"Parameter '{param_name}' {error_msg}: got {param_value}")


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Input price array
        period: EMA period

    Returns:
        EMA values array
    """
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Args:
        data: Input price array
        period: SMA period

    Returns:
        SMA values array
    """
    sma = np.zeros_like(data)
    cumsum = np.cumsum(data)
    sma[:period - 1] = np.nan
    sma[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return sma


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calculate True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        True Range values
    """
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    return np.maximum(np.maximum(tr1, tr2), tr3)


def calculate_rsi(data: Union[pd.DataFrame, np.ndarray, pd.Series],
                  period: int = 14,
                  column: str = 'close') -> IndicatorResult:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data: Price data
        period: RSI period (default: 14)
        column: Column name for price data (default: 'close')

    Returns:
        IndicatorResult containing RSI values and metadata

    Reference:
        Wilder, J. Welles. "New Concepts in Technical Trading Systems."
    """
    try:
        validate_parameters(period=period)
        df = validate_input(data, required_columns=[column] if isinstance(data, pd.DataFrame) else None,
                           min_periods=period + 1)

        prices = df[column].values if isinstance(df, pd.DataFrame) else df.values.flatten()

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = _ema(gains, period)
        avg_loss = _ema(losses, period)

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))

        # Handle first period values
        rsi = np.concatenate([[np.nan], rsi])

        metadata = {
            'indicator': 'RSI',
            'period': period,
            'overbought': 70,
            'oversold': 30,
            'current_value': float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
            'is_overbought': rsi[-1] > 70 if not np.isnan(rsi[-1]) else False,
            'is_oversold': rsi[-1] < 30 if not np.isnan(rsi[-1]) else False
        }

        return IndicatorResult(values=rsi, metadata=metadata)

    except Exception as e:
        return IndicatorResult(values=np.array([]), metadata={}, success=False, error=str(e))


def calculate_macd(data: Union[pd.DataFrame, np.ndarray, pd.Series],
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9,
                   column: str = 'close') -> IndicatorResult:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        data: Price data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        column: Column name for price data (default: 'close')

    Returns:
        IndicatorResult containing MACD line, signal line, and histogram
    """
    try:
        validate_parameters(fast_period=fast_period, slow_period=slow_period,
                          signal_period=signal_period)
        df = validate_input(data, required_columns=[column] if isinstance(data, pd.DataFrame) else None,
                           min_periods=slow_period + signal_period)

        prices = df[column].values if isinstance(df, pd.DataFrame) else df.values.flatten()

        ema_fast = _ema(prices, fast_period)
        ema_slow = _ema(prices, slow_period)

        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        metadata = {
            'indicator': 'MACD',
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'current_macd': float(macd_line[-1]),
            'current_signal': float(signal_line[-1]),
            'current_histogram': float(histogram[-1]),
            'is_bullish': macd_line[-1] > signal_line[-1],
            'histogram_direction': 'increasing' if histogram[-1] > histogram[-2] else 'decreasing'
        }

        return IndicatorResult(
            values={
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            },
            metadata=metadata
        )

    except Exception as e:
        return IndicatorResult(values={}, metadata={}, success=False, error=str(e))


def calculate_ema(data: Union[pd.DataFrame, np.ndarray, pd.Series],
                  period: int = 20,
                  column: str = 'close') -> IndicatorResult:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price data
        period: EMA period (default: 20)
        column: Column name for price data (default: 'close')

    Returns:
        IndicatorResult containing EMA values
    """
    try:
        validate_parameters(period=period)
        df = validate_input(data, required_columns=[column] if isinstance(data, pd.DataFrame) else None,
                           min_periods=period)

        prices = df[column].values if isinstance(df, pd.DataFrame) else df.values.flatten()
        ema_values = _ema(prices, period)

        metadata = {
            'indicator': 'EMA',
            'period': period,
            'current_value': float(ema_values[-1]),
            'trend': 'upward' if ema_values[-1] > ema_values[-2] else 'downward'
        }

        return IndicatorResult(values=ema_values, metadata=metadata)

    except Exception as e:
        return IndicatorResult(values=np.array([]), metadata={}, success=False, error=str(e))


def calculate_bollinger_bands(data: Union[pd.DataFrame, np.ndarray, pd.Series],
                              period: int = 20,
                              num_std: float = 2.0,
                              column: str = 'close') -> IndicatorResult:
    """
    Calculate Bollinger Bands.

    Args:
        data: Price data
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        column: Column name for price data (default: 'close')

    Returns:
        IndicatorResult containing upper, middle, and lower bands
    """
    try:
        validate_parameters(period=period, num_std=num_std)
        df = validate_input(data, required_columns=[column] if isinstance(data, pd.DataFrame) else None,
                           min_periods=period)

        prices = df[column].values if isinstance(df, pd.DataFrame) else df.values.flatten()

        middle_band = _sma(prices, period)
        rolling_std = pd.Series(prices).rolling(window=period).std().values

        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)

        bandwidth = ((upper_band - lower_band) / middle_band) * 100
        percent_b = ((prices - lower_band) / (upper_band - lower_band)) * 100

        metadata = {
            'indicator': 'Bollinger Bands',
            'period': period,
            'num_std': num_std,
            'current_upper': float(upper_band[-1]),
            'current_middle': float(middle_band[-1]),
            'current_lower': float(lower_band[-1]),
            'bandwidth': float(bandwidth[-1]),
            'percent_b': float(percent_b[-1]),
            'position': 'above_upper' if prices[-1] > upper_band[-1]
                       else 'below_lower' if prices[-1] < lower_band[-1]
                       else 'within_bands'
        }

        return IndicatorResult(
            values={
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band,
                'bandwidth': bandwidth,
                'percent_b': percent_b
            },
            metadata=metadata
        )

    except Exception as e:
        return IndicatorResult(values={}, metadata={}, success=False, error=str(e))


def calculate_atr(data: pd.DataFrame,
                  period: int = 14,
                  high_col: str = 'high',
                  low_col: str = 'low',
                  close_col: str = 'close') -> IndicatorResult:
    """
    Calculate Average True Range (ATR).

    Args:
        data: OHLC DataFrame
        period: ATR period (default: 14)
        high_col: High price column name
        low_col: Low price column name
        close_col: Close price column name

    Returns:
        IndicatorResult containing ATR values
    """
    try:
        validate_parameters(period=period)
        df = validate_input(data, required_columns=[high_col, low_col, close_col],
                           min_periods=period + 1)

        high = df[high_col].values
        low = df[low_col].values
        close = df[close_col].values

        tr = _true_range(high, low, close)
        atr = _ema(tr, period)

        metadata = {
            'indicator': 'ATR',
            'period': period,
            'current_value': float(atr[-1]),
            'volatility': 'high' if atr[-1] > np.mean(atr[~np.isnan(atr)]) else 'low'
        }

        return IndicatorResult(values=atr, metadata=metadata)

    except Exception as e:
        return IndicatorResult(values=np.array([]), metadata={}, success=False, error=str(e))


def calculate_adx(data: pd.DataFrame,
                  period: int = 14,
                  high_col: str = 'high',
                  low_col: str = 'low',
                  close_col: str = 'close') -> IndicatorResult:
    """
    Calculate Average Directional Index (ADX).

    Args:
        data: OHLC DataFrame
        period: ADX period (default: 14)
        high_col: High price column name
        low_col: Low price column name
        close_col: Close price column name

    Returns:
        IndicatorResult containing ADX, +DI, and -DI values
    """
    try:
        validate_parameters(period=period)
        df = validate_input(data, required_columns=[high_col, low_col, close_col],
                           min_periods=period * 2)

        high = df[high_col].values
        low = df[low_col].values
        close = df[close_col].values

        # Calculate directional movements
        up_move = np.diff(high)
        down_move = np.diff(low)
        up_move = np.concatenate([[0], up_move])
        down_move = np.concatenate([[0], down_move])

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr = _true_range(high, low, close)

        # Smooth using EMA
        atr = _ema(tr, period)
        plus_di = 100 * _ema(plus_dm, period) / atr
        minus_di = 100 * _ema(minus_dm, period) / atr

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where(np.isnan(dx), 0, dx)
        adx = _ema(dx, period)

        metadata = {
            'indicator': 'ADX',
            'period': period,
            'current_adx': float(adx[-1]),
            'current_plus_di': float(plus_di[-1]),
            'current_minus_di': float(minus_di[-1]),
            'trend_strength': 'strong' if adx[-1] > 25 else 'weak',
            'trend_direction': 'bullish' if plus_di[-1] > minus_di[-1] else 'bearish'
        }

        return IndicatorResult(
            values={
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            },
            metadata=metadata
        )

    except Exception as e:
        return IndicatorResult(values={}, metadata={}, success=False, error=str(e))


def calculate_ichimoku(data: pd.DataFrame,
                       tenkan_period: int = 9,
                       kijun_period: int = 26,
                       senkou_span_b_period: int = 52,
                       displacement: int = 26,
                       high_col: str = 'high',
                       low_col: str = 'low',
                       close_col: str = 'close') -> IndicatorResult:
    """
    Calculate Ichimoku Cloud indicators.

    Args:
        data: OHLC DataFrame
        tenkan_period: Tenkan-sen (conversion line) period (default: 9)
        kijun_period: Kijun-sen (base line) period (default: 26)
        senkou_span_b_period: Senkou Span B period (default: 52)
        displacement: Displacement period (default: 26)
        high_col: High price column name
        low_col: Low price column name
        close_col: Close price column name

    Returns:
        IndicatorResult containing all Ichimoku components
    """
    try:
        validate_parameters(tenkan_period=tenkan_period, kijun_period=kijun_period,
                          senkou_span_b_period=senkou_span_b_period,
                          displacement=displacement)
        df = validate_input(data, required_columns=[high_col, low_col, close_col],
                           min_periods=senkou_span_b_period + displacement)

        high = df[high_col].values
        low = df[low_col].values
        close = df[close_col].values

        def _rolling_max_min(high, low, period):
            """Calculate rolling max of high and min of low."""
            max_high = pd.Series(high).rolling(window=period).max().values
            min_low = pd.Series(low).rolling(window=period).min().values
            return (max_high + min_low) / 2

        tenkan_sen = _rolling_max_min(high, low, tenkan_period)
        kijun_sen = _rolling_max_min(high, low, kijun_period)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = _rolling_max_min(high, low, senkou_span_b_period)

        # Shift forward
        chikou_span = np.full_like(close, np.nan)
        chikou_span[:-displacement] = close[displacement:]

        # Create cloud values (shifted forward)
        cloud_a = np.full_like(close, np.nan)
        cloud_b = np.full_like(close, np.nan)
        cloud_a[displacement:] = senkou_span_a[:-displacement]
        cloud_b[displacement:] = senkou_span_b[:-displacement]

        # Determine cloud color and position
        current_price = close[-1]
        current_cloud_a = cloud_a[-1]
        current_cloud_b = cloud_b[-1]

        if not np.isnan(current_cloud_a) and not np.isnan(current_cloud_b):
            cloud_top = max(current_cloud_a, current_cloud_b)
            cloud_bottom = min(current_cloud_a, current_cloud_b)
            cloud_color = 'green' if current_cloud_a > current_cloud_b else 'red'
            price_position = 'above_cloud' if current_price > cloud_top \
                           else 'below_cloud' if current_price < cloud_bottom \
                           else 'within_cloud'
        else:
            cloud_color = 'unknown'
            price_position = 'unknown'

        metadata = {
            'indicator': 'Ichimoku Cloud',
            'tenkan_period': tenkan_period,
            'kijun_period': kijun_period,
            'senkou_span_b_period': senkou_span_b_period,
            'displacement': displacement,
            'current_tenkan': float(tenkan_sen[-1]),
            'current_kijun': float(kijun_sen[-1]),
            'current_senkou_a': float(senkou_span_a[-1]),
            'current_senkou_b': float(senkou_span_b[-1]),
            'cloud_color': cloud_color,
            'price_position': price_position,
            'is_bullish': tenkan_sen[-1] > kijun_sen[-1] if not np.isnan(tenkan_sen[-1]) else False
        }

        return IndicatorResult(
            values={
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span,
                'cloud_a': cloud_a,
                'cloud_b': cloud_b
            },
            metadata=metadata
        )

    except Exception as e:
        return IndicatorResult(values={}, metadata={}, success=False, error=str(e))


def calculate_stochastic_rsi(data: Union[pd.DataFrame, np.ndarray, pd.Series],
                             rsi_period: int = 14,
                             stoch_k_period: int = 3,
                             stoch_d_period: int = 3,
                             smooth_k: int = 3,
                             smooth_d: int = 3,
                             column: str = 'close') -> IndicatorResult:
    """
    Calculate Stochastic RSI.

    Args:
        data: Price data
        rsi_period: RSI period (default: 14)
        stoch_k_period: Stochastic %K period (default: 3)
        stoch_d_period: Stochastic %D period (default: 3)
        smooth_k: %K smoothing period (default: 3)
        smooth_d: %D smoothing period (default: 3)
        column: Column name for price data (default: 'close')

    Returns:
        IndicatorResult containing Stochastic RSI values
    """
    try:
        validate_parameters(rsi_period=rsi_period, stoch_k_period=stoch_k_period,
                          stoch_d_period=stoch_d_period, smooth_k=smooth_k,
                          smooth_d=smooth_d)
        df = validate_input(data, required_columns=[column] if isinstance(data, pd.DataFrame) else None,
                           min_periods=rsi_period + stoch_k_period + stoch_d_period)

        prices = df[column].values if isinstance(df, pd.DataFrame) else df.values.flatten()

        # Calculate RSI first
        rsi_result = calculate_rsi(prices, period=rsi_period)
        if not rsi_result.success:
            raise ValueError("Failed to calculate RSI for Stochastic RSI")

        rsi_values = rsi_result.values

        # Calculate Stochastic of RSI
        stoch_rsi = np.full_like(rsi_values, np.nan)
        for i in range(rsi_period, len(rsi_values)):
            if not np.isnan(rsi_values[i]):
                rsi_window = rsi_values[max(0, i - stoch_k_period + 1):i + 1]
                rsi_window = rsi_window[~np.isnan(rsi_window)]
                if len(rsi_window) > 0:
                    min_rsi = np.min(rsi_window)
                    max_rsi = np.max(rsi_window)
                    if max_rsi != min_rsi:
                        stoch_rsi[i] = (rsi_values[i] - min_rsi) / (max_rsi - min_rsi) * 100

        # Smooth %K and %D
        k_line = _ema(np.nan_to_num(stoch_rsi), smooth_k)
        d_line = _ema(k_line, smooth_d)

        metadata = {
            'indicator': 'Stochastic RSI',
            'rsi_period': rsi_period,
            'k_period': stoch_k_period,
            'd_period': stoch_d_period,
            'current_k': float(k_line[-1]),
            'current_d': float(d_line[-1]),
            'is_overbought': k_line[-1] > 80,
            'is_oversold': k_line[-1] < 20,
            'crossover': 'bullish' if k_line[-1] > d_line[-1] and k_line[-2] <= d_line[-2]
                        else 'bearish' if k_line[-1] < d_line[-1] and k_line[-2] >= d_line[-2]
                        else 'none'
        }

        return IndicatorResult(
            values={
                'stoch_rsi': stoch_rsi,
                'k_line': k_line,
                'd_line': d_line
            },
            metadata=metadata
        )

    except Exception as e:
        return IndicatorResult(values={}, metadata={}, success=False, error=str(e))


def calculate_obv(data: pd.DataFrame,
                  close_col: str = 'close',
                  volume_col: str = 'volume') -> IndicatorResult:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        data: OHLCV DataFrame
        close_col: Close price column name
        volume_col: Volume column name

    Returns:
        IndicatorResult containing OBV values
    """
    try:
        df = validate_input(data, required_columns=[close_col, volume_col])

        close = df[close_col].values
        volume = df[volume_col].values

        obv = np.zeros_like(volume)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # Calculate OBV trend
        obv_sma = _sma(obv, 20)
        obv_trend = 'bullish' if obv[-1] > obv_sma[-1] else 'bearish'

        metadata = {
            'indicator': 'OBV',
            'current_value': float(obv[-1]),
            'trend': obv_trend,
            'divergence': 'positive' if close[-1] < close[-20] and obv[-1] > obv[-20]
                         else 'negative' if close[-1] > close[-20] and obv[-1] < obv[-20]
                         else 'none'
        }

        return IndicatorResult(values=obv, metadata=metadata)

    except Exception as e:
        return IndicatorResult(values=np.array([]), metadata={}, success=False, error=str(e))


def calculate_volume_profile(data: pd.DataFrame,
                             num_bins: int = 24,
                             price_col: str = 'close',
                             volume_col: str = 'volume',
                             high_col: str = 'high',
                             low_col: str = 'low') -> IndicatorResult:
    """
    Calculate Volume Profile (Market Profile).

    Args:
        data: OHLCV DataFrame
        num_bins: Number of price bins (default: 24)
        price_col: Close price column name
        volume_col: Volume column name
        high_col: High price column name
        low_col: Low price column name

    Returns:
        IndicatorResult containing volume profile data
    """
    try:
        df = validate_input(data, required_columns=[price_col, volume_col, high_col, low_col],
                           min_periods=num_bins)

        high = df[high_col].values
        low = df[low_col].values
        close = df[price_col].values
        volume = df[volume_col].values

        # Determine price range
        price_min = np.min(low)
        price_max = np.max(high)
        bin_size = (price_max - price_min) / num_bins

        if bin_size == 0:
            raise ValidationError("Price range is zero, cannot create volume profile")

        # Create price bins
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate volume for each bin
        volume_profile = np.zeros(num_bins)
        for i in range(len(df)):
            bin_idx = np.digitize(close[i], bins) - 1
            if 0 <= bin_idx < num_bins:
                volume_profile[bin_idx] += volume[i]

        # Find Point of Control (POC) - highest volume node
        poc_idx = np.argmax(volume_profile)
        poc_price = bin_centers[poc_idx]
        poc_volume = volume_profile[poc_idx]

        # Calculate Value Area (70% of total volume)
        total_volume = np.sum(volume_profile)
        sorted_idx = np.argsort(volume_profile)[::-1]
        value_area_volume = 0
        value_area_idx = []

        for idx in sorted_idx:
            if value_area_volume / total_volume < 0.70:
                value_area_volume += volume_profile[idx]
                value_area_idx.append(idx)
            else:
                break

        value_area_low = bin_centers[min(value_area_idx)] if value_area_idx else price_min
        value_area_high = bin_centers[max(value_area_idx)] if value_area_idx else price_max

        metadata = {
            'indicator': 'Volume Profile',
            'num_bins': num_bins,
            'poc_price': float(poc_price),
            'poc_volume': float(poc_volume),
            'value_area_low': float(value_area_low),
            'value_area_high': float(value_area_high),
            'current_price_position': 'above_va' if close[-1] > value_area_high
                                    else 'below_va' if close[-1] < value_area_low
                                    else 'within_va'
        }

        return IndicatorResult(
            values={
                'bins': bin_centers,
                'volume': volume_profile,
                'poc_price': poc_price,
                'value_area_low': value_area_low,
                'value_area_high': value_area_high
            },
            metadata=metadata
        )

    except Exception as e:
        return IndicatorResult(values={}, metadata={}, success=False, error=str(e))


def calculate_fibonacci_levels(data: Union[pd.DataFrame, np.ndarray, pd.Series],
                               lookback: int = 100,
                               column: str = 'close',
                               high_col: Optional[str] = None,
                               low_col: Optional[str] = None) -> IndicatorResult:
    """
    Calculate Fibonacci retracement and extension levels.

    Args:
        data: Price data
        lookback: Lookback period for high/low calculation (default: 100)
        column: Close price column name
        high_col: High price column name (optional, for better accuracy)
        low_col: Low price column name (optional, for better accuracy)

    Returns:
        IndicatorResult containing Fibonacci levels
    """
    try:
        validate_parameters(lookback=lookback)

        if isinstance(data, pd.DataFrame) and high_col and low_col:
            df = validate_input(data, required_columns=[column, high_col, low_col],
                               min_periods=lookback)
            high = df[high_col].values[-lookback:]
            low = df[low_col].values[-lookback:]
            close = df[column].values[-lookback:]
        else