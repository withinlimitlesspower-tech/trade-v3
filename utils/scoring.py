```py
"""
Signal Scoring System for Crypto Trading Bot

Provides comprehensive signal scoring with weighted components:
- Trend Strength (25%)
- Momentum Quality (20%)
- Volume Confirmation (15%)
- S/R Proximity (15%)
- Pattern Quality (10%)
- Market Context (10%)
- Divergence Bonus (5%)

Each component is scored 0-100 and combined with configurable weights.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction enum"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Signal strength classification"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class ScoringConfig:
    """Configuration for scoring system weights and thresholds"""
    # Component weights (must sum to 1.0)
    trend_weight: float = 0.25
    momentum_weight: float = 0.20
    volume_weight: float = 0.15
    sr_proximity_weight: float = 0.15
    pattern_weight: float = 0.10
    market_context_weight: float = 0.10
    divergence_weight: float = 0.05

    # Scoring thresholds
    min_total_score: float = 30.0  # Minimum for valid signal
    strong_signal_threshold: float = 70.0
    very_strong_threshold: float = 85.0

    # Component-specific thresholds
    trend_min_strength: float = 20.0
    momentum_min_quality: float = 15.0
    volume_min_confirmation: float = 10.0

    def validate(self) -> bool:
        """Validate configuration weights sum to 1.0"""
        total = (
            self.trend_weight +
            self.momentum_weight +
            self.volume_weight +
            self.sr_proximity_weight +
            self.pattern_weight +
            self.market_context_weight +
            self.divergence_weight
        )
        return abs(total - 1.0) < 0.001


@dataclass
class ComponentScores:
    """Individual component scores"""
    trend_strength: float = 0.0
    momentum_quality: float = 0.0
    volume_confirmation: float = 0.0
    sr_proximity: float = 0.0
    pattern_quality: float = 0.0
    market_context: float = 0.0
    divergence_bonus: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "trend_strength": self.trend_strength,
            "momentum_quality": self.momentum_quality,
            "volume_confirmation": self.volume_confirmation,
            "sr_proximity": self.sr_proximity,
            "pattern_quality": self.pattern_quality,
            "market_context": self.market_context,
            "divergence_bonus": self.divergence_bonus
        }

    def validate(self) -> bool:
        """Validate all scores are in valid range 0-100"""
        scores = [
            self.trend_strength,
            self.momentum_quality,
            self.volume_confirmation,
            self.sr_proximity,
            self.pattern_quality,
            self.market_context,
            self.divergence_bonus
        ]
        return all(0 <= score <= 100 for score in scores)


@dataclass
class SignalScore:
    """Complete signal score with metadata"""
    total_score: float = 0.0
    direction: SignalDirection = SignalDirection.NEUTRAL
    strength: SignalStrength = SignalStrength.VERY_WEAK
    components: ComponentScores = field(default_factory=ComponentScores)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def classify_strength(self) -> SignalStrength:
        """Classify signal strength based on total score"""
        if self.total_score >= 85:
            return SignalStrength.VERY_STRONG
        elif self.total_score >= 70:
            return SignalStrength.STRONG
        elif self.total_score >= 50:
            return SignalStrength.MODERATE
        elif self.total_score >= 30:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_score": round(self.total_score, 2),
            "direction": self.direction.value,
            "strength": self.strength.value,
            "components": self.components.to_dict(),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class TrendScorer:
    """Scores trend strength based on multiple indicators"""

    @staticmethod
    def score_ema_alignment(
        price: float,
        ema_short: float,
        ema_medium: float,
        ema_long: float
    ) -> float:
        """
        Score based on EMA alignment and price position
        
        Args:
            price: Current price
            ema_short: Short-term EMA (e.g., 9)
            ema_medium: Medium-term EMA (e.g., 21)
            ema_long: Long-term EMA (e.g., 50)
        
        Returns:
            Score 0-100
        """
        try:
            score = 0.0
            
            # Check EMA order (bullish: short > medium > long)
            if ema_short > ema_medium > ema_long:
                score += 40
            elif ema_short < ema_medium < ema_long:
                score += 40  # Bearish alignment also valid
            elif ema_short > ema_long:
                score += 20
            else:
                score += 10
            
            # Price position relative to EMAs
            if price > ema_short:
                score += 20
            elif price > ema_medium:
                score += 15
            elif price > ema_long:
                score += 10
            else:
                score += 5
            
            # EMA spread (volatility-adjusted trend strength)
            spread = abs(ema_short - ema_long) / price * 100
            if spread > 2.0:
                score += 20
            elif spread > 1.0:
                score += 15
            elif spread > 0.5:
                score += 10
            else:
                score += 5
            
            # Distance from EMAs (trend conviction)
            dist_short = abs(price - ema_short) / price * 100
            if dist_short > 1.0:
                score += 20
            elif dist_short > 0.5:
                score += 15
            else:
                score += 10
            
            return min(100, max(0, score))
            
        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.error(f"Error scoring EMA alignment: {e}")
            return 50.0  # Neutral score on error

    @staticmethod
    def score_adx(adx_value: float) -> float:
        """
        Score based on ADX (Average Directional Index)
        
        Args:
            adx_value: ADX value (0-100)
        
        Returns:
            Score 0-100
        """
        try:
            if adx_value >= 50:
                return 90 + min(10, (adx_value - 50) * 0.5)
            elif adx_value >= 35:
                return 70 + (adx_value - 35) * 1.33
            elif adx_value >= 25:
                return 50 + (adx_value - 25) * 2.0
            elif adx_value >= 20:
                return 30 + (adx_value - 20) * 4.0
            else:
                return max(0, adx_value * 1.5)
                
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring ADX: {e}")
            return 50.0

    @staticmethod
    def score_macd(
        macd_line: float,
        signal_line: float,
        histogram: float
    ) -> float:
        """
        Score based on MACD
        
        Args:
            macd_line: MACD line value
            signal_line: Signal line value
            histogram: MACD histogram value
        
        Returns:
            Score 0-100
        """
        try:
            score = 0.0
            
            # MACD line vs signal line
            if macd_line > signal_line:
                score += 30
                if histogram > 0 and histogram > abs(histogram * 0.1):
                    score += 20  # Increasing momentum
            else:
                score += 10
            
            # Histogram strength
            hist_abs = abs(histogram)
            if hist_abs > 0:
                if hist_abs > 1.0:
                    score += 25
                elif hist_abs > 0.5:
                    score += 20
                elif hist_abs > 0.1:
                    score += 15
                else:
                    score += 10
            
            # Zero line cross
            if macd_line > 0 and signal_line > 0:
                score += 15
            elif macd_line < 0 and signal_line < 0:
                score += 5
            
            # Histogram slope (acceleration)
            # Note: This would need previous histogram value
            score += 10  # Base score for MACD presence
            
            return min(100, max(0, score))
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring MACD: {e}")
            return 50.0


class MomentumScorer:
    """Scores momentum quality"""

    @staticmethod
    def score_rsi(rsi_value: float, direction: SignalDirection) -> float:
        """
        Score based on RSI
        
        Args:
            rsi_value: RSI value (0-100)
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            if direction == SignalDirection.LONG:
                if 30 <= rsi_value <= 40:
                    return 90  # Oversold bounce zone
                elif 40 < rsi_value <= 50:
                    return 70  # Bullish momentum building
                elif 50 < rsi_value <= 60:
                    return 50  # Neutral bullish
                elif 60 < rsi_value <= 70:
                    return 30  # Getting overbought
                elif rsi_value > 70:
                    return 10  # Overbought
                else:  # rsi < 30
                    return 50  # Extreme oversold, risky
                    
            elif direction == SignalDirection.SHORT:
                if 60 <= rsi_value <= 70:
                    return 90  # Overbought rejection zone
                elif 50 <= rsi_value < 60:
                    return 70  # Bearish momentum building
                elif 40 <= rsi_value < 50:
                    return 50  # Neutral bearish
                elif 30 <= rsi_value < 40:
                    return 30  # Getting oversold
                elif rsi_value < 30:
                    return 10  # Oversold
                else:  # rsi > 70
                    return 50  # Extreme overbought, risky
                    
            else:  # NEUTRAL
                if 40 <= rsi_value <= 60:
                    return 70  # Neutral zone
                else:
                    return 30  # Directional bias
                    
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring RSI: {e}")
            return 50.0

    @staticmethod
    def score_stochastic(
        k_value: float,
        d_value: float,
        direction: SignalDirection
    ) -> float:
        """
        Score based on Stochastic Oscillator
        
        Args:
            k_value: %K value (0-100)
            d_value: %D value (0-100)
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            score = 0.0
            
            # Cross detection
            if k_value > d_value:
                score += 30
            else:
                score += 10
            
            # Level scoring based on direction
            if direction == SignalDirection.LONG:
                if k_value < 20 and d_value < 20:
                    score += 40  # Oversold
                elif k_value < 30:
                    score += 30
                elif k_value > 80:
                    score -= 20  # Overbought for long
                    
            elif direction == SignalDirection.SHORT:
                if k_value > 80 and d_value > 80:
                    score += 40  # Overbought
                elif k_value > 70:
                    score += 30
                elif k_value < 20:
                    score -= 20  # Oversold for short
            
            # Divergence potential (middle range)
            if 30 <= k_value <= 70:
                score += 20
            
            # Convergence of K and D
            spread = abs(k_value - d_value)
            if spread < 5:
                score += 10  # Tight, potential breakout
            
            return min(100, max(0, score))
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring Stochastic: {e}")
            return 50.0

    @staticmethod
    def score_cci(cci_value: float, direction: SignalDirection) -> float:
        """
        Score based on Commodity Channel Index
        
        Args:
            cci_value: CCI value
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            if direction == SignalDirection.LONG:
                if cci_value < -100:
                    return 90  # Oversold
                elif cci_value < -50:
                    return 70
                elif cci_value < 0:
                    return 50
                elif cci_value < 100:
                    return 30
                else:
                    return 10  # Overbought
                    
            elif direction == SignalDirection.SHORT:
                if cci_value > 100:
                    return 90  # Overbought
                elif cci_value > 50:
                    return 70
                elif cci_value > 0:
                    return 50
                elif cci_value > -100:
                    return 30
                else:
                    return 10  # Oversold
                    
            else:
                if -100 <= cci_value <= 100:
                    return 70
                else:
                    return 30
                    
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring CCI: {e}")
            return 50.0


class VolumeScorer:
    """Scores volume confirmation"""

    @staticmethod
    def score_volume_ratio(
        current_volume: float,
        avg_volume: float,
        price_change_pct: float
    ) -> float:
        """
        Score based on volume relative to average
        
        Args:
            current_volume: Current period volume
            avg_volume: Average volume (e.g., 20-period)
            price_change_pct: Price change percentage
        
        Returns:
            Score 0-100
        """
        try:
            if avg_volume <= 0:
                return 50.0
            
            volume_ratio = current_volume / avg_volume
            
            score = 0.0
            
            # Volume ratio scoring
            if volume_ratio >= 2.0:
                score += 40  # Very high volume
            elif volume_ratio >= 1.5:
                score += 30
            elif volume_ratio >= 1.2:
                score += 20
            elif volume_ratio >= 0.8:
                score += 15
            else:
                score += 5  # Low volume
            
            # Volume-price confirmation
            abs_price_change = abs(price_change_pct)
            if volume_ratio > 1.5 and abs_price_change > 1.0:
                score += 30  # Strong confirmation
            elif volume_ratio > 1.2 and abs_price_change > 0.5:
                score += 20
            elif volume_ratio > 1.0:
                score += 10
            
            # Volume trend (increasing vs decreasing)
            # Note: Would need previous volume ratio
            score += 15  # Base score
            
            # Volume stability
            if 0.8 <= volume_ratio <= 1.2:
                score += 15  # Stable volume
            elif volume_ratio > 2.0:
                score += 5  # Spike, less reliable
            
            return min(100, max(0, score))
            
        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.error(f"Error scoring volume ratio: {e}")
            return 50.0

    @staticmethod
    def score_obv(
        obv_value: float,
        prev_obv: float,
        price: float,
        prev_price: float
    ) -> float:
        """
        Score based on On-Balance Volume
        
        Args:
            obv_value: Current OBV
            prev_obv: Previous OBV
            price: Current price
            prev_price: Previous price
        
        Returns:
            Score 0-100
        """
        try:
            score = 0.0
            
            # OBV trend
            obv_change = obv_value - prev_obv
            price_change = price - prev_price
            
            # Confirmation
            if (obv_change > 0 and price_change > 0) or \
               (obv_change < 0 and price_change < 0):
                score += 50  # Strong confirmation
            elif obv_change > 0 and price_change < 0:
                score += 20  # Divergence (bullish)
            elif obv_change < 0 and price_change > 0:
                score += 20  # Divergence (bearish)
            else:
                score += 10  # No clear signal
            
            # OBV momentum
            obv_momentum = abs(obv_change) / max(abs(prev_obv), 1)
            if obv_momentum > 0.05:
                score += 25
            elif obv_momentum > 0.02:
                score += 20
            elif obv_momentum > 0.01:
                score += 15
            else:
                score += 10
            
            # OBV relative to its moving average
            # Note: Would need OBV MA
            score += 25  # Base score
            
            return min(100, max(0, score))
            
        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.error(f"Error scoring OBV: {e}")
            return 50.0

    @staticmethod
    def score_money_flow(mfi_value: float, direction: SignalDirection) -> float:
        """
        Score based on Money Flow Index
        
        Args:
            mfi_value: MFI value (0-100)
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            if direction == SignalDirection.LONG:
                if mfi_value < 20:
                    return 90  # Oversold with volume confirmation
                elif mfi_value < 30:
                    return 70
                elif mfi_value < 50:
                    return 50
                elif mfi_value < 80:
                    return 30
                else:
                    return 10  # Overbought
                    
            elif direction == SignalDirection.SHORT:
                if mfi_value > 80:
                    return 90  # Overbought with volume confirmation
                elif mfi_value > 70:
                    return 70
                elif mfi_value > 50:
                    return 50
                elif mfi_value > 20:
                    return 30
                else:
                    return 10  # Oversold
                    
            else:
                if 30 <= mfi_value <= 70:
                    return 70
                else:
                    return 30
                    
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring MFI: {e}")
            return 50.0


class SupportResistanceScorer:
    """Scores proximity to support/resistance levels"""

    @staticmethod
    def score_sr_proximity(
        price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        direction: SignalDirection
    ) -> float:
        """
        Score based on proximity to S/R levels
        
        Args:
            price: Current price
            support_levels: List of support levels
            resistance_levels: List of resistance levels
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            score = 0.0
            
            if not support_levels and not resistance_levels:
                return 50.0  # Neutral if no levels
            
            # Find nearest levels
            nearest_support = None
            nearest_resistance = None
            
            if support_levels:
                supports_below = [s for s in support_levels if s < price]
                if supports_below:
                    nearest_support = max(supports_below)
            
            if resistance_levels:
                resistances_above = [r for r in resistance_levels if r > price]
                if resistances_above:
                    nearest_resistance = min(resistances_above)
            
            if direction == SignalDirection.LONG:
                # Long: price near support is good
                if nearest_support:
                    dist_to_support = abs(price - nearest_support) / price * 100
                    if dist_to_support < 0.5:
                        score += 50  # Very close to support
                    elif dist_to_support < 1.0:
                        score += 40
                    elif dist_to_support < 2.0:
                        score += 30
                    elif dist_to_support < 3.0:
                        score += 20
                    else:
                        score += 10
                
                # Distance to resistance (room to run)
                if nearest_resistance:
                    dist_to_resistance = abs(nearest_resistance - price) / price * 100
                    if dist_to_resistance > 3.0:
                        score += 30  # Plenty of room
                    elif dist_to_resistance > 2.0:
                        score += 20
                    elif dist_to_resistance > 1.0:
                        score += 10
                    else:
                        score += 5  # Near resistance
                
                # Multiple support levels
                if support_levels:
                    nearby_supports = sum(1 for s in support_levels 
                                        if abs(price - s) / price * 100 < 2.0)
                    score += min(20, nearby_supports * 10)
                    
            elif direction == SignalDirection.SHORT:
                # Short: price near resistance is good
                if nearest_resistance:
                    dist_to_resistance = abs(nearest_resistance - price) / price * 100
                    if dist_to_resistance < 0.5:
                        score += 50
                    elif dist_to_resistance < 1.0:
                        score += 40
                    elif dist_to_resistance < 2.0:
                        score += 30
                    elif dist_to_resistance < 3.0:
                        score += 20
                    else:
                        score += 10
                
                # Distance to support (room to fall)
                if nearest_support:
                    dist_to_support = abs(price - nearest_support) / price * 100
                    if dist_to_support > 3.0:
                        score += 30
                    elif dist_to_support > 2.0:
                        score += 20
                    elif dist_to_support > 1.0:
                        score += 10
                    else:
                        score += 5
                
                # Multiple resistance levels
                if resistance_levels:
                    nearby_resistances = sum(1 for r in resistance_levels 
                                           if abs(price - r) / price * 100 < 2.0)
                    score += min(20, nearby_resistances * 10)
            
            else:  # NEUTRAL
                # In neutral zone (between S/R)
                if nearest_support and nearest_resistance:
                    mid_point = (nearest_support + nearest_resistance) / 2
                    dist_from_mid = abs(price - mid_point) / (nearest_resistance - nearest_support) * 100
                    if dist_from_mid < 20:
                        score += 70  # True neutral
                    else:
                        score += 40
                else:
                    score += 50
            
            return min(100, max(0, score))
            
        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.error(f"Error scoring S/R proximity: {e}")
            return 50.0

    @staticmethod
    def score_pivot_points(
        price: float,
        pivot: float,
        r1: float,
        r2: float,
        s1: float,
        s2: float,
        direction: SignalDirection
    ) -> float:
        """
        Score based on pivot points
        
        Args:
            price: Current price
            pivot: Pivot point
            r1, r2: Resistance levels
            s1, s2: Support levels
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            score = 0.0
            
            if direction == SignalDirection.LONG:
                if price <= s1:
                    score += 50  # Below first support
                    if price <= s2:
                        score += 30  # Extreme oversold
                elif price <= pivot:
                    score += 30
                elif price <= r1:
                    score += 15
                else:
                    score += 5
                    
            elif direction == SignalDirection.SHORT:
                if price >= r1:
                    score += 50  # Above first resistance
                    if price >= r2:
                        score += 30  # Extreme overbought
                elif price >= pivot:
                    score += 30
                elif price >= s1:
                    score += 15
                else:
                    score += 5
                    
            else:
                if s1 <= price <= r1:
                    score += 70
                else:
                    score += 30
            
            return min(100, max(0, score))
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring pivot points: {e}")
            return 50.0


class PatternScorer:
    """Scores chart pattern quality"""

    @staticmethod
    def score_candlestick_pattern(
        pattern_name: str,
        pattern_strength: float,
        direction: SignalDirection
    ) -> float:
        """
        Score based on candlestick pattern
        
        Args:
            pattern_name: Name of the pattern
            pattern_strength: Pattern strength (0-1)
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            # Base scores for common patterns
            pattern_scores = {
                # Bullish patterns
                "hammer": 70,
                "bullish_engulfing": 80,
                "morning_star": 85,
                "piercing_line": 75,
                "bullish_harami": 65,
                "three_white_soldiers": 85,
                "bullish_doji_star": 70,
                
                # Bearish patterns
                "shooting_star": 70,
                "bearish_engulfing": 80,
                "evening_star": 85,
                "dark_cloud_cover": 75,
                "bearish_harami": 65,
                "three_black_crows": 85,
                "bearish_doji_star": 70,
                
                # Neutral/indecision
                "doji": 50,
                "spinning_top": 40,
                "marubozu": 60
            }
            
            base_score = pattern_scores.get(pattern_name.lower(), 50)
            
            # Adjust for pattern strength
            adjusted_score = base_score * (0.5 + pattern_strength * 0.5)
            
            # Direction alignment
            bullish_patterns = {"hammer", "bullish_engulfing", "morning_star",
                              "piercing_line", "bullish_harami", "three_white_soldiers",
                              "bullish_doji_star"}
            bearish_patterns = {"shooting_star", "bearish_engulfing", "evening_star",
                              "dark_cloud_cover", "bearish_harami", "three_black_crows",
                              "bearish_doji_star"}
            
            pattern_direction = SignalDirection.NEUTRAL
            if pattern_name.lower() in bullish_patterns:
                pattern_direction = SignalDirection.LONG
            elif pattern_name.lower() in bearish_patterns:
                pattern_direction = SignalDirection.SHORT
            
            if pattern_direction == direction:
                adjusted_score *= 1.2  # Bonus for alignment
            elif direction != SignalDirection.NEUTRAL:
                adjusted_score *= 0.7  # Penalty for misalignment
            
            return min(100, max(0, adjusted_score))
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring candlestick pattern: {e}")
            return 50.0

    @staticmethod
    def score_chart_pattern(
        pattern_type: str,
        pattern_quality: float,
        breakout_confirmed: bool
    ) -> float:
        """
        Score based on chart pattern (e.g., head and shoulders, triangles)
        
        Args:
            pattern_type: Type of chart pattern
            pattern_quality: Pattern quality score (0-1)
            breakout_confirmed: Whether breakout is confirmed
        
        Returns:
            Score 0-100
        """
        try:
            # Base scores for common patterns
            pattern_scores = {
                "head_and_shoulders": 85,
                "inverse_head_and_shoulders": 85,
                "ascending_triangle": 75,
                "descending_triangle": 75,
                "symmetrical_triangle": 65,
                "double_top": 80,
                "double_bottom": 80,
                "triple_top": 85,
                "triple_bottom": 85,
                "flag": 70,
                "pennant": 70,
                "wedge": 60,
                "channel": 55,
                "rectangle": 60,
                "cup_and_handle": 80
            }
            
            base_score = pattern_scores.get(pattern_type.lower(), 50)
            
            # Adjust for pattern quality
            adjusted_score = base_score * (0.3 + pattern_quality * 0.7)
            
            # Breakout confirmation bonus
            if breakout_confirmed:
                adjusted_score *= 1.3
            
            return min(100, max(0, adjusted_score))
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring chart pattern: {e}")
            return 50.0


class MarketContextScorer:
    """Scores market context and conditions"""

    @staticmethod
    def score_market_trend(
        btc_trend: str,
        market_cap_trend: str,
        dominance_trend: str
    ) -> float:
        """
        Score based on overall market conditions
        
        Args:
            btc_trend: BTC trend direction ("bullish", "bearish", "neutral")
            market_cap_trend: Total market cap trend
            dominance_trend: BTC dominance trend
        
        Returns:
            Score 0-100
        """
        try:
            score = 0.0
            
            # BTC trend (highest weight)
            if btc_trend == "bullish":
                score += 40
            elif btc_trend == "neutral":
                score += 20
            else:
                score += 5
            
            # Market cap trend
            if market_cap_trend == "bullish":
                score += 30
            elif market_cap_trend == "neutral":
                score += 15
            else:
                score += 5
            
            # BTC dominance
            if dominance_trend == "falling":
                score += 20  # Alt season potential
            elif dominance_trend == "stable":
                score += 15
            else:
                score += 10  # BTC dominance rising, altcoins may suffer
            
            # Overall market health
            bullish_signals = sum(1 for t in [btc_trend, market_cap_trend] 
                                if t == "bullish")
            if bullish_signals >= 2:
                score += 10
            
            return min(100, max(0, score))
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring market trend: {e}")
            return 50.0

    @staticmethod
    def score_fear_greed_index(fgi_value: float) -> float:
        """
        Score based on Fear & Greed Index
        
        Args:
            fgi_value: Fear & Greed Index value (0-100)
        
        Returns:
            Score 0-100
        """
        try:
            # Extreme fear (0-25): Potential buying opportunity
            if fgi_value <= 25:
                return 80
            # Fear (25-45): Cautious buying
            elif fgi_value <= 45:
                return 60
            # Neutral (45-55): Normal conditions
            elif fgi_value <= 55:
                return 50
            # Greed (55-75): Caution for longs
            elif fgi_value <= 75:
                return 30
            # Extreme greed (75-100): Potential sell signal
            else:
                return 15
                
        except (TypeError, ValueError) as e:
            logger.error(f"Error scoring Fear & Greed Index: {e}")
            return 50.0

    @staticmethod
    def score_volatility(
        current_volatility: float,
        avg_volatility: float,
        direction: SignalDirection
    ) -> float:
        """
        Score based on market volatility
        
        Args:
            current_volatility: Current volatility measure
            avg_volatility: Average volatility
            direction: Expected signal direction
        
        Returns:
            Score 0-100
        """
        try:
            if avg_volatility <= 0:
                return 50.0
            
            vol_ratio = current_volatility / avg_volatility
            
            score = 0.0
            
            if direction == SignalDirection.LONG:
                # Low volatility is good for trend following
                if vol_ratio < 0.7:
                    score += 50  # Low vol, good for trend
                elif vol_ratio < 1.0:
                    score += 40
                elif vol_ratio < 1.3:
                    score += 25
                elif vol_ratio < 1.5:
                    score += 15
                else:
                    score += 5  # High volatility, risky
                    
            elif direction == SignalDirection.SHORT:
                # High volatility can be good for shorts
                if vol_ratio > 1.5:
                    score += 50
                elif vol_ratio > 1.3:
                    score += 40
                elif vol_ratio > 1.0:
                    score += 25
                elif vol_ratio > 0.7:
                    score += 15
                else:
                    score += 5
                    
            else:
                if 0.8 <=