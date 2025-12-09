"""
Strategy Configuration: Deterministic Rules per Market Regime

This module defines clear, backtestable entry/exit rules for each market regime.
All rules are deterministic and can be tested on historical data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications"""
    MOMENTUM = "MOMENTUM"
    RANGING = "RANGING"
    TRANSITIONAL = "TRANSITIONAL"


class TrendDirection(Enum):
    """Trend strength classifications"""
    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    NEUTRAL = "NEUTRAL"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"


class VolatilityState(Enum):
    """Volatility classifications"""
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class RiskMode(Enum):
    """Risk appetite levels"""
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"


@dataclass
class SignalConfig:
    """Configuration for a trading signal"""

    # Entry thresholds
    rsi_min: float
    rsi_max: float

    # Trend requirements
    price_vs_ema20: str  # "ABOVE", "BELOW", "ANY"
    ema20_vs_ema50: str  # "ABOVE", "BELOW", "ANY"

    # Momentum
    macd_vs_signal: str  # "ABOVE", "BELOW", "ANY"

    # Volume
    volume_ratio_min: float
    volume_ratio_max: float

    # Bollinger Bands
    bb_position_min: float  # 0.0 = lower band, 1.0 = upper band
    bb_position_max: float

    # Risk/Reward
    stop_loss_atr_multiple: float
    take_profit_atr_multiple: float

    # Position sizing
    position_size_multiplier: float  # 1.0 = full size, 0.5 = half size

    # Confidence threshold
    min_confidence: float


class StrategyConfig:
    """
    Deterministic trading rules for each market regime.

    These rules replace non-deterministic LLM decision-making with clear,
    backtestable logic while still using LLM for regime classification.
    """

    def __init__(self, testnet: bool = True):
        self.testnet = testnet

        # Volume thresholds adjust for testnet vs mainnet
        self.volume_threshold = 0.3 if testnet else 0.8

        # Strategy configurations
        self.configs = self._build_configs()

    def _build_configs(self) -> Dict:
        """Build all strategy configurations"""

        return {
            # ==================== MOMENTUM REGIME ====================
            "MOMENTUM": {
                "STRONG_BULL": {
                    "LONG": SignalConfig(
                        rsi_min=50,
                        rsi_max=85,
                        price_vs_ema20="ABOVE",
                        ema20_vs_ema50="ABOVE",
                        macd_vs_signal="ABOVE",
                        volume_ratio_min=self.volume_threshold,
                        volume_ratio_max=10.0,
                        bb_position_min=0.5,  # Above mid-band
                        bb_position_max=1.0,
                        stop_loss_atr_multiple=2.5,
                        take_profit_atr_multiple=4.0,
                        position_size_multiplier=1.0,  # Full size
                        min_confidence=0.55
                    ),
                    "SHORT": None  # Don't short in strong bull
                },

                "WEAK_BULL": {
                    "LONG": SignalConfig(
                        rsi_min=45,
                        rsi_max=80,
                        price_vs_ema20="ABOVE",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="ABOVE",
                        volume_ratio_min=self.volume_threshold,
                        volume_ratio_max=10.0,
                        bb_position_min=0.4,
                        bb_position_max=1.0,
                        stop_loss_atr_multiple=2.0,
                        take_profit_atr_multiple=3.0,
                        position_size_multiplier=0.75,  # Reduced size
                        min_confidence=0.60
                    ),
                    "SHORT": None
                },

                "STRONG_BEAR": {
                    "LONG": None,  # Don't long in strong bear
                    "SHORT": SignalConfig(
                        rsi_min=15,
                        rsi_max=50,
                        price_vs_ema20="BELOW",
                        ema20_vs_ema50="BELOW",
                        macd_vs_signal="BELOW",
                        volume_ratio_min=self.volume_threshold,
                        volume_ratio_max=10.0,
                        bb_position_min=0.0,
                        bb_position_max=0.5,  # Below mid-band
                        stop_loss_atr_multiple=2.5,
                        take_profit_atr_multiple=4.0,
                        position_size_multiplier=1.0,
                        min_confidence=0.55
                    )
                },

                "WEAK_BEAR": {
                    "LONG": None,
                    "SHORT": SignalConfig(
                        rsi_min=20,
                        rsi_max=55,
                        price_vs_ema20="BELOW",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="BELOW",
                        volume_ratio_min=self.volume_threshold,
                        volume_ratio_max=10.0,
                        bb_position_min=0.0,
                        bb_position_max=0.6,
                        stop_loss_atr_multiple=2.0,
                        take_profit_atr_multiple=3.0,
                        position_size_multiplier=0.75,
                        min_confidence=0.60
                    )
                },

                "NEUTRAL": None  # Don't trade neutral momentum (contradiction)
            },

            # ==================== RANGING REGIME ====================
            "RANGING": {
                "STRONG_BULL": None,  # Contradiction - shouldn't be both
                "WEAK_BULL": {
                    "LONG": SignalConfig(
                        rsi_min=0,
                        rsi_max=35,  # Oversold in range
                        price_vs_ema20="BELOW",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="ANY",
                        volume_ratio_min=0.0,
                        volume_ratio_max=0.7,  # Declining volume
                        bb_position_min=0.0,
                        bb_position_max=0.25,  # Near lower band
                        stop_loss_atr_multiple=1.5,
                        take_profit_atr_multiple=2.5,
                        position_size_multiplier=1.0,
                        min_confidence=0.60
                    ),
                    "SHORT": SignalConfig(
                        rsi_min=65,
                        rsi_max=100,  # Overbought in range
                        price_vs_ema20="ABOVE",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="ANY",
                        volume_ratio_min=0.0,
                        volume_ratio_max=0.7,
                        bb_position_min=0.75,  # Near upper band
                        bb_position_max=1.0,
                        stop_loss_atr_multiple=1.5,
                        take_profit_atr_multiple=2.5,
                        position_size_multiplier=1.0,
                        min_confidence=0.60
                    )
                },

                "NEUTRAL": {
                    "LONG": SignalConfig(
                        rsi_min=0,
                        rsi_max=30,
                        price_vs_ema20="ANY",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="ANY",
                        volume_ratio_min=0.0,
                        volume_ratio_max=0.7,
                        bb_position_min=0.0,
                        bb_position_max=0.20,
                        stop_loss_atr_multiple=1.5,
                        take_profit_atr_multiple=2.0,
                        position_size_multiplier=0.75,
                        min_confidence=0.65
                    ),
                    "SHORT": SignalConfig(
                        rsi_min=70,
                        rsi_max=100,
                        price_vs_ema20="ANY",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="ANY",
                        volume_ratio_min=0.0,
                        volume_ratio_max=0.7,
                        bb_position_min=0.80,
                        bb_position_max=1.0,
                        stop_loss_atr_multiple=1.5,
                        take_profit_atr_multiple=2.0,
                        position_size_multiplier=0.75,
                        min_confidence=0.65
                    )
                },

                "WEAK_BEAR": {
                    "LONG": SignalConfig(
                        rsi_min=0,
                        rsi_max=35,
                        price_vs_ema20="BELOW",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="ANY",
                        volume_ratio_min=0.0,
                        volume_ratio_max=0.7,
                        bb_position_min=0.0,
                        bb_position_max=0.25,
                        stop_loss_atr_multiple=1.5,
                        take_profit_atr_multiple=2.5,
                        position_size_multiplier=1.0,
                        min_confidence=0.60
                    ),
                    "SHORT": SignalConfig(
                        rsi_min=65,
                        rsi_max=100,
                        price_vs_ema20="ABOVE",
                        ema20_vs_ema50="ANY",
                        macd_vs_signal="ANY",
                        volume_ratio_min=0.0,
                        volume_ratio_max=0.7,
                        bb_position_min=0.75,
                        bb_position_max=1.0,
                        stop_loss_atr_multiple=1.5,
                        take_profit_atr_multiple=2.5,
                        position_size_multiplier=1.0,
                        min_confidence=0.60
                    )
                },

                "STRONG_BEAR": None
            },

            # ==================== TRANSITIONAL REGIME ====================
            "TRANSITIONAL": {
                # Very conservative - only take obvious setups
                "STRONG_BULL": {
                    "LONG": SignalConfig(
                        rsi_min=55,
                        rsi_max=80,
                        price_vs_ema20="ABOVE",
                        ema20_vs_ema50="ABOVE",
                        macd_vs_signal="ABOVE",
                        volume_ratio_min=1.5,  # Require very high volume
                        volume_ratio_max=10.0,
                        bb_position_min=0.6,
                        bb_position_max=1.0,
                        stop_loss_atr_multiple=2.0,
                        take_profit_atr_multiple=3.0,
                        position_size_multiplier=0.5,  # Half size
                        min_confidence=0.70
                    ),
                    "SHORT": None
                },

                "STRONG_BEAR": {
                    "LONG": None,
                    "SHORT": SignalConfig(
                        rsi_min=20,
                        rsi_max=45,
                        price_vs_ema20="BELOW",
                        ema20_vs_ema50="BELOW",
                        macd_vs_signal="BELOW",
                        volume_ratio_min=1.5,
                        volume_ratio_max=10.0,
                        bb_position_min=0.0,
                        bb_position_max=0.4,
                        stop_loss_atr_multiple=2.0,
                        take_profit_atr_multiple=3.0,
                        position_size_multiplier=0.5,
                        min_confidence=0.70
                    )
                },

                # Don't trade weak signals in transitional regime
                "WEAK_BULL": None,
                "WEAK_BEAR": None,
                "NEUTRAL": None
            }
        }

    def get_config(self, regime: str, trend: str, side: str) -> Optional[SignalConfig]:
        """
        Get signal configuration for specific market conditions.

        Args:
            regime: "MOMENTUM", "RANGING", "TRANSITIONAL"
            trend: "STRONG_BULL", "WEAK_BULL", "NEUTRAL", "WEAK_BEAR", "STRONG_BEAR"
            side: "LONG" or "SHORT"

        Returns:
            SignalConfig if configuration exists, None otherwise
        """
        try:
            regime_configs = self.configs.get(regime)
            if not regime_configs:
                return None

            trend_configs = regime_configs.get(trend)
            if not trend_configs:
                return None

            return trend_configs.get(side)
        except (KeyError, AttributeError):
            return None

    def get_all_configs_for_regime(self, regime: str) -> Dict:
        """Get all configurations for a specific regime"""
        return self.configs.get(regime, {})
