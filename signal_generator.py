"""
Deterministic Signal Generator

This module generates trading signals based on clear, backtestable rules.
All logic is deterministic - same inputs produce same outputs every time.
"""

from typing import Dict, List, Optional
from loguru import logger
from strategy_config import StrategyConfig, SignalConfig


class TradingSignal:
    """Represents a trading signal"""

    def __init__(self,
                 action: str,
                 confidence: float,
                 reason: str,
                 entry_price: float,
                 stop_loss_pct: float,
                 take_profit_pct: float,
                 position_size_multiplier: float = 1.0,
                 metadata: Optional[Dict] = None):
        self.action = action  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'HOLD'
        self.confidence = confidence
        self.reason = reason
        self.entry_price = entry_price
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_multiplier = position_size_multiplier
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary (compatible with existing code)"""
        return {
            'action': self.action,
            'confidence': self.confidence,
            'reasoning': self.reason,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'position_size_multiplier': self.position_size_multiplier,
            'metadata': self.metadata
        }


class SignalGenerator:
    """
    Generates deterministic trading signals based on regime and technical indicators.

    This replaces the non-deterministic LLM decision-making with clear rules
    while preserving the ability to adapt strategy based on market regime.
    """

    def __init__(self, testnet: bool = True):
        self.strategy_config = StrategyConfig(testnet=testnet)
        logger.info("SignalGenerator initialized (deterministic mode)")

    def generate_signal(self,
                        market_data: Dict,
                        regime_data: Dict,
                        current_position: Optional[Dict] = None) -> TradingSignal:
        """
        Generate trading signal based on deterministic rules.

        Args:
            market_data: Market snapshot with indicators and orderbook
            regime_data: Regime classification from LLM
            current_position: Current position if any

        Returns:
            TradingSignal with action, confidence, and parameters
        """

        # If we have a position, check exit conditions first
        if current_position:
            exit_signal = self._check_exit_conditions(market_data, regime_data, current_position)
            if exit_signal:
                return exit_signal

        # No position - check entry conditions
        entry_signal = self._check_entry_conditions(market_data, regime_data)
        return entry_signal

    def _check_exit_conditions(self,
                                market_data: Dict,
                                regime_data: Dict,
                                position: Dict) -> Optional[TradingSignal]:
        """
        Check if current position should be closed based on deterministic rules.

        Exit triggers:
        1. Take profit / stop loss hit (handled by exchange)
        2. Regime change (momentum → ranging or vice versa)
        3. Trend reversal
        4. Technical exit signals
        """

        ind = market_data['indicators']
        current_side = position['side']

        # Check for regime change
        if self._should_exit_on_regime_change(regime_data, current_side):
            return TradingSignal(
                action='CLOSE',
                confidence=0.80,
                reason=f"Regime changed to {regime_data['regime']} / {regime_data['trend']}",
                entry_price=market_data['current_price'],
                stop_loss_pct=0,
                take_profit_pct=0,
                metadata={'exit_reason': 'REGIME_CHANGE'}
            )

        # Check for technical exit signals
        if current_side == 'LONG':
            # Exit long if:
            # - Price crosses below EMA20
            # - RSI drops below 40 (losing momentum)
            # - MACD crosses below signal
            if (ind['price_vs_ma20_pct'] < -0.5 or
                ind['rsi_14'] < 40 or
                ind['macd_histogram'] < -0.001):

                return TradingSignal(
                    action='CLOSE',
                    confidence=0.75,
                    reason="Technical exit: momentum lost",
                    entry_price=market_data['current_price'],
                    stop_loss_pct=0,
                    take_profit_pct=0,
                    metadata={'exit_reason': 'TECHNICAL'}
                )

        elif current_side == 'SHORT':
            # Exit short if:
            # - Price crosses above EMA20
            # - RSI rises above 60
            # - MACD crosses above signal
            if (ind['price_vs_ma20_pct'] > 0.5 or
                ind['rsi_14'] > 60 or
                ind['macd_histogram'] > 0.001):

                return TradingSignal(
                    action='CLOSE',
                    confidence=0.75,
                    reason="Technical exit: momentum lost",
                    entry_price=market_data['current_price'],
                    stop_loss_pct=0,
                    take_profit_pct=0,
                    metadata={'exit_reason': 'TECHNICAL'}
                )

        # Check for profit taking in ranging markets
        if regime_data['regime'] == 'RANGING':
            pnl_pct = position.get('pnl_pct', 0)

            # Take profits quickly in ranging markets (mean reversion)
            if pnl_pct > 1.5:  # 1.5% profit
                return TradingSignal(
                    action='CLOSE',
                    confidence=0.85,
                    reason=f"Ranging market profit taking: {pnl_pct:.2f}%",
                    entry_price=market_data['current_price'],
                    stop_loss_pct=0,
                    take_profit_pct=0,
                    metadata={'exit_reason': 'PROFIT_TARGET'}
                )

        # No exit signal
        return None

    def _should_exit_on_regime_change(self, regime_data: Dict, current_side: str) -> bool:
        """Check if regime change warrants position exit"""

        regime = regime_data['regime']
        trend = regime_data['trend']

        # Exit long if trend becomes bearish
        if current_side == 'LONG':
            if trend in ['WEAK_BEAR', 'STRONG_BEAR']:
                return True

        # Exit short if trend becomes bullish
        if current_side == 'SHORT':
            if trend in ['WEAK_BULL', 'STRONG_BULL']:
                return True

        # Exit if regime becomes transitional (unclear)
        if regime == 'TRANSITIONAL':
            return True

        return False

    def _check_entry_conditions(self,
                                 market_data: Dict,
                                 regime_data: Dict) -> TradingSignal:
        """
        Check entry conditions based on deterministic rules.

        Returns HOLD if no conditions met, or entry signal if conditions satisfied.
        """

        regime = regime_data['regime']
        trend = regime_data['trend']

        # Get strategy configuration for this regime/trend
        long_config = self.strategy_config.get_config(regime, trend, "LONG")
        short_config = self.strategy_config.get_config(regime, trend, "SHORT")

        # Check LONG conditions
        if long_config:
            long_signal = self._evaluate_signal(market_data, regime_data, long_config, "LONG")
            if long_signal:
                return long_signal

        # Check SHORT conditions
        if short_config:
            short_signal = self._evaluate_signal(market_data, regime_data, short_config, "SHORT")
            if short_signal:
                return short_signal

        # No entry conditions met - HOLD
        return TradingSignal(
            action='HOLD',
            confidence=0.50,
            reason=f"No entry conditions met ({regime} / {trend})",
            entry_price=market_data['current_price'],
            stop_loss_pct=2.0,
            take_profit_pct=3.0
        )

    def _evaluate_signal(self,
                         market_data: Dict,
                         regime_data: Dict,
                         config: SignalConfig,
                         side: str) -> Optional[TradingSignal]:
        """
        Evaluate if market conditions match signal configuration.

        This is the core deterministic logic that replaces LLM decision-making.
        """

        ind = market_data['indicators']
        ob = market_data['orderbook']

        # Track which conditions are met (for transparency)
        conditions_met = []
        conditions_failed = []

        # 1. RSI Check
        if config.rsi_min <= ind['rsi_14'] <= config.rsi_max:
            conditions_met.append(f"RSI {ind['rsi_14']:.1f} in range [{config.rsi_min}-{config.rsi_max}]")
        else:
            conditions_failed.append(f"RSI {ind['rsi_14']:.1f} outside [{config.rsi_min}-{config.rsi_max}]")
            return None

        # 2. Price vs EMA20
        if config.price_vs_ema20 != "ANY":
            if config.price_vs_ema20 == "ABOVE" and ind['price_vs_ma20_pct'] > 0:
                conditions_met.append(f"Price {ind['price_vs_ma20_pct']:.2f}% above EMA20")
            elif config.price_vs_ema20 == "BELOW" and ind['price_vs_ma20_pct'] < 0:
                conditions_met.append(f"Price {ind['price_vs_ma20_pct']:.2f}% below EMA20")
            else:
                conditions_failed.append(f"Price position vs EMA20 incorrect")
                return None

        # 3. EMA20 vs EMA50
        if config.ema20_vs_ema50 != "ANY":
            ema_diff_pct = ((ind['ma_20'] - ind['ma_50']) / ind['ma_50']) * 100

            if config.ema20_vs_ema50 == "ABOVE" and ema_diff_pct > 0:
                conditions_met.append(f"EMA20 {ema_diff_pct:.2f}% above EMA50")
            elif config.ema20_vs_ema50 == "BELOW" and ema_diff_pct < 0:
                conditions_met.append(f"EMA20 {ema_diff_pct:.2f}% below EMA50")
            else:
                conditions_failed.append(f"EMA alignment incorrect")
                return None

        # 4. MACD
        if config.macd_vs_signal != "ANY":
            if config.macd_vs_signal == "ABOVE" and ind['macd_histogram'] > 0:
                conditions_met.append(f"MACD bullish ({ind['macd_histogram']:.4f})")
            elif config.macd_vs_signal == "BELOW" and ind['macd_histogram'] < 0:
                conditions_met.append(f"MACD bearish ({ind['macd_histogram']:.4f})")
            else:
                conditions_failed.append(f"MACD signal incorrect")
                return None

        # 5. Volume
        if not (config.volume_ratio_min <= ind['volume_ratio'] <= config.volume_ratio_max):
            conditions_failed.append(f"Volume {ind['volume_ratio']:.2f}x outside [{config.volume_ratio_min}-{config.volume_ratio_max}]")
            return None
        else:
            conditions_met.append(f"Volume {ind['volume_ratio']:.2f}x in range")

        # 6. Bollinger Band Position
        if not (config.bb_position_min <= ind['bollinger_position'] <= config.bb_position_max):
            conditions_failed.append(f"BB position {ind['bollinger_position']:.2f} outside range")
            return None
        else:
            conditions_met.append(f"BB position {ind['bollinger_position']:.2%}")

        # ALL CONDITIONS MET - Generate signal
        entry_price = ob['best_ask'] if side == "LONG" else ob['best_bid']

        # Calculate stop loss and take profit based on ATR
        stop_loss_pct = config.stop_loss_atr_multiple * ind['atr_pct']
        take_profit_pct = config.take_profit_atr_multiple * ind['atr_pct']

        # Calculate confidence based on how strongly conditions are met
        confidence = self._calculate_confidence(ind, config, side)

        # Ensure confidence meets minimum threshold
        if confidence < config.min_confidence:
            logger.debug(f"{side} signal confidence {confidence:.2f} below threshold {config.min_confidence}")
            return None

        # Build reason string
        reason = f"{regime_data['regime']} {side}: {', '.join(conditions_met[:3])}"

        logger.info(f"✅ {side} SIGNAL GENERATED:")
        logger.info(f"   Regime: {regime_data['regime']} / {regime_data['trend']}")
        logger.info(f"   Confidence: {confidence:.2f} (min: {config.min_confidence})")
        logger.info(f"   Conditions: {' | '.join(conditions_met)}")
        logger.info(f"   TP: {take_profit_pct:.2f}%, SL: {stop_loss_pct:.2f}%")

        action = 'OPEN_LONG' if side == "LONG" else 'OPEN_SHORT'

        return TradingSignal(
            action=action,
            confidence=confidence,
            reason=reason,
            entry_price=entry_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            position_size_multiplier=config.position_size_multiplier,
            metadata={
                'regime': regime_data['regime'],
                'trend': regime_data['trend'],
                'conditions_met': conditions_met,
                'risk_mode': regime_data['risk_mode']
            }
        )

    def _calculate_confidence(self, ind: Dict, config: SignalConfig, side: str) -> float:
        """
        Calculate signal confidence based on strength of conditions.

        Confidence is deterministic but reflects how "strong" the signal is.
        """

        confidence = 0.50  # Base confidence

        # RSI strength
        if side == "LONG":
            # In momentum: Higher RSI = more confidence (trend following)
            # In ranging: Lower RSI = more confidence (mean reversion)
            if config.rsi_min < 50:  # Mean reversion setup
                # Lower RSI = higher confidence
                rsi_score = (50 - ind['rsi_14']) / 50
            else:  # Momentum setup
                # Higher RSI = higher confidence (but cap at 85)
                rsi_score = (ind['rsi_14'] - 50) / 35
        else:  # SHORT
            if config.rsi_max > 50:  # Mean reversion
                rsi_score = (ind['rsi_14'] - 50) / 50
            else:  # Momentum
                rsi_score = (50 - ind['rsi_14']) / 35

        confidence += max(0, min(0.15, rsi_score * 0.15))

        # Volume strength
        if ind['volume_ratio'] > 1.5:
            confidence += 0.10  # High volume = more confidence
        elif ind['volume_ratio'] > 1.0:
            confidence += 0.05

        # Trend strength
        trend_strength = abs(ind['price_vs_ma20_pct'])
        if trend_strength > 3.0:
            confidence += 0.10  # Strong trend = more confidence
        elif trend_strength > 1.5:
            confidence += 0.05

        # MACD strength
        macd_strength = abs(ind['macd_histogram'])
        if macd_strength > 0.01:
            confidence += 0.05

        # Volatility adjustment (lower confidence in high volatility)
        if ind['atr_pct'] > 3.0:
            confidence -= 0.05

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
