"""
Backtesting Engine

Simulates strategy on historical data to validate performance before risking real capital.
This is essential for hedge-fund quality validation.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

from signal_generator import SignalGenerator
from performance_metrics import PerformanceMetrics
from strategy_config import StrategyConfig


class BacktestEngine:
    """
    Backtest trading strategy on historical data.

    This engine replays historical market data and simulates trades
    using the deterministic signal generator.
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 position_size_pct: float = 10.0,
                 fee_pct: float = 0.05,
                 slippage_pct: float = 0.1):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital in USD
            position_size_pct: Position size as % of capital
            fee_pct: Trading fees (%)
            slippage_pct: Expected slippage (%)
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        self.signal_generator = SignalGenerator(testnet=False)  # Use mainnet rules
        self.performance = PerformanceMetrics(save_path="data/backtest_results.json")

        # State
        self.capital = initial_capital
        self.position: Optional[Dict] = None
        self.trades: List[Dict] = []

        logger.info(f"BacktestEngine initialized: ${initial_capital} capital, {position_size_pct}% position size")

    def run(self,
            historical_data: List[Dict],
            regime_classifications: Optional[List[Dict]] = None) -> Dict:
        """
        Run backtest on historical data.

        Args:
            historical_data: List of market snapshots with indicators
            regime_classifications: Optional pre-classified regimes (or will use heuristics)

        Returns:
            Backtest results with performance metrics
        """

        logger.info(f"Starting backtest: {len(historical_data)} data points")

        self._reset_state()

        for i, market_data in enumerate(historical_data):
            # Get regime classification
            if regime_classifications and i < len(regime_classifications):
                regime_data = regime_classifications[i]
            else:
                # Use heuristic regime classification
                regime_data = self._classify_regime_heuristic(market_data)

            # Record equity
            self.performance.record_equity(self.capital, market_data['timestamp'])

            # Check if we have a position
            if self.position:
                # Update position P&L
                self._update_position(market_data)

                # Check exit conditions
                exit_signal = self.signal_generator._check_exit_conditions(
                    market_data, regime_data, self.position
                )

                if exit_signal:
                    self._close_position(market_data, exit_signal.reason)
                    continue

                # Check if TP/SL hit
                if self._check_tp_sl(market_data):
                    continue

            # No position - check entry
            if not self.position:
                entry_signal = self.signal_generator._check_entry_conditions(
                    market_data, regime_data
                )

                if entry_signal.action in ['OPEN_LONG', 'OPEN_SHORT']:
                    self._open_position(market_data, entry_signal, regime_data)

        # Close any remaining position
        if self.position and historical_data:
            self._close_position(historical_data[-1], "BACKTEST_END")

        # Calculate final metrics
        results = self._generate_results()

        logger.info(f"Backtest complete: {len(self.trades)} trades, Final capital: ${self.capital:.2f}")

        return results

    def _reset_state(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.performance = PerformanceMetrics(save_path="data/backtest_results.json")

    def _classify_regime_heuristic(self, market_data: Dict) -> Dict:
        """
        Simple heuristic regime classification (when LLM not available).

        Uses the same logic as MarketDataCollector's regime detection.
        """
        ind = market_data['indicators']

        # Use existing regime classification if available
        if 'final_regime' in ind:
            regime = ind['final_regime']
        elif 'market_regime' in ind:
            regime = ind['market_regime']
        else:
            regime = "RANGING"

        # Classify trend
        if ind['trend'] == "BULLISH" and abs(ind['price_vs_ma20_pct']) > 2.0:
            trend = "STRONG_BULL"
        elif ind['trend'] == "BULLISH":
            trend = "WEAK_BULL"
        elif ind['trend'] == "BEARISH" and abs(ind['price_vs_ma20_pct']) > 2.0:
            trend = "STRONG_BEAR"
        elif ind['trend'] == "BEARISH":
            trend = "WEAK_BEAR"
        else:
            trend = "NEUTRAL"

        # Volatility
        if ind['atr_pct'] > 2.5:
            volatility = "HIGH"
        elif ind['atr_pct'] < 1.0:
            volatility = "LOW"
        else:
            volatility = "NORMAL"

        return {
            'regime': regime,
            'trend': trend,
            'volatility': volatility,
            'risk_mode': 'NORMAL',
            'confidence': 0.70,
            'reasoning': 'Heuristic classification'
        }

    def _open_position(self, market_data: Dict, signal, regime_data: Dict):
        """Open new position"""

        side = 'LONG' if signal.action == 'OPEN_LONG' else 'SHORT'

        # Calculate position size
        position_size_usd = self.capital * (self.position_size_pct / 100) * signal.position_size_multiplier
        entry_price = signal.entry_price

        # Apply slippage
        if side == 'LONG':
            entry_price *= (1 + self.slippage_pct / 100)
        else:
            entry_price *= (1 - self.slippage_pct / 100)

        size_coins = position_size_usd / entry_price

        # Deduct fees
        fee = position_size_usd * (self.fee_pct / 100)
        self.capital -= fee

        # Calculate TP/SL prices
        if side == 'LONG':
            tp_price = entry_price * (1 + signal.take_profit_pct / 100)
            sl_price = entry_price * (1 - signal.stop_loss_pct / 100)
        else:
            tp_price = entry_price * (1 - signal.take_profit_pct / 100)
            sl_price = entry_price * (1 + signal.stop_loss_pct / 100)

        self.position = {
            'side': side,
            'entry_price': entry_price,
            'size': size_coins,
            'entry_time': market_data['timestamp'],
            'tp_price': tp_price,
            'sl_price': sl_price,
            'regime': regime_data['regime'],
            'trend': regime_data['trend'],
            'current_price': entry_price,
            'pnl_usd': 0,
            'pnl_pct': 0
        }

        logger.debug(f"Opened {side} position: {size_coins:.4f} @ ${entry_price:.2f}, TP: ${tp_price:.2f}, SL: ${sl_price:.2f}")

    def _update_position(self, market_data: Dict):
        """Update position with current price"""
        if not self.position:
            return

        current_price = market_data['current_price']
        self.position['current_price'] = current_price

        # Calculate P&L
        if self.position['side'] == 'LONG':
            pnl_usd = (current_price - self.position['entry_price']) * self.position['size']
            pnl_pct = ((current_price - self.position['entry_price']) / self.position['entry_price']) * 100
        else:  # SHORT
            pnl_usd = (self.position['entry_price'] - current_price) * self.position['size']
            pnl_pct = ((self.position['entry_price'] - current_price) / self.position['entry_price']) * 100

        self.position['pnl_usd'] = pnl_usd
        self.position['pnl_pct'] = pnl_pct

    def _check_tp_sl(self, market_data: Dict) -> bool:
        """Check if TP or SL hit"""
        if not self.position:
            return False

        current_price = market_data['current_price']

        if self.position['side'] == 'LONG':
            # Check TP
            if current_price >= self.position['tp_price']:
                self._close_position(market_data, "TP_HIT")
                return True
            # Check SL
            if current_price <= self.position['sl_price']:
                self._close_position(market_data, "SL_HIT")
                return True

        else:  # SHORT
            # Check TP
            if current_price <= self.position['tp_price']:
                self._close_position(market_data, "TP_HIT")
                return True
            # Check SL
            if current_price >= self.position['sl_price']:
                self._close_position(market_data, "SL_HIT")
                return True

        return False

    def _close_position(self, market_data: Dict, reason: str):
        """Close position"""
        if not self.position:
            return

        exit_price = market_data['current_price']

        # Apply slippage
        if self.position['side'] == 'LONG':
            exit_price *= (1 - self.slippage_pct / 100)
        else:
            exit_price *= (1 + self.slippage_pct / 100)

        # Calculate final P&L
        if self.position['side'] == 'LONG':
            pnl_usd = (exit_price - self.position['entry_price']) * self.position['size']
        else:
            pnl_usd = (self.position['entry_price'] - exit_price) * self.position['size']

        # Deduct fees
        position_value = exit_price * self.position['size']
        fee = position_value * (self.fee_pct / 100)
        pnl_usd -= fee

        # Update capital
        self.capital += pnl_usd

        # Record trade
        trade_record = {
            'symbol': market_data['symbol'],
            'side': self.position['side'],
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'size': self.position['size'],
            'pnl_usd': pnl_usd,
            'pnl_pct': (pnl_usd / (self.position['entry_price'] * self.position['size'])) * 100,
            'entry_time': self.position['entry_time'],
            'exit_time': market_data['timestamp'],
            'exit_reason': reason,
            'regime': self.position['regime'],
            'trend': self.position['trend']
        }

        self.trades.append(trade_record)
        self.performance.record_trade(trade_record)

        logger.debug(f"Closed {self.position['side']} position: P&L ${pnl_usd:.2f} ({trade_record['pnl_pct']:.2f}%) - {reason}")

        self.position = None

    def _generate_results(self) -> Dict:
        """Generate backtest results"""

        metrics = self.performance.get_metrics()

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return_usd': self.capital - self.initial_capital,
            'total_return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'metrics': metrics,
            'trades': self.trades,
            'config': {
                'position_size_pct': self.position_size_pct,
                'fee_pct': self.fee_pct,
                'slippage_pct': self.slippage_pct
            }
        }

        return results

    def print_results(self):
        """Print backtest results"""
        self.performance.print_summary()

        print(f"\n💼 CAPITAL:")
        print(f"   Initial: ${self.initial_capital:.2f}")
        print(f"   Final: ${self.capital:.2f}")
        print(f"   Return: ${self.capital - self.initial_capital:.2f} ({((self.capital - self.initial_capital) / self.initial_capital) * 100:.2f}%)")
