"""
Performance Metrics Tracker

Calculates hedge-fund quality metrics:
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown
- Win Rate, Profit Factor
- Sortino Ratio (downside risk)
- Calmar Ratio (return/max drawdown)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from loguru import logger


class PerformanceMetrics:
    """
    Track and calculate comprehensive trading performance metrics.

    This is essential for validating strategy performance and comparing
    against hedge fund standards.
    """

    def __init__(self, save_path: str = "data/performance_metrics.json"):
        self.save_path = save_path
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []

        # Load existing data
        self._load_from_disk()

        logger.info(f"PerformanceMetrics initialized: {len(self.trades)} trades loaded")

    def _load_from_disk(self):
        """Load performance data from disk"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.equity_curve = data.get('equity_curve', [])
                    self.daily_returns = data.get('daily_returns', [])
                    logger.info(f"Loaded performance data: {len(self.trades)} trades")
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")

    def _save_to_disk(self):
        """Save performance data to disk"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            data = {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'daily_returns': self.daily_returns,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def record_trade(self, trade_data: Dict):
        """
        Record a completed trade.

        Args:
            trade_data: {
                'symbol': str,
                'side': 'LONG' or 'SHORT',
                'entry_price': float,
                'exit_price': float,
                'size': float,
                'pnl_usd': float,
                'pnl_pct': float,
                'entry_time': ISO timestamp,
                'exit_time': ISO timestamp,
                'exit_reason': str,
                'regime': str,
                'trend': str
            }
        """
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_data.get('symbol'),
            'side': trade_data.get('side'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('exit_price'),
            'size': trade_data.get('size'),
            'pnl_usd': trade_data.get('pnl_usd', 0),
            'pnl_pct': trade_data.get('pnl_pct', 0),
            'entry_time': trade_data.get('entry_time'),
            'exit_time': trade_data.get('exit_time'),
            'exit_reason': trade_data.get('exit_reason', 'UNKNOWN'),
            'regime': trade_data.get('regime', 'UNKNOWN'),
            'trend': trade_data.get('trend', 'UNKNOWN')
        }

        self.trades.append(trade_record)
        self._save_to_disk()

        logger.info(f"Trade recorded: {trade_record['side']} P&L: ${trade_record['pnl_usd']:.2f} ({trade_record['pnl_pct']:.2f}%)")

    def record_equity(self, balance: float, timestamp: Optional[str] = None):
        """Record equity snapshot for equity curve"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': balance
        })

        # Calculate daily returns if we have previous day's equity
        if len(self.equity_curve) >= 2:
            prev_balance = self.equity_curve[-2]['balance']
            daily_return = (balance - prev_balance) / prev_balance
            self.daily_returns.append(daily_return)

        self._save_to_disk()

    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns hedge-fund quality metrics for strategy validation.
        """
        if not self.trades:
            return self._empty_metrics()

        trades_data = self.trades
        winning_trades = [t for t in trades_data if t['pnl_usd'] > 0]
        losing_trades = [t for t in trades_data if t['pnl_usd'] < 0]

        # Basic metrics
        total_trades = len(trades_data)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum([t['pnl_usd'] for t in trades_data])
        avg_win = np.mean([t['pnl_usd'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_usd'] for t in losing_trades]) if losing_trades else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        best_trade = max([t['pnl_pct'] for t in trades_data]) if trades_data else 0
        worst_trade = min([t['pnl_pct'] for t in trades_data]) if trades_data else 0

        # Profit Factor (gross profit / gross loss)
        gross_profit = sum([t['pnl_usd'] for t in winning_trades])
        gross_loss = abs(sum([t['pnl_usd'] for t in losing_trades]))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        # Expectancy (average $ per trade)
        expectancy = avg_pnl

        # Risk/Reward Ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Sharpe Ratio (risk-adjusted returns)
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Sortino Ratio (downside risk)
        sortino_ratio = self._calculate_sortino_ratio()

        # Maximum Drawdown
        max_drawdown_pct, max_drawdown_usd = self._calculate_max_drawdown()

        # Calmar Ratio (annual return / max drawdown)
        annual_return_pct = self._calculate_annual_return()
        calmar_ratio = (annual_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0

        # Average trade duration
        avg_duration_hours = self._calculate_avg_duration()

        metrics = {
            # Basic Stats
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate_pct': win_rate,

            # P&L
            'total_pnl_usd': total_pnl,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'avg_pnl_usd': avg_pnl,
            'best_trade_pct': best_trade,
            'worst_trade_pct': worst_trade,

            # Advanced Metrics
            'profit_factor': profit_factor,
            'expectancy_usd': expectancy,
            'risk_reward_ratio': risk_reward,

            # Risk-Adjusted Returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,

            # Drawdown
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_usd': max_drawdown_usd,

            # Performance
            'annual_return_pct': annual_return_pct,
            'avg_trade_duration_hours': avg_duration_hours,

            # Quality Scores (hedge fund standards)
            'quality_score': self._calculate_quality_score(sharpe_ratio, max_drawdown_pct, profit_factor, win_rate),
            'hedge_fund_ready': self._is_hedge_fund_ready(sharpe_ratio, max_drawdown_pct, total_trades)
        }

        return metrics

    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades exist"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0.0,
            'total_pnl_usd': 0.0,
            'avg_win_usd': 0.0,
            'avg_loss_usd': 0.0,
            'avg_pnl_usd': 0.0,
            'best_trade_pct': 0.0,
            'worst_trade_pct': 0.0,
            'profit_factor': 0.0,
            'expectancy_usd': 0.0,
            'risk_reward_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_usd': 0.0,
            'annual_return_pct': 0.0,
            'avg_trade_duration_hours': 0.0,
            'quality_score': 0.0,
            'hedge_fund_ready': False
        }

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        Calculate Sharpe Ratio (annualized).

        Sharpe = (Return - RiskFreeRate) / StdDev

        Hedge fund standard: > 1.5 is good, > 2.0 is excellent
        """
        if len(self.daily_returns) < 2:
            return 0.0

        returns = np.array(self.daily_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize (assuming 365 days)
        annual_return = mean_return * 365
        annual_std = std_return * np.sqrt(365)
        annual_risk_free = risk_free_rate

        sharpe = (annual_return - annual_risk_free) / annual_std

        return float(sharpe)

    def _calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility).

        Sortino = (Return - Target) / DownsideStdDev
        """
        if len(self.daily_returns) < 2:
            return 0.0

        returns = np.array(self.daily_returns)
        mean_return = np.mean(returns)

        # Only consider negative returns for downside deviation
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        # Annualize
        annual_return = mean_return * 365
        annual_downside_std = downside_std * np.sqrt(365)

        sortino = (annual_return - target_return) / annual_downside_std

        return float(sortino)

    def _calculate_max_drawdown(self) -> tuple:
        """
        Calculate maximum drawdown (peak to trough decline).

        Returns:
            (max_drawdown_pct, max_drawdown_usd)
        """
        if not self.equity_curve:
            return (0.0, 0.0)

        balances = [e['balance'] for e in self.equity_curve]
        peak = balances[0]
        max_dd_pct = 0.0
        max_dd_usd = 0.0

        for balance in balances:
            if balance > peak:
                peak = balance

            drawdown_usd = peak - balance
            drawdown_pct = (drawdown_usd / peak) * 100 if peak > 0 else 0

            if drawdown_pct > max_dd_pct:
                max_dd_pct = drawdown_pct
                max_dd_usd = drawdown_usd

        return (float(max_dd_pct), float(max_dd_usd))

    def _calculate_annual_return(self) -> float:
        """Calculate annualized return percentage"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0

        initial_balance = self.equity_curve[0]['balance']
        final_balance = self.equity_curve[-1]['balance']

        if initial_balance == 0:
            return 0.0

        # Calculate total return
        total_return = (final_balance - initial_balance) / initial_balance

        # Calculate time period in years
        start_time = datetime.fromisoformat(self.equity_curve[0]['timestamp'])
        end_time = datetime.fromisoformat(self.equity_curve[-1]['timestamp'])
        days = (end_time - start_time).days

        if days == 0:
            return 0.0

        years = days / 365.0

        # Annualize return
        annual_return = (total_return / years) * 100

        return float(annual_return)

    def _calculate_avg_duration(self) -> float:
        """Calculate average trade duration in hours"""
        if not self.trades:
            return 0.0

        durations = []
        for trade in self.trades:
            try:
                if trade.get('entry_time') and trade.get('exit_time'):
                    entry = datetime.fromisoformat(trade['entry_time'])
                    exit_t = datetime.fromisoformat(trade['exit_time'])
                    duration_hours = (exit_t - entry).total_seconds() / 3600
                    durations.append(duration_hours)
            except:
                continue

        if not durations:
            return 0.0

        return float(np.mean(durations))

    def _calculate_quality_score(self, sharpe: float, max_dd: float, profit_factor: float, win_rate: float) -> float:
        """
        Calculate overall strategy quality score (0-100).

        Combines multiple metrics into single score for easy comparison.
        """
        score = 0.0

        # Sharpe contribution (0-30 points)
        if sharpe > 2.0:
            score += 30
        elif sharpe > 1.5:
            score += 25
        elif sharpe > 1.0:
            score += 20
        elif sharpe > 0.5:
            score += 10

        # Max Drawdown contribution (0-25 points)
        if max_dd < 5:
            score += 25
        elif max_dd < 10:
            score += 20
        elif max_dd < 15:
            score += 15
        elif max_dd < 20:
            score += 10

        # Profit Factor contribution (0-25 points)
        if profit_factor > 2.0:
            score += 25
        elif profit_factor > 1.5:
            score += 20
        elif profit_factor > 1.3:
            score += 15
        elif profit_factor > 1.1:
            score += 10

        # Win Rate contribution (0-20 points)
        if win_rate > 60:
            score += 20
        elif win_rate > 55:
            score += 15
        elif win_rate > 50:
            score += 10
        elif win_rate > 45:
            score += 5

        return score

    def _is_hedge_fund_ready(self, sharpe: float, max_dd: float, total_trades: int) -> bool:
        """
        Determine if strategy meets hedge fund standards.

        Criteria:
        - Sharpe Ratio > 1.5
        - Max Drawdown < 15%
        - At least 100 trades (statistical significance)
        """
        return (sharpe > 1.5 and
                max_dd < 15.0 and
                total_trades >= 100)

    def print_summary(self):
        """Print comprehensive performance summary"""
        metrics = self.get_metrics()

        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)

        print(f"\n📊 BASIC STATS:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate_pct']:.2f}%")
        print(f"   Winners: {metrics['winning_trades']} | Losers: {metrics['losing_trades']}")

        print(f"\n💰 P&L:")
        print(f"   Total P&L: ${metrics['total_pnl_usd']:.2f}")
        print(f"   Avg Win: ${metrics['avg_win_usd']:.2f}")
        print(f"   Avg Loss: ${metrics['avg_loss_usd']:.2f}")
        print(f"   Best Trade: {metrics['best_trade_pct']:.2f}%")
        print(f"   Worst Trade: {metrics['worst_trade_pct']:.2f}%")

        print(f"\n📈 ADVANCED METRICS:")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Expectancy: ${metrics['expectancy_usd']:.2f} per trade")
        print(f"   Risk/Reward: {metrics['risk_reward_ratio']:.2f}")

        print(f"\n🎯 RISK-ADJUSTED RETURNS:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f} {self._sharpe_rating(metrics['sharpe_ratio'])}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio: {metrics['calmar_ratio']:.2f}")

        print(f"\n⚠️  RISK METRICS:")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}% (${metrics['max_drawdown_usd']:.2f})")
        print(f"   Annual Return: {metrics['annual_return_pct']:.2f}%")

        print(f"\n⭐ QUALITY ASSESSMENT:")
        print(f"   Quality Score: {metrics['quality_score']:.0f}/100")
        print(f"   Hedge Fund Ready: {'✅ YES' if metrics['hedge_fund_ready'] else '❌ NO (need more data)'}")

        print("="*80 + "\n")

    def _sharpe_rating(self, sharpe: float) -> str:
        """Get rating for Sharpe ratio"""
        if sharpe > 2.0:
            return "(⭐⭐⭐ Excellent)"
        elif sharpe > 1.5:
            return "(⭐⭐ Good)"
        elif sharpe > 1.0:
            return "(⭐ Acceptable)"
        elif sharpe > 0.5:
            return "(⚠️  Below Target)"
        else:
            return "(❌ Poor)"
