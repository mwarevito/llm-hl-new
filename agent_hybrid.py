"""
Hybrid Trading Agent (Production-Ready)

This agent uses:
- LLM for regime classification (cached for 1-4 hours) → NON-DETERMINISTIC but stable
- Deterministic rules for trade signals → FULLY DETERMINISTIC and backtestable
- Enhanced performance metrics (Sharpe ratio, drawdown, etc.)

This hybrid approach preserves LLM intelligence while making the system:
✅ Reproducible (same conditions → same trades)
✅ Backtestable (can validate on historical data)
✅ Fast (no LLM latency on every trade)
✅ Cost-effective (95% fewer API calls)
✅ Transparent (clear entry/exit rules)
"""

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv
from loguru import logger

# Import existing components from agent_improved.py
# (We reuse: RiskManager, OrderTracker, MarketDataCollector, HyperliquidExecutor)
from agent_improved import (
    RiskManager,
    OrderTracker,
    MarketDataCollector,
    HyperliquidExecutor,
    start_health_server
)

# Import new hybrid components
from llm_regime_classifier import LLMRegimeClassifier
from signal_generator import SignalGenerator
from performance_metrics import PerformanceMetrics

load_dotenv()


class HybridTradingAgent:
    """
    Hybrid Trading Agent combining LLM intelligence with deterministic execution.

    Architecture:
    1. LLM classifies market regime every 1-4 hours (cached)
    2. Deterministic rules generate trade signals based on regime
    3. Clear risk management and performance tracking
    """

    def __init__(self,
                 symbol: str = "BTC",
                 position_size_pct: float = 10.0,
                 check_interval_seconds: int = 300,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 regime_cache_hours: float = 2.0,
                 max_daily_loss_pct: float = 5.0,
                 max_spread_bps: float = 50.0,
                 testnet: bool = True):

        self.symbol = symbol
        self.position_size_pct = position_size_pct
        self.check_interval = check_interval_seconds
        self.testnet = testnet

        logger.info("="*80)
        logger.info("INITIALIZING HYBRID TRADING AGENT (PRODUCTION MODE)")
        logger.info("="*80)

        # Initialize components
        self.data_collector = MarketDataCollector(testnet=testnet)

        # NEW: Regime classifier with caching
        self.regime_classifier = LLMRegimeClassifier(
            model=llm_model,
            provider=llm_provider,
            cache_duration_hours=regime_cache_hours
        )

        # NEW: Deterministic signal generator
        self.signal_generator = SignalGenerator(testnet=testnet)

        # Existing components
        self.executor = HyperliquidExecutor(testnet=testnet)
        self.risk_manager = RiskManager(
            max_daily_loss_pct=max_daily_loss_pct,
            max_position_size_pct=position_size_pct,
            max_spread_bps=max_spread_bps
        )

        # NEW: Enhanced performance tracking
        self.performance = PerformanceMetrics(save_path="data/hybrid_performance.json")

        self.order_tracker = OrderTracker()

        logger.success("✅ Hybrid Trading Agent initialized")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Position Size: {position_size_pct}% of account")
        logger.info(f"   Regime Cache: {regime_cache_hours}h")
        logger.info(f"   Mode: {'TESTNET' if testnet else 'MAINNET'}")
        logger.info("="*80)

    def run_once(self) -> Dict:
        """Run single iteration with hybrid decision-making"""

        logger.info(f"\n{'='*80}\nAgent Iteration - {datetime.now()}\n{'='*80}")

        try:
            # 1. Get account balance
            account_balance = self.executor.get_account_balance()
            if account_balance <= 0:
                logger.error("Cannot retrieve account balance")
                return {'error': 'No account balance'}

            self.performance.record_equity(account_balance)

            # 2. Risk checks
            if not self.risk_manager.check_account_balance(account_balance):
                return {'error': 'Insufficient balance'}

            if not self.risk_manager.check_daily_loss_limit(account_balance):
                return {'error': 'Daily loss limit exceeded'}

            if not self.risk_manager.check_weekly_loss_limit(account_balance):
                return {'error': 'Weekly loss limit exceeded'}

            # 3. Collect market data
            market_data = self.data_collector.get_market_snapshot(self.symbol)

            # 4. Check spread
            if not self.risk_manager.check_spread(
                market_data['orderbook']['best_bid'],
                market_data['orderbook']['best_ask']
            ):
                logger.warning("Spread too wide - skipping iteration")
                return {'skipped': 'Wide spread'}

            # 5. Get current position
            current_position = self.executor.get_current_position(self.symbol)

            # Reconcile order tracker
            reconcile_result = self.order_tracker.reconcile_with_exchange(
                exchange_position=current_position,
                exchange_orders=[]
            )

            if reconcile_result.get('position_closed_externally'):
                logger.info("Position closed externally (TP/SL hit)")

            if current_position:
                logger.info(f"Current position: {current_position['side']} {current_position['size']:.4f} @ ${current_position['entry_price']:.2f}, P&L: {current_position['pnl_pct']:.2f}%")

            # ======== HYBRID DECISION-MAKING ========
            # Step 1: Get regime classification from LLM (cached)
            regime_data = self.regime_classifier.get_regime(market_data)

            logger.info(f"🎯 Market Regime: {regime_data['regime']} / {regime_data['trend']}")
            logger.info(f"   Risk Mode: {regime_data['risk_mode']}, Confidence: {regime_data['confidence']:.2f}")

            # Step 2: Generate deterministic signal based on regime
            signal = self.signal_generator.generate_signal(
                market_data,
                regime_data,
                current_position
            )

            logger.info(f"📊 Signal: {signal.action} (confidence: {signal.confidence:.2f})")
            logger.info(f"   Reasoning: {signal.reason}")

            # Convert signal to dict for compatibility with executor
            decision = signal.to_dict()

            # ======== END HYBRID DECISION-MAKING ========

            # 7. Calculate position size (with consecutive loss adjustment)
            execution_price = market_data['orderbook']['best_ask'] if decision['action'] == 'OPEN_LONG' else market_data['orderbook']['best_bid']
            position_size_coins = self.risk_manager.calculate_position_size(account_balance, execution_price)

            # Apply signal's position size multiplier
            position_size_coins *= decision.get('position_size_multiplier', 1.0)

            # 8. Execute trade
            executed = self.executor.execute_trade(
                symbol=self.symbol,
                decision=decision,
                position_size_coins=position_size_coins,
                risk_checks_passed=True
            )

            # 9. Track performance
            if executed and decision['action'] == 'CLOSE' and current_position:
                # Record trade with regime metadata
                trade_data = {
                    **current_position,
                    'exit_reason': decision.get('reasoning', 'MANUAL'),
                    'regime': regime_data['regime'],
                    'trend': regime_data['trend'],
                    'exit_time': datetime.now().isoformat(),
                    'entry_time': datetime.now().isoformat()  # TODO: track actual entry time
                }

                self.risk_manager.record_trade(current_position['pnl_usd'])
                self.performance.record_trade(trade_data)

                logger.success(f"✅ Trade closed: ${current_position['pnl_usd']:.2f} ({current_position['pnl_pct']:.2f}%)")

            result = {
                'timestamp': datetime.now().isoformat(),
                'account_balance': account_balance,
                'regime': regime_data,
                'signal': decision,
                'position_size_coins': position_size_coins,
                'executed': executed,
                'daily_pnl': self.risk_manager.daily_pnl_usd,
                'weekly_pnl': self.risk_manager.weekly_pnl_usd
            }

            return result

        except Exception as e:
            logger.error(f"Error in run_once: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e)}

    def run_loop(self):
        """Run continuous trading loop"""

        logger.info(f"Starting continuous trading loop (check every {self.check_interval}s)")
        logger.info("Press Ctrl+C to stop")

        iteration = 0

        while True:
            try:
                iteration += 1
                logger.info(f"\n{'='*80}\nIteration {iteration}\n{'='*80}")

                result = self.run_once()

                # Print performance summary every 10 iterations
                if iteration % 10 == 0:
                    self.print_performance_summary()

                # Log result
                if 'error' in result:
                    logger.error(f"Iteration error: {result['error']}")
                else:
                    logger.info(f"Iteration complete. Daily P&L: ${self.risk_manager.daily_pnl_usd:.2f}, Weekly: ${self.risk_manager.weekly_pnl_usd:.2f}")

                logger.info(f"Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("\n" + "="*80)
                logger.info("Agent stopped by user")
                self.print_performance_summary()
                logger.info("="*80)
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Sleeping 60s before retry...")
                time.sleep(60)

    def print_performance_summary(self):
        """Print enhanced performance summary"""
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE SUMMARY (HYBRID SYSTEM)")
        logger.info("="*80)

        self.performance.print_summary()

        # Add regime cache statistics
        logger.info("\n📊 REGIME CACHE STATS:")
        if self.regime_classifier.cache.is_valid():
            cache_data = self.regime_classifier.cache.get()
            logger.info(f"   Current Regime: {cache_data['regime']} / {cache_data['trend']}")
            logger.info(f"   Valid Until: {cache_data['valid_until']}")
            logger.info(f"   Status: ✅ CACHED (fast execution)")
        else:
            logger.info(f"   Status: ⚠️ EXPIRED (will refresh on next check)")

        logger.info("="*80)


if __name__ == "__main__":
    # Start health check server for Railway
    start_health_server(port=int(os.getenv('PORT', 8080)))

    # Initialize hybrid agent
    agent = HybridTradingAgent(
        symbol="BTC",
        position_size_pct=10.0,
        check_interval_seconds=300,  # 5 minutes
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        regime_cache_hours=2.0,  # Cache regime for 2 hours
        max_daily_loss_pct=5.0,
        max_spread_bps=50.0,
        testnet=True
    )

    # Check run mode
    run_mode = os.getenv("RUN_MODE", "once").lower()

    if run_mode == "continuous":
        logger.info("Starting agent in CONTINUOUS mode")
        agent.run_loop()
    else:
        logger.info("Starting agent in SINGLE RUN mode")
        logger.info("Running single test iteration...")
        result = agent.run_once()

        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)

        # Print simplified result
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Execution successful")
            print(f"\nRegime: {result['regime']['regime']} / {result['regime']['trend']}")
            print(f"Signal: {result['signal']['action']} (confidence: {result['signal']['confidence']:.2f})")
            print(f"Reasoning: {result['signal']['reasoning']}")
            print(f"Account Balance: ${result['account_balance']:.2f}")
            print(f"Daily P&L: ${result['daily_pnl']:.2f}")

        print("="*80)

        # Print performance metrics if any trades exist
        agent.print_performance_summary()
