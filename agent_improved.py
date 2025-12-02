import os
import json
import time
import traceback
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np

# Hyperliquid
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Technical indicators
import ta

# LLM
from openai import OpenAI
# from anthropic import Anthropic  # Alternative

load_dotenv()

# Configure logging with sanitization
logger.add(
    "logs/llm_agent_improved_{time}.log",
    rotation="1 day",
    filter=lambda record: "PRIVATE" not in record["message"]  # Basic key filtering
)


STATE_FILE = 'data/risk_state.json'

class RiskManager:
    """Manages risk limits and circuit breakers"""

    def __init__(self,
                 max_daily_loss_pct: float = 5.0,
                 max_position_size_pct: float = 10.0,
                 min_account_balance_usd: float = 100.0,
                 max_spread_bps: float = 50.0):
        """
        Initialize risk manager

        Args:
            max_daily_loss_pct: Maximum daily loss percentage before circuit breaker
            max_position_size_pct: Maximum position size as % of account
            min_account_balance_usd: Minimum account balance required to trade
            max_spread_bps: Maximum allowed spread in basis points
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size_pct = max_position_size_pct
        self.min_account_balance_usd = min_account_balance_usd
        self.max_spread_bps = max_spread_bps

        # Track daily P&L - Initialize BEFORE loading state
        self.daily_pnl_usd = 0.0
        self.last_reset_date = datetime.now().date()
        self.last_reset_time = datetime.now()
        self.trades_today = 0

        # Load saved state (will overwrite defaults if exists)
        self.load_state()

        logger.info(f"RiskManager initialized: max_daily_loss={max_daily_loss_pct}%, max_position={max_position_size_pct}%")

    def reset_daily_limits(self):
        """Reset daily limits at start of new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            logger.info(f"Resetting daily limits. Previous P&L: ${self.daily_pnl_usd:.2f}, Trades: {self.trades_today}")
            self.daily_pnl_usd = 0.0
            self.trades_today = 0
            self.last_reset_date = current_date
            self.last_reset_time = datetime.now()
            self.save_state()  # Persist the reset

    def check_daily_loss_limit(self, account_balance: float) -> bool:
        """Check if daily loss limit exceeded"""
        self.reset_daily_limits()

        loss_pct = (self.daily_pnl_usd / account_balance) * 100 if account_balance > 0 else 0

        if loss_pct <= -self.max_daily_loss_pct:
            logger.error(f"CIRCUIT BREAKER: Daily loss limit exceeded ({loss_pct:.2f}% < -{self.max_daily_loss_pct}%)")
            return False

        return True

    def check_spread(self, best_bid: float, best_ask: float) -> bool:
        """Check if spread is acceptable"""
        if best_bid <= 0 or best_ask <= 0:
            logger.warning("Invalid orderbook prices")
            return False

        spread_bps = ((best_ask - best_bid) / best_bid) * 10000

        if spread_bps > self.max_spread_bps:
            logger.warning(f"Spread too wide: {spread_bps:.2f} bps > {self.max_spread_bps} bps")
            return False

        return True


    def load_state(self):
        """Load daily PnL state from file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    # Check if it's the same day
                    saved_date = datetime.fromisoformat(state['last_reset_time']).date()
                    current_date = datetime.now().date()
                    
                    if saved_date == current_date:
                        self.daily_pnl_usd = state['daily_pnl_usd']
                        self.last_reset_time = datetime.fromisoformat(state['last_reset_time'])
                        logger.info(f"Loaded risk state: Daily P&L ${self.daily_pnl_usd:.2f}")
                    else:
                        logger.info("Saved state is from previous day. Resetting.")
        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")

    def save_state(self):
        """Save daily PnL state to file"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

            state = {
                'daily_pnl_usd': self.daily_pnl_usd,
                'last_reset_time': self.last_reset_time.isoformat()
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def check_account_balance(self, balance: float) -> bool:
        """Check if account has sufficient balance"""
        if balance < self.min_account_balance_usd:
            logger.error(f"Insufficient account balance: ${balance:.2f} < ${self.min_account_balance_usd}")
            return False
        return True

    def calculate_position_size(self, account_balance: float, price: float) -> float:
        """Calculate safe position size based on account balance"""
        max_position_usd = account_balance * (self.max_position_size_pct / 100)
        position_coins = max_position_usd / price

        logger.info(f"Position sizing: Account=${account_balance:.2f}, Max position=${max_position_usd:.2f}, Coins={position_coins:.4f}")

        return position_coins

    def record_trade(self, pnl_usd: float):
        """Record trade result for daily tracking"""
        self.daily_pnl_usd += pnl_usd
        self.trades_today += 1
        logger.info(f"Trade recorded: P&L=${pnl_usd:.2f}, Daily total=${self.daily_pnl_usd:.2f}, Trades today={self.trades_today}")
        self.save_state()  # Persist state after each trade


class PerformanceTracker:
    """Track trading performance metrics"""

    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def record_trade(self, trade_data: Dict):
        """Record a completed trade"""
        self.trades.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_data.get('symbol'),
            'side': trade_data.get('side'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('exit_price'),
            'size': trade_data.get('size'),
            'pnl_usd': trade_data.get('pnl_usd', 0),
            'pnl_pct': trade_data.get('pnl_pct', 0),
            'duration_seconds': trade_data.get('duration_seconds', 0)
        })

    def record_equity(self, balance: float):
        """Record equity snapshot"""
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'balance': balance
        })

    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win_usd': 0.0,
                'avg_loss_usd': 0.0,
                'avg_pnl_pct': 0.0,
                'total_pnl_usd': 0.0,
                'best_trade_pct': 0.0,
                'worst_trade_pct': 0.0
            }

        winning_trades = [t for t in self.trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_usd'] < 0]

        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'avg_win_usd': np.mean([t['pnl_usd'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss_usd': np.mean([t['pnl_usd'] for t in losing_trades]) if losing_trades else 0,
            'total_pnl_usd': sum([t['pnl_usd'] for t in self.trades]),
            'avg_pnl_pct': np.mean([t['pnl_pct'] for t in self.trades]) if self.trades else 0,
            'best_trade_pct': max([t['pnl_pct'] for t in self.trades]) if self.trades else 0,
            'worst_trade_pct': min([t['pnl_pct'] for t in self.trades]) if self.trades else 0,
        }

        return metrics

    def print_summary(self):
        """Print performance summary"""
        metrics = self.get_metrics()

        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Total P&L: ${metrics['total_pnl_usd']:.2f}")
        logger.info(f"Average P&L: {metrics['avg_pnl_pct']:.2f}%")
        logger.info(f"Best Trade: {metrics['best_trade_pct']:.2f}%")
        logger.info(f"Worst Trade: {metrics['worst_trade_pct']:.2f}%")
        logger.info("="*80)


class MarketDataCollector:
    """Fetch and process market data from Hyperliquid"""

    def __init__(self, testnet: bool = True):
        url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(url, skip_ws=True)

    def get_market_snapshot(self, symbol: str = "BTC") -> Dict:
        """Get comprehensive market snapshot with indicators"""

        # Get recent candles
        end_time = int(time.time() * 1000)
        interval = "1h"
        lookback = 100

        # Hyperliquid API: candles_snapshot(name, interval, startTime, endTime)
        candles = self.info.candles_snapshot(
            name=symbol,
            interval=interval,
            startTime=end_time - (lookback * 3600 * 1000),
            endTime=end_time
        )

        if not candles or len(candles) < 50:
            raise ValueError(f"Insufficient candle data: {len(candles) if candles else 0} candles")

        # Convert to DataFrame
        # Hyperliquid candles format: {'t': timestamp, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume}
        df = pd.DataFrame(candles)
        df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        df['volume'] = pd.to_numeric(df['volume'])

        # Calculate indicators
        indicators = self._calculate_indicators(df)

        # Get orderbook
        book = self.info.l2_snapshot(symbol)

        # Parse orderbook (Hyperliquid format may vary)
        try:
            if 'levels' in book and isinstance(book['levels'], list) and len(book['levels']) >= 2:
                # Format: book['levels'][0] = bids, book['levels'][1] = asks
                bids = book['levels'][0] if book['levels'][0] else []
                asks = book['levels'][1] if book['levels'][1] else []
            else:
                # Alternate format: book may have 'bids' and 'asks' directly
                bids = book.get('bids', [])
                asks = book.get('asks', [])

            # Check if bids/asks are list of lists or list of dicts
            if bids and isinstance(bids[0], dict):
                best_bid = float(bids[0]['px'])
                bid_depth = sum([float(b['sz']) for b in bids[:10]])
            elif bids:
                best_bid = float(bids[0][0])
                bid_depth = sum([float(b[1]) for b in bids[:10]])
            else:
                best_bid = float(df['close'].iloc[-1])
                bid_depth = 0

            if asks and isinstance(asks[0], dict):
                best_ask = float(asks[0]['px'])
                ask_depth = sum([float(a['sz']) for a in asks[:10]])
            elif asks:
                best_ask = float(asks[0][0])
                ask_depth = sum([float(a[1]) for a in asks[:10]])
            else:
                best_ask = float(df['close'].iloc[-1])
                ask_depth = 0
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to parse orderbook, using close price: {e}")
            # Fallback to last close price
            best_bid = float(df['close'].iloc[-1])
            best_ask = float(df['close'].iloc[-1])
            bid_depth = 0
            ask_depth = 0

        # Format market snapshot
        snapshot = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(df['close'].iloc[-1]),
            'indicators': indicators,
            'orderbook': {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_depth_10': bid_depth,
                'ask_depth_10': ask_depth,
            },
            'recent_candles': df[['close', 'volume']].tail(5).to_dict('records')
        }

        logger.info(f"Market snapshot collected for {symbol}: ${snapshot['current_price']:.2f}")
        return snapshot

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Moving Averages
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]

        # FIXED: Trend direction based on PRICE vs MAs, not MA crossover
        price_vs_ma20 = ((current_price - ma_20) / ma_20 * 100)
        price_vs_ma50 = ((current_price - ma_50) / ma_50 * 100)

        # Price above both MAs = bullish, below both = bearish
        if current_price > ma_20 and current_price > ma_50:
            trend = "BULLISH"
        elif current_price < ma_20 and current_price < ma_50:
            trend = "BEARISH"
        else:
            trend = "MIXED"

        # Trend strength based on how far price is from MAs
        trend_strength = abs(price_vs_ma20)

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

        # MACD
        macd_indicator = ta.trend.MACD(close)
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        macd_diff = macd_indicator.macd_diff().iloc[-1]

        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close)
        bb_high = bb_indicator.bollinger_hband().iloc[-1]
        bb_low = bb_indicator.bollinger_lband().iloc[-1]
        bb_mid = bb_indicator.bollinger_mavg().iloc[-1]

        # BB position (where price is relative to bands)
        bb_position = (close.iloc[-1] - bb_low) / (bb_high - bb_low) if (bb_high - bb_low) > 0 else 0.5

        # ATR (volatility)
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1]
        atr_pct = (atr / close.iloc[-1]) * 100

        # Volume analysis
        volume_ma_20 = volume.rolling(20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / volume_ma_20 if volume_ma_20 > 0 else 1.0

        # Price momentum
        returns_24h = ((close.iloc[-1] - close.iloc[-24]) / close.iloc[-24] * 100) if len(close) >= 24 else 0

        # Price momentum over 10 hours (for regime detection)
        returns_10h = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100) if len(close) >= 10 else 0

        # MARKET REGIME DETECTION
        # Momentum regime: strong directional move with volume
        # Mean reversion regime: range-bound, low momentum
        # Transitional: mixed signals

        momentum_score = 0

        # Factor 1: Price change magnitude (10h)
        if abs(returns_10h) > 3.0:  # >3% move in 10 hours
            momentum_score += 2
        elif abs(returns_10h) > 1.5:  # >1.5% move
            momentum_score += 1

        # Factor 2: RSI trending (not bouncing at extremes)
        if rsi > 60 and returns_10h > 0:  # Strong bullish momentum
            momentum_score += 2
        elif rsi < 40 and returns_10h < 0:  # Strong bearish momentum
            momentum_score += 2
        elif 30 <= rsi <= 70:  # RSI in middle = ranging
            momentum_score -= 1

        # Factor 3: Bollinger Band position
        if bb_position > 0.8 or bb_position < 0.2:  # Near extremes
            if abs(returns_10h) > 2.0:  # With momentum = breakout
                momentum_score += 2
            else:  # Without momentum = reversal zone
                momentum_score -= 1

        # Factor 4: Volume confirmation
        if volume_ratio > 1.3:  # High volume supports momentum
            momentum_score += 1
        elif volume_ratio < 0.7:  # Low volume = ranging
            momentum_score -= 1

        # Factor 5: Trend alignment
        if trend == "BULLISH" and returns_10h > 1.0:
            momentum_score += 1
        elif trend == "BEARISH" and returns_10h < -1.0:
            momentum_score += 1
        elif trend == "MIXED":
            momentum_score -= 1

        # Determine market regime
        if momentum_score >= 4:
            market_regime = "MOMENTUM"
        elif momentum_score <= 1:
            market_regime = "RANGING"
        else:
            market_regime = "TRANSITIONAL"

        indicators = {
            'ma_20': float(ma_20),
            'ma_50': float(ma_50),
            'trend': trend,
            'trend_strength_pct': float(trend_strength),
            'price_vs_ma20_pct': float(price_vs_ma20),
            'price_vs_ma50_pct': float(price_vs_ma50),
            'rsi_14': float(rsi),
            'macd': float(macd),
            'macd_signal': float(macd_signal),
            'macd_histogram': float(macd_diff),
            'bollinger_upper': float(bb_high),
            'bollinger_lower': float(bb_low),
            'bollinger_middle': float(bb_mid),
            'bollinger_position': float(bb_position),
            'atr': float(atr),
            'atr_pct': float(atr_pct),
            'volume_ratio': float(volume_ratio),
            'returns_24h_pct': float(returns_24h),
            'returns_10h_pct': float(returns_10h),
            'market_regime': market_regime,
            'momentum_score': momentum_score
        }

        return indicators


@dataclass
class LLMConfig:
    """Configuration for LLM parameters"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.5
    max_tokens: int = 800
    provider: str = "openai"

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
            provider=os.getenv("LLM_PROVIDER", "openai")
        )


class LLMTradingDecision:
    """Use LLM to make trading decisions"""

    def __init__(self, model: str = "gpt-4o", provider: str = "openai"):
        self.provider = provider
        self.model = model

        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        logger.info(f"LLM initialized: {provider}/{model}")

    def get_trading_decision(self, market_data: Dict, current_position: Optional[Dict] = None) -> Dict:
        """
        Get trading decision from LLM

        Returns:
            {
                'action': 'OPEN_LONG' | 'OPEN_SHORT' | 'CLOSE' | 'HOLD',
                'confidence': 0.0-1.0,
                'reasoning': str,
                'take_profit_pct': float,
                'stop_loss_pct': float
            }
        """

        # Create prompt
        prompt = self._create_prompt(market_data, current_position)

        # Get LLM response
        if self.provider == "openai":
            response = self._query_openai(prompt)
        else:
            response = self._query_anthropic(prompt)

        # Parse response
        decision = self._parse_response(response)

        logger.info(f"LLM Decision: {decision['action']} (confidence: {decision['confidence']:.2f})")
        logger.info(f"Reasoning: {decision['reasoning']}")

        return decision

    def _create_prompt(self, market_data: Dict, current_position: Optional[Dict]) -> str:
        """Create structured prompt for LLM with enhanced context"""

        position_info = "No position currently open."
        if current_position:
            position_info = f"""
Current Position: {current_position['side']} {current_position['size']:.4f} @ ${current_position['entry_price']:.2f}
Current Price: ${current_position['current_price']:.2f}
Unrealized P&L: {current_position['pnl_pct']:.2f}%
"""

        ind = market_data['indicators']
        ob = market_data['orderbook']

        # Calculate spread percentage
        spread_pct = ((ob['best_ask'] - ob['best_bid']) / ob['best_bid'] * 100) if ob['best_bid'] > 0 else 0

        prompt = f"""You are an expert cryptocurrency trading analyst. Analyze the following market data and provide a trading decision.

MARKET DATA FOR {market_data['symbol']}:
- Current Price: ${market_data['current_price']:.2f}
- Timestamp: {market_data['timestamp']}

🎯 MARKET REGIME: {ind['market_regime']} (Score: {ind.get('momentum_score', 0)})
- MOMENTUM: Strong directional move - trade with the trend, RSI 70-80 is normal
- RANGING: Sideways choppy action - mean reversion, buy dips/sell rallies
- TRANSITIONAL: Mixed signals - wait for clarity or reduce position size

TREND ANALYSIS:
- Trend Direction: {ind['trend']}
- Trend Strength: {ind['trend_strength_pct']:.2f}%
- MA(20): ${ind['ma_20']:.2f}
- MA(50): ${ind['ma_50']:.2f}
- Price vs MA(20): {ind['price_vs_ma20_pct']:.2f}%
- Price vs MA(50): {ind['price_vs_ma50_pct']:.2f}%

MOMENTUM INDICATORS:
- RSI(14): {ind['rsi_14']:.2f} [{self._rsi_interpretation(ind['rsi_14'])}]
- MACD: {ind['macd']:.4f}
- MACD Signal: {ind['macd_signal']:.4f}
- MACD Histogram: {ind['macd_histogram']:.4f} [{'BULLISH' if ind['macd_histogram'] > 0 else 'BEARISH'}]

VOLATILITY & RISK:
- Bollinger Bands: Upper ${ind['bollinger_upper']:.2f}, Middle ${ind['bollinger_middle']:.2f}, Lower ${ind['bollinger_lower']:.2f}
- Price Position in BB: {ind['bollinger_position']*100:.1f}% (0%=lower band, 100%=upper band)
- ATR: {ind['atr']:.2f} ({ind['atr_pct']:.2f}% of price)
- 10h Returns: {ind['returns_10h_pct']:.2f}%
- 24h Returns: {ind['returns_24h_pct']:.2f}%

VOLUME:
- Volume Ratio: {ind['volume_ratio']:.2f}x average [{'HIGH' if ind['volume_ratio'] > 1.5 else 'NORMAL' if ind['volume_ratio'] > 0.7 else 'LOW'}]

LIQUIDITY:
- Best Bid: ${ob['best_bid']:.2f}
- Best Ask: ${ob['best_ask']:.2f}
- Spread: {spread_pct:.3f}%
- Bid Depth (10 levels): {ob['bid_depth_10']:.2f}
- Ask Depth (10 levels): {ob['ask_depth_10']:.2f}

CURRENT POSITION:
{position_info}

TRADING INSTRUCTIONS (REGIME-SPECIFIC):

📊 CURRENT REGIME: {ind['market_regime']}

{"="*60}
IF MOMENTUM REGIME (current: {'✅ YES' if ind['market_regime'] == 'MOMENTUM' else '❌ NO'}):
{"="*60}
Strategy: TREND-FOLLOWING / MOMENTUM TRADING

✅ ENTRY SIGNALS (LONG):
- Price > MA(20) AND MA(20) > MA(50) (confirmed uptrend)
- RSI 60-80 is NORMAL in strong trends (NOT overbought!)
- Price breaking above BB upper band + volume spike = BREAKOUT
- 10h returns > +2% with volume ratio > 1.2
- MACD histogram positive and expanding

✅ ENTRY SIGNALS (SHORT):
- Price < MA(20) AND MA(20) < MA(50) (confirmed downtrend)
- RSI 20-40 is NORMAL in strong downtrends (NOT oversold!)
- Price breaking below BB lower band + volume spike = BREAKDOWN
- 10h returns < -2% with volume ratio > 1.2
- MACD histogram negative and expanding

🚫 AVOID:
- Counter-trend trades (don't fade strong momentum)
- Waiting for "oversold" in uptrend or "overbought" in downtrend
- Mean reversion signals (they fail in momentum regimes)

Confidence threshold: > 0.55 for entries

{"="*60}
IF RANGING REGIME (current: {'✅ YES' if ind['market_regime'] == 'RANGING' else '❌ NO'}):
{"="*60}
Strategy: MEAN REVERSION

✅ ENTRY SIGNALS (LONG):
- RSI < 30 (oversold bounce)
- Price near BB lower band (< 20%)
- Recent pullback -2% or more
- Volume declining (exhaustion)

✅ ENTRY SIGNALS (SHORT):
- RSI > 70 (overbought fade)
- Price near BB upper band (> 80%)
- Recent rally +2% or more
- Volume declining (exhaustion)

🚫 AVOID:
- Chasing breakouts (likely false in ranging market)
- Holding through extremes (take quick profits)

Confidence threshold: > 0.60 for entries

{"="*60}
IF TRANSITIONAL REGIME (current: {'✅ YES' if ind['market_regime'] == 'TRANSITIONAL' else '❌ NO'}):
{"="*60}
Strategy: WAIT FOR CLARITY or reduce size

- Only take highest-confidence setups (> 0.70)
- Prefer closing losing positions
- Wait for regime to clarify before opening new positions

{"="*60}
UNIVERSAL RULES (ALL REGIMES):
{"="*60}

1. POSITION MANAGEMENT:
   - If you have a winning position (>2% profit), consider taking profits
   - If position is losing and technical conditions have reversed, consider closing
   - Don't open new positions if spread > {spread_pct:.2f}% or volume ratio < 0.5

2. STOP-LOSS & TAKE-PROFIT:
   - Base SL/TP on ATR: Higher volatility = wider stops
   - Suggested SL: {ind['atr_pct']*1.5:.2f}% to {ind['atr_pct']*2.5:.2f}% (1.5-2.5x ATR)
   - Suggested TP: {ind['atr_pct']*2.5:.2f}% to {ind['atr_pct']*4:.2f}% (2.5-4x ATR)
   - Minimum Risk/Reward ratio: 1.5:1

3. VOLUME & LIQUIDITY:
   - Require volume ratio > 0.8 for new positions
   - Wide spreads = poor execution, skip trade

Respond ONLY with valid JSON in this exact format:
{{
    "action": "OPEN_LONG" or "OPEN_SHORT" or "CLOSE" or "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your decision covering trend, momentum, volatility, and risk",
    "take_profit_pct": 2.5,
    "stop_loss_pct": 1.5
}}

CRITICAL RULES:
- Confidence thresholds based on regime:
  * MOMENTUM: > 0.55 (lower threshold because RSI 70-80 is normal in trends)
  * RANGING: > 0.60 (slightly higher, more false signals)
  * TRANSITIONAL: > 0.70 (only take best setups)
- take_profit_pct should be based on ATR (typically {ind['atr_pct']*2.5:.2f}% to {ind['atr_pct']*4:.2f}%)
- stop_loss_pct should be based on ATR (typically {ind['atr_pct']*1.5:.2f}% to {ind['atr_pct']*2.5:.2f}%)
- Ensure TP/SL ratio is at least 1.5:1
- IMPORTANT: In MOMENTUM regime, RSI 70-80 does NOT mean overbought if trend is up!
- IMPORTANT: In RANGING regime, fade extremes; in MOMENTUM regime, follow breakouts
- Provide detailed reasoning that references regime, trend, RSI context, and volume
"""

        return prompt

    def _rsi_interpretation(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi >= 70:
            return "OVERBOUGHT"
        elif rsi <= 30:
            return "OVERSOLD"
        elif 45 <= rsi <= 55:
            return "NEUTRAL"
        elif rsi > 55:
            return "BULLISH"
        else:
            return "BEARISH"

    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency trading analyst. Always respond with valid JSON. Be conservative and prioritize capital preservation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Very low temperature for consistency
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response with validation"""

        try:
            # Extract JSON from response (in case LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]

            decision = json.loads(json_str)

            # Validate required fields
            required = ['action', 'confidence', 'reasoning']
            for field in required:
                if field not in decision:
                    raise ValueError(f"Missing required field: {field}")

            # Validate action
            valid_actions = ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'HOLD']
            if decision['action'] not in valid_actions:
                logger.warning(f"Invalid action '{decision['action']}', defaulting to HOLD")
                decision['action'] = 'HOLD'

            # Ensure numeric fields
            decision['confidence'] = float(decision.get('confidence', 0.5))
            decision['take_profit_pct'] = float(decision.get('take_profit_pct', 2.0))
            decision['stop_loss_pct'] = float(decision.get('stop_loss_pct', 1.0))

            # Clamp values
            decision['confidence'] = max(0.0, min(1.0, decision['confidence']))
            decision['take_profit_pct'] = max(0.5, min(15.0, decision['take_profit_pct']))
            decision['stop_loss_pct'] = max(0.3, min(10.0, decision['stop_loss_pct']))

            return decision

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response}")

            # Return safe default
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': 'Failed to parse LLM response',
                'take_profit_pct': 2.0,
                'stop_loss_pct': 1.0
            }


class HyperliquidExecutor:
    """Execute trades on Hyperliquid with improved safety"""

    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL

        self.exchange = Exchange(
            wallet=None,  # Will be set with private key
            base_url=url,
            account_address=os.getenv("HYPERLIQUID_ADDRESS")
        )

        # Set private key
        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        if private_key:
            self.exchange.wallet = private_key

        self.info = Info(url, skip_ws=True)

        logger.info(f"Hyperliquid executor initialized (testnet={testnet})")

    def get_account_balance(self) -> float:
        """Get account balance in USD"""
        try:
            user_state = self.info.user_state(os.getenv("HYPERLIQUID_ADDRESS"))

            if 'marginSummary' in user_state:
                # Account value includes collateral and unrealized PnL
                account_value = float(user_state['marginSummary']['accountValue'])
                logger.info(f"Account balance: ${account_value:.2f}")
                return account_value

            logger.warning("Could not fetch account balance")
            return 0.0

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return 0.0

    def get_current_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol"""
        try:
            user_state = self.info.user_state(os.getenv("HYPERLIQUID_ADDRESS"))

            if 'assetPositions' in user_state:
                for pos in user_state['assetPositions']:
                    if pos['position']['coin'] == symbol:
                        size = float(pos['position']['szi'])
                        if size != 0:
                            entry_price = float(pos['position']['entryPx'])
                            current_price = float(pos['position']['markPx'])
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100

                            # Calculate PnL in USD
                            if size > 0:  # Long
                                pnl_usd = (current_price - entry_price) * size
                            else:  # Short
                                pnl_usd = (entry_price - current_price) * abs(size)

                            return {
                                'symbol': symbol,
                                'side': 'LONG' if size > 0 else 'SHORT',
                                'size': abs(size),
                                'entry_price': entry_price,
                                'current_price': current_price,
                                'pnl_pct': pnl_pct,
                                'pnl_usd': pnl_usd
                            }

            return None

        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None

    def execute_trade(self,
                     symbol: str,
                     decision: Dict,
                     position_size_coins: float,
                     risk_checks_passed: bool = True) -> bool:
        """
        Execute trade based on LLM decision with safety checks

        Args:
            symbol: Trading pair
            decision: LLM decision dict
            position_size_coins: Position size in coins (calculated by risk manager)
            risk_checks_passed: Whether risk checks have passed
        """

        if not risk_checks_passed:
            logger.error("Risk checks failed - trade blocked")
            return False

        try:
            action = decision['action']

            if action == 'HOLD':
                logger.info("Action is HOLD - no trade executed")
                return True

            if action == 'CLOSE':
                return self._close_position(symbol)

            if action in ['OPEN_LONG', 'OPEN_SHORT']:
                # Check if position already exists
                existing_position = self.get_current_position(symbol)
                if existing_position:
                    logger.warning(f"Position already exists: {existing_position['side']}. Skipping new position.")
                    return False

                return self._open_position(
                    symbol=symbol,
                    side=action,
                    size_coins=position_size_coins,
                    tp_pct=decision['take_profit_pct'],
                    sl_pct=decision['stop_loss_pct']
                )

            return False

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def _open_position(self, symbol: str, side: str, size_coins: float,
                      tp_pct: float, sl_pct: float) -> bool:
        """Open new position with TP/SL"""

        # Get current orderbook
        book = self.info.l2_snapshot(symbol)

        # FIXED: Use correct price for market orders
        # For LONG: use ask price (buying)
        # For SHORT: use bid price (selling)
        if side == 'OPEN_LONG':
            execution_price = float(book['levels'][1][0][0]) if book['levels'][1] else 0  # Ask price
            is_buy = True
        else:  # OPEN_SHORT
            execution_price = float(book['levels'][0][0][0]) if book['levels'][0] else 0  # Bid price
            is_buy = False

        if execution_price <= 0:
            logger.error("Invalid orderbook data")
            return False

        logger.info(f"Opening {side} position: {size_coins:.4f} {symbol} @ ${execution_price:.2f}")
        logger.info(f"TP: {tp_pct:.2f}%, SL: {sl_pct:.2f}%")

        try:
            # Place market order
            order = self.exchange.market_open(
                coin=symbol,
                is_buy=is_buy,
                sz=size_coins,
                px=None  # Market order
            )

            logger.success(f"Position opened: {order}")

            # Calculate TP/SL prices
            if is_buy:  # Long
                tp_price = execution_price * (1 + tp_pct / 100)
                sl_price = execution_price * (1 - sl_pct / 100)
            else:  # Short
                tp_price = execution_price * (1 - tp_pct / 100)
                sl_price = execution_price * (1 + sl_pct / 100)

            logger.info(f"Calculated TP: ${tp_price:.2f}, SL: ${sl_price:.2f}")

            # Place TP/SL orders
            # Note: Implementation depends on Hyperliquid API specifics
            # This is a placeholder - consult Hyperliquid docs for exact implementation
            try:
                self._place_tp_sl_orders(symbol, size_coins, tp_price, sl_price, is_buy)
            except Exception as e:
                logger.error(f"Failed to place TP/SL orders: {e}")
                logger.warning("Position opened but TP/SL not set - MANUAL MONITORING REQUIRED")

            return True

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return False

    def _place_tp_sl_orders(self, symbol: str, size: float, tp_price: float, sl_price: float, is_buy: bool):
        """
        Place take-profit and stop-loss orders

        Note: This implementation depends on Hyperliquid's order types.
        You may need to use limit orders or trigger orders.
        Consult Hyperliquid API documentation for exact implementation.
        """

        # Take Profit (limit order on opposite side)
        try:
            tp_order = self.exchange.order(
                coin=symbol,
                is_buy=not is_buy,  # Opposite side to close
                sz=size,
                limit_px=tp_price,
                order_type={"limit": {"tif": "Gtc"}},  # Good-til-canceled
                reduce_only=True
            )
            logger.info(f"TP order placed: {tp_order}")
        except Exception as e:
            logger.error(f"Failed to place TP order: {e}")

        # Stop Loss (trigger order)
        try:
            sl_order = self.exchange.order(
                coin=symbol,
                is_buy=not is_buy,  # Opposite side to close
                sz=size,
                limit_px=sl_price,
                order_type={"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}},
                reduce_only=True
            )
            logger.info(f"SL order placed: {sl_order}")
        except Exception as e:
            logger.error(f"Failed to place SL order: {e}")

    def _close_position(self, symbol: str) -> bool:
        """Close existing position"""

        position = self.get_current_position(symbol)
        if not position:
            logger.warning("No position to close")
            return False

        try:
            is_buy = (position['side'] == 'SHORT')  # Reverse to close

            logger.info(f"Closing {position['side']} position: {position['size']} {symbol}")

            # Cancel any existing TP/SL orders first
            try:
                self.exchange.cancel_all(symbol)
                logger.info("Cancelled existing orders")
            except Exception as e:
                logger.warning(f"Failed to cancel orders: {e}")

            order = self.exchange.market_close(
                coin=symbol,
                sz=position['size']
            )

            logger.success(f"Position closed: {order}")
            logger.info(f"Final P&L: {position['pnl_pct']:.2f}% (${position['pnl_usd']:.2f})")

            return True

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False


class LLMTradingAgent:
    """Main agent orchestrator with improved safety"""

    def __init__(self,
                 symbol: str = "BTC",
                 position_size_pct: float = 10.0,  # Changed to percentage of account
                 check_interval_seconds: int = 300,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4o",
                 max_daily_loss_pct: float = 5.0,
                 max_spread_bps: float = 50.0):

        self.symbol = symbol
        self.position_size_pct = position_size_pct
        self.check_interval = check_interval_seconds

        # Initialize components
        self.data_collector = MarketDataCollector(testnet=True)
        self.llm_decision = LLMTradingDecision(model=llm_model, provider=llm_provider)
        self.executor = HyperliquidExecutor(testnet=True)
        self.risk_manager = RiskManager(
            max_daily_loss_pct=max_daily_loss_pct,
            max_position_size_pct=position_size_pct,
            max_spread_bps=max_spread_bps
        )
        self.performance_tracker = PerformanceTracker()

        logger.success("LLM Trading Agent (IMPROVED) initialized")
        logger.info(f"Symbol: {symbol}, Position Size: {position_size_pct}% of account")

    def run_once(self) -> Dict:
        """Run single iteration with enhanced safety checks"""

        logger.info(f"\n{'='*80}\nAgent Iteration - {datetime.now()}\n{'='*80}")

        try:
            # 1. Get account balance
            account_balance = self.executor.get_account_balance()
            if account_balance <= 0:
                logger.error("Cannot retrieve account balance")
                return {'error': 'No account balance'}

            self.performance_tracker.record_equity(account_balance)

            # 2. Check risk limits
            if not self.risk_manager.check_account_balance(account_balance):
                return {'error': 'Insufficient balance'}

            if not self.risk_manager.check_daily_loss_limit(account_balance):
                return {'error': 'Daily loss limit exceeded - CIRCUIT BREAKER ACTIVE'}

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
            if current_position:
                logger.info(f"Current position: {current_position['side']} {current_position['size']:.4f} @ ${current_position['entry_price']:.2f}, P&L: {current_position['pnl_pct']:.2f}%")

            # 6. Get LLM decision
            decision = self.llm_decision.get_trading_decision(market_data, current_position)

            # 7. Calculate position size
            execution_price = market_data['orderbook']['best_ask'] if decision['action'] == 'OPEN_LONG' else market_data['orderbook']['best_bid']
            position_size_coins = self.risk_manager.calculate_position_size(account_balance, execution_price)

            # 8. Execute trade
            executed = self.executor.execute_trade(
                symbol=self.symbol,
                decision=decision,
                position_size_coins=position_size_coins,
                risk_checks_passed=True
            )

            # 9. Track performance
            if executed and decision['action'] == 'CLOSE' and current_position:
                self.risk_manager.record_trade(current_position['pnl_usd'])
                self.performance_tracker.record_trade(current_position)

            result = {
                'timestamp': datetime.now().isoformat(),
                'account_balance': account_balance,
                'market_data': market_data,
                'decision': decision,
                'position_size_coins': position_size_coins,
                'executed': executed,
                'daily_pnl': self.risk_manager.daily_pnl_usd
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
                    self.performance_tracker.print_summary()

                # Log result
                if 'error' in result:
                    logger.error(f"Iteration error: {result['error']}")
                else:
                    logger.info(f"Iteration complete. Daily P&L: ${self.risk_manager.daily_pnl_usd:.2f}")

                logger.info(f"Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("\n" + "="*80)
                logger.info("Agent stopped by user")
                self.performance_tracker.print_summary()
                logger.info("="*80)
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Sleeping 60s before retry...")
                time.sleep(60)


# Health check endpoint for Railway
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress healthcheck logs
        pass

def start_health_server(port=8080):
    """Start health check HTTP server in background thread"""
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health check server started on port {port}")


if __name__ == "__main__":
    # Start health check server for Railway
    start_health_server(port=int(os.getenv('PORT', 8080)))
    
    # Improved agent with safety features
    agent = LLMTradingAgent(
        symbol="BTC",
        position_size_pct=10.0,  # 10% of account per trade
        check_interval_seconds=300,  # 5 minutes
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_daily_loss_pct=5.0,  # Circuit breaker at 5% daily loss
        max_spread_bps=50.0  # Max 0.5% spread
    )

    # Check run mode
    run_mode = os.getenv("RUN_MODE", "once").lower()
    result = None # Initialize result for potential use outside the if/else

    if run_mode == "continuous":
        logger.info("Starting agent in CONTINUOUS mode")
        agent.run_loop()
    else:
        logger.info("Starting agent in SINGLE RUN mode (set RUN_MODE=continuous to run loop)")
        logger.info("Running single test iteration...")
        result = agent.run_once()

        print("\n" + "="*80)
        print("RESULT:")
        print(json.dumps(result, indent=2, default=str))
        print("="*80)

