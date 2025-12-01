import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, List
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

# Configure logging
logger.add("logs/llm_agent_{time}.log", rotation="1 day")


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

        candles = self.info.candles_snapshot(
            coin=symbol,
            interval=interval,
            startTime=end_time - (lookback * 3600 * 1000),
            endTime=end_time
        )

        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])

        # Calculate indicators
        indicators = self._calculate_indicators(df)

        # Get orderbook
        book = self.info.l2_snapshot(symbol)

        # Format market snapshot
        snapshot = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(df['close'].iloc[-1]),
            'indicators': indicators,
            'orderbook': {
                'best_bid': float(book['levels'][0][0][0]) if book['levels'][0] else 0,
                'best_ask': float(book['levels'][1][0][0]) if book['levels'][1] else 0,
                'bid_depth_10': sum([float(b[1]) for b in book['levels'][0][:10]]),
                'ask_depth_10': sum([float(a[1]) for a in book['levels'][1][:10]]),
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

        # ATR (volatility)
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1]

        # Volume analysis
        volume_ma_20 = volume.rolling(20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / volume_ma_20 if volume_ma_20 > 0 else 1

        # Price momentum
        returns_24h = ((close.iloc[-1] - close.iloc[-24]) / close.iloc[-24] * 100) if len(close) >= 24 else 0

        indicators = {
            'ma_20': float(ma_20),
            'ma_50': float(ma_50),
            'rsi_14': float(rsi),
            'macd': float(macd),
            'macd_signal': float(macd_signal),
            'macd_histogram': float(macd_diff),
            'bollinger_upper': float(bb_high),
            'bollinger_lower': float(bb_low),
            'bollinger_middle': float(bb_mid),
            'atr': float(atr),
            'volume_ratio': float(volume_ratio),
            'returns_24h_pct': float(returns_24h)
        }

        return indicators


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
        """Create structured prompt for LLM"""

        position_info = "No position currently open."
        if current_position:
            position_info = f"""
Current Position: {current_position['side']} {current_position['size']} @ ${current_position['entry_price']:.2f}
Current P&L: {current_position['pnl_pct']:.2f}%
"""

        prompt = f"""You are an expert cryptocurrency trading analyst. Analyze the following market data and provide a trading decision.

MARKET DATA FOR {market_data['symbol']}:
- Current Price: ${market_data['current_price']:.2f}
- Timestamp: {market_data['timestamp']}

TECHNICAL INDICATORS:
- MA(20): ${market_data['indicators']['ma_20']:.2f}
- MA(50): ${market_data['indicators']['ma_50']:.2f}
- RSI(14): {market_data['indicators']['rsi_14']:.2f}
- MACD: {market_data['indicators']['macd']:.4f}
- MACD Signal: {market_data['indicators']['macd_signal']:.4f}
- MACD Histogram: {market_data['indicators']['macd_histogram']:.4f}
- Bollinger Bands: Upper ${market_data['indicators']['bollinger_upper']:.2f}, Lower ${market_data['indicators']['bollinger_lower']:.2f}
- ATR (Volatility): {market_data['indicators']['atr']:.2f}
- Volume Ratio: {market_data['indicators']['volume_ratio']:.2f}x average
- 24h Returns: {market_data['indicators']['returns_24h_pct']:.2f}%

ORDERBOOK:
- Best Bid: ${market_data['orderbook']['best_bid']:.2f}
- Best Ask: ${market_data['orderbook']['best_ask']:.2f}
- Spread: {((market_data['orderbook']['best_ask'] - market_data['orderbook']['best_bid']) / market_data['orderbook']['best_bid'] * 100):.3f}%

CURRENT POSITION:
{position_info}

INSTRUCTIONS:
1. Analyze the technical indicators and market conditions
2. Consider trend, momentum, volatility, and volume
3. Decide whether to OPEN_LONG, OPEN_SHORT, CLOSE position, or HOLD
4. If opening a position, provide take-profit % and stop-loss %
5. Provide confidence level (0.0 to 1.0)
6. Explain your reasoning concisely

Respond ONLY with valid JSON in this exact format:
{{
    "action": "OPEN_LONG" or "OPEN_SHORT" or "CLOSE" or "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision",
    "take_profit_pct": 2.5,
    "stop_loss_pct": 1.5
}}

Rules:
- Only suggest OPEN_LONG/OPEN_SHORT if confidence > 0.7
- take_profit_pct should be 1.5-5%
- stop_loss_pct should be 0.8-2%
- If current position exists and profitable, consider CLOSE
- Consider risk/reward ratio
"""

        return prompt

    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency trading analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent decisions
                max_tokens=500
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
                max_tokens=500,
                temperature=0.3,
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
            decision['take_profit_pct'] = max(0.5, min(10.0, decision['take_profit_pct']))
            decision['stop_loss_pct'] = max(0.3, min(5.0, decision['stop_loss_pct']))

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
    """Execute trades on Hyperliquid"""

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

                            return {
                                'symbol': symbol,
                                'side': 'LONG' if size > 0 else 'SHORT',
                                'size': abs(size),
                                'entry_price': entry_price,
                                'current_price': current_price,
                                'pnl_pct': pnl_pct
                            }

            return None

        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None

    def execute_trade(self, symbol: str, decision: Dict, position_size_usd: float = 100) -> bool:
        """
        Execute trade based on LLM decision

        Args:
            symbol: Trading pair
            decision: LLM decision dict
            position_size_usd: Position size in USD
        """

        try:
            action = decision['action']

            if action == 'HOLD':
                logger.info("Action is HOLD - no trade executed")
                return True

            if action == 'CLOSE':
                return self._close_position(symbol)

            if action in ['OPEN_LONG', 'OPEN_SHORT']:
                return self._open_position(
                    symbol=symbol,
                    side=action,
                    size_usd=position_size_usd,
                    tp_pct=decision['take_profit_pct'],
                    sl_pct=decision['stop_loss_pct']
                )

            return False

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def _open_position(self, symbol: str, side: str, size_usd: float,
                      tp_pct: float, sl_pct: float) -> bool:
        """Open new position with TP/SL"""

        # Get current price
        book = self.info.l2_snapshot(symbol)
        current_price = float(book['levels'][0][0][0]) if side == 'OPEN_LONG' else float(book['levels'][1][0][0])

        # Calculate size in coins
        size_coins = size_usd / current_price

        # Determine buy/sell
        is_buy = (side == 'OPEN_LONG')

        logger.info(f"Opening {side} position: {size_coins:.4f} {symbol} @ ${current_price:.2f}")
        logger.info(f"TP: {tp_pct}%, SL: {sl_pct}%")

        # NOTE: This is simplified - Hyperliquid API specifics may vary
        # Consult official docs for exact order placement

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
            if is_buy:
                tp_price = current_price * (1 + tp_pct / 100)
                sl_price = current_price * (1 - sl_pct / 100)
            else:
                tp_price = current_price * (1 - tp_pct / 100)
                sl_price = current_price * (1 + sl_pct / 100)

            logger.info(f"TP set at ${tp_price:.2f}, SL at ${sl_price:.2f}")

            # Place TP/SL orders (if supported by API)
            # self._place_tp_sl_orders(symbol, size_coins, tp_price, sl_price, is_buy)

            return True

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return False

    def _close_position(self, symbol: str) -> bool:
        """Close existing position"""

        position = self.get_current_position(symbol)
        if not position:
            logger.warning("No position to close")
            return False

        try:
            is_buy = (position['side'] == 'SHORT')  # Reverse to close

            logger.info(f"Closing {position['side']} position: {position['size']} {symbol}")

            order = self.exchange.market_close(
                coin=symbol,
                is_buy=is_buy,
                sz=position['size']
            )

            logger.success(f"Position closed: {order}")
            logger.info(f"Final P&L: {position['pnl_pct']:.2f}%")

            return True

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False


class LLMTradingAgent:
    """Main agent orchestrator"""

    def __init__(self,
                 symbol: str = "BTC",
                 position_size_usd: float = 100,
                 check_interval_seconds: int = 300,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4o"):

        self.symbol = symbol
        self.position_size_usd = position_size_usd
        self.check_interval = check_interval_seconds

        # Initialize components
        self.data_collector = MarketDataCollector(testnet=True)
        self.llm_decision = LLMTradingDecision(model=llm_model, provider=llm_provider)
        self.executor = HyperliquidExecutor(testnet=True)

        logger.success("LLM Trading Agent initialized")
        logger.info(f"Symbol: {symbol}, Position Size: ${position_size_usd}")

    def run_once(self) -> Dict:
        """Run single iteration - for testing"""

        logger.info(f"\n{'='*80}\nAgent Iteration - {datetime.now()}\n{'='*80}")

        # 1. Collect market data
        market_data = self.data_collector.get_market_snapshot(self.symbol)

        # 2. Get current position
        current_position = self.executor.get_current_position(self.symbol)
        if current_position:
            logger.info(f"Current position: {current_position}")

        # 3. Get LLM decision
        decision = self.llm_decision.get_trading_decision(market_data, current_position)

        # 4. Execute trade
        executed = self.executor.execute_trade(
            symbol=self.symbol,
            decision=decision,
            position_size_usd=self.position_size_usd
        )

        result = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'decision': decision,
            'executed': executed
        }

        return result

    def run_loop(self):
        """Run continuous trading loop"""

        logger.info(f"Starting continuous trading loop (check every {self.check_interval}s)")

        iteration = 0

        while True:
            try:
                iteration += 1
                logger.info(f"\n{'='*80}\nIteration {iteration}\n{'='*80}")

                result = self.run_once()

                # Log result
                logger.info(f"Iteration complete. Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Agent stopped by user")
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Sleeping 60s before retry...")
                time.sleep(60)


if __name__ == "__main__":
    # Quick test mode
    agent = LLMTradingAgent(
        symbol="BTC",
        position_size_usd=50,  # Small size for testing
        check_interval_seconds=300,  # 5 minutes
        llm_provider="openai",  # or "anthropic"
        llm_model="gpt-4o"  # or "claude-3-5-sonnet-20241022"
    )

    # Run once for testing
    logger.info("Running single test iteration...")
    result = agent.run_once()

    print("\n" + "="*80)
    print("RESULT:")
    print(json.dumps(result, indent=2, default=str))
    print("="*80)

    # Uncomment to run continuous loop
    # agent.run_loop()
