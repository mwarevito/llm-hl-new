# LLM Trading Agent for Hyperliquid

A trading agent that uses LLMs (GPT-4 or Claude) to make trading decisions based on market data and technical indicators.

## How It Works

1. **Market Data Collection**: Fetches real-time market data from Hyperliquid (OHLCV, orderbook)
2. **Technical Analysis**: Calculates indicators (MA, RSI, MACD, Bollinger Bands, ATR, Volume)
3. **LLM Decision**: Sends data to GPT-4 or Claude for trading decision
4. **Execution**: Executes trades with TP/SL on Hyperliquid

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Choose your LLM provider
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...

# Hyperliquid credentials
HYPERLIQUID_ADDRESS=0x...
HYPERLIQUID_PRIVATE_KEY=0x...
```

### 3. Run Single Test

Test the agent with a single iteration (no actual trades on testnet):

```bash
python agent.py
```

This will:
- Fetch current BTC market data
- Calculate technical indicators
- Get LLM trading decision
- Show the result (without executing)

### 4. Run Continuous Mode

Uncomment the last line in `agent.py`:

```python
# agent.run_loop()  # Uncomment this
```

Then run:

```bash
python agent.py
```

## Configuration

Edit the `__main__` section in `agent.py`:

```python
agent = LLMTradingAgent(
    symbol="BTC",                    # Trading pair
    position_size_usd=50,            # Position size in USD
    check_interval_seconds=300,      # Check every 5 minutes
    llm_provider="openai",           # "openai" or "anthropic"
    llm_model="gpt-4o"               # or "claude-3-5-sonnet-20241022"
)
```

## Components

### MarketDataCollector
- Fetches market data from Hyperliquid
- Calculates technical indicators
- Returns comprehensive market snapshot

### LLMTradingDecision
- Creates structured prompt with market data
- Queries LLM (OpenAI or Anthropic)
- Parses and validates JSON response

### HyperliquidExecutor
- Executes trades on Hyperliquid
- Manages positions and orders
- Handles TP/SL (take profit / stop loss)

### LLMTradingAgent
- Orchestrates all components
- Runs trading loop
- Handles logging and error recovery

## LLM Decision Format

The LLM returns decisions in this format:

```json
{
    "action": "OPEN_LONG",
    "confidence": 0.85,
    "reasoning": "Strong bullish momentum with RSI at 65...",
    "take_profit_pct": 2.5,
    "stop_loss_pct": 1.5
}
```

Actions:
- `OPEN_LONG`: Open long position
- `OPEN_SHORT`: Open short position
- `CLOSE`: Close current position
- `HOLD`: Do nothing

## Safety Features

- Runs on testnet by default
- Low confidence threshold (>0.7) for opening positions
- Clamped TP/SL ranges (0.5-10% TP, 0.3-5% SL)
- Error handling and retry logic
- Comprehensive logging

## Logs

Logs are saved to `logs/llm_agent_{timestamp}.log` with rotation.

## Switching to Mainnet

**WARNING**: Only use mainnet when you're confident in the strategy!

Change in agent initialization:

```python
self.data_collector = MarketDataCollector(testnet=False)
self.executor = HyperliquidExecutor(testnet=False)
```

## Customization

### Add More Indicators

Edit `MarketDataCollector._calculate_indicators()`:

```python
# Add EMA
ema_12 = close.ewm(span=12).mean().iloc[-1]
indicators['ema_12'] = float(ema_12)
```

### Modify LLM Prompt

Edit `LLMTradingDecision._create_prompt()` to change trading strategy.

### Change Decision Logic

Edit `LLMTradingDecision._parse_response()` for custom validation.

## Known Limitations

- Simplified TP/SL implementation (check Hyperliquid docs for advanced order types)
- No position sizing based on account balance
- No risk management across multiple positions
- Single timeframe analysis (1h candles)

## Troubleshooting

### "Failed to get market data"
- Check Hyperliquid API status
- Verify symbol name (use "BTC" not "BTCUSD")

### "OpenAI API error"
- Verify API key is valid
- Check you have credits

### "Failed to open position"
- Ensure wallet has funds on testnet
- Check Hyperliquid API documentation for order format

## Disclaimer

This is a proof of concept for educational purposes. Trading cryptocurrencies carries significant risk. Always test thoroughly on testnet before using real funds. The authors are not responsible for any financial losses.

## License

MIT
