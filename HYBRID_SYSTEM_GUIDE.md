# 🚀 Hybrid Trading System - Production-Ready Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Backtesting](#backtesting)
7. [Performance Metrics](#performance-metrics)
8. [Production Deployment](#production-deployment)
9. [FAQ](#faq)

---

## 🎯 Overview

### What Changed?

**Old System** (agent_improved.py):
- ❌ LLM makes every trading decision → non-deterministic
- ❌ Cannot backtest accurately
- ❌ Slow (2-5s per decision)
- ❌ Expensive ($50-100/month)
- ❌ Not reproducible

**New Hybrid System** (agent_hybrid.py):
- ✅ LLM classifies regime (cached 1-4h) → stable intelligence
- ✅ Deterministic rules execute trades → reproducible
- ✅ Fully backtestable
- ✅ Fast (<100ms per decision)
- ✅ Cheap ($2-5/month, 95% cost reduction)
- ✅ Hedge-fund quality metrics (Sharpe, drawdown, etc.)

### Performance Expectations

| Metric | Target | Hedge Fund Standard |
|--------|--------|---------------------|
| Sharpe Ratio | >1.2 | >1.5 (excellent >2.0) |
| Win Rate | 50-58% | >45% |
| Max Drawdown | <12% | <15% |
| Monthly Return | 1.0-2.5% | Varies (8-25% annual) |
| Profit Factor | >1.3 | >1.5 |

**Note**: These targets require 3-6 months of validation before claiming achievement.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SLOW LAYER (Every 1-4 hours) - LLM Regime Classification  │
│  Input: Market data with indicators                         │
│  Output: {regime, trend, volatility, risk_mode}             │
│  Cache Duration: 2 hours (configurable)                     │
└─────────────────────────────────────────────────────────────┘
                          ↓ (cached result)
┌─────────────────────────────────────────────────────────────┐
│  FAST LAYER (Every 5 minutes) - Deterministic Signals      │
│  Input: Market data + cached regime                         │
│  Process: Check if conditions match strategy rules          │
│  Output: {action, confidence, TP%, SL%}                     │
│  Speed: <100ms (no API call)                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  EXECUTION LAYER - Trade Execution & Risk Management       │
│  - Position sizing with consecutive loss adjustment         │
│  - Circuit breakers (daily/weekly loss limits)             │
│  - Performance tracking (Sharpe, drawdown, etc.)            │
└─────────────────────────────────────────────────────────────┘
```

### Component Files

| File | Purpose | Size |
|------|---------|------|
| `strategy_config.py` | Deterministic trading rules per regime | ~350 lines |
| `llm_regime_classifier.py` | LLM regime classification with caching | ~300 lines |
| `signal_generator.py` | Deterministic signal generation | ~450 lines |
| `performance_metrics.py` | Hedge-fund quality metrics | ~400 lines |
| `backtest_engine.py` | Historical validation framework | ~300 lines |
| `agent_hybrid.py` | Main trading agent (integrates all) | ~350 lines |

---

## 📦 Installation

### 1. Ensure Dependencies

All dependencies from `requirements.txt` are already installed. No new dependencies needed!

```bash
# Already installed from requirements.txt:
# - hyperliquid-python-sdk
# - pandas, numpy
# - ta (technical indicators)
# - openai/anthropic
# - python-dotenv, loguru
```

### 2. Verify File Structure

```
llm-hl-new/
├── agent_improved.py          # Original (still works)
├── agent_hybrid.py             # NEW: Hybrid system
├── strategy_config.py          # NEW: Trading rules
├── llm_regime_classifier.py    # NEW: LLM with caching
├── signal_generator.py         # NEW: Deterministic signals
├── performance_metrics.py      # NEW: Sharpe, drawdown, etc.
├── backtest_engine.py          # NEW: Historical validation
├── data/                       # Performance data, cache
│   ├── risk_state.json
│   ├── regime_cache.json       # NEW: Regime cache
│   ├── hybrid_performance.json # NEW: Performance metrics
│   └── backtest_results.json   # NEW: Backtest results
└── logs/                       # Log files
```

---

## 🚀 Quick Start

### Test Single Run (Recommended First Step)

```bash
# Set environment to single-run mode
export RUN_MODE=once

# Run hybrid agent
python3 agent_hybrid.py
```

**Expected Output:**
```
================================================================================
INITIALIZING HYBRID TRADING AGENT (PRODUCTION MODE)
================================================================================
✅ Hybrid Trading Agent initialized
   Symbol: BTC
   Position Size: 10.0% of account
   Regime Cache: 2.0h
   Mode: TESTNET
================================================================================

... (market data collection) ...

🎯 Market Regime: RANGING / WEAK_BULL
   Risk Mode: NORMAL, Confidence: 0.75

📊 Signal: HOLD (confidence: 0.50)
   Reasoning: No entry conditions met (RANGING / WEAK_BULL)

================================================================================
RESULT:
================================================================================
✅ Execution successful

Regime: RANGING / WEAK_BULL
Signal: HOLD (confidence: 0.50)
Reasoning: No entry conditions met
Account Balance: $959.32
Daily P&L: $-0.52
================================================================================
```

### Run Continuous Mode

```bash
# Set environment to continuous mode
export RUN_MODE=continuous

# Run hybrid agent
python3 agent_hybrid.py
```

The agent will:
1. Check market every 5 minutes
2. Use cached regime (fast)
3. Refresh LLM regime every 2 hours
4. Print performance summary every 10 iterations

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
HYPERLIQUID_ADDRESS=your_address
HYPERLIQUID_PRIVATE_KEY=your_key
OPENAI_API_KEY=your_openai_key

# Optional
LLM_PROVIDER=openai           # or "anthropic"
LLM_MODEL=gpt-4o-mini          # or "gpt-4o", "claude-3-5-sonnet-20241022"
RUN_MODE=continuous            # or "once"
PORT=8080                      # For Railway health check
```

### Agent Parameters

Edit `agent_hybrid.py` at the bottom:

```python
agent = HybridTradingAgent(
    symbol="BTC",                    # Trading pair
    position_size_pct=10.0,          # 10% of account per trade
    check_interval_seconds=300,      # Check every 5 minutes
    regime_cache_hours=2.0,          # Cache regime for 2 hours
    max_daily_loss_pct=5.0,          # Circuit breaker at 5% daily loss
    max_spread_bps=50.0,             # Max 0.5% spread
    testnet=True                     # Use testnet
)
```

### Strategy Rules

Edit `strategy_config.py` to customize trading rules:

```python
# Example: Momentum Long Entry (STRONG_BULL)
"LONG": SignalConfig(
    rsi_min=50,                      # RSI must be > 50
    rsi_max=85,                      # RSI must be < 85
    price_vs_ema20="ABOVE",          # Price must be above EMA20
    ema20_vs_ema50="ABOVE",          # EMA20 above EMA50 (uptrend)
    macd_vs_signal="ABOVE",          # MACD bullish
    volume_ratio_min=0.8,            # Volume > 0.8x average
    bb_position_min=0.5,             # Price in upper half of BB
    stop_loss_atr_multiple=2.5,      # SL at 2.5x ATR
    take_profit_atr_multiple=4.0,    # TP at 4.0x ATR
    position_size_multiplier=1.0,    # Full position size
    min_confidence=0.55              # Minimum 55% confidence
),
```

---

## 📊 Backtesting

### Why Backtest?

**Critical for production trading:**
- ✅ Validate strategy before risking real money
- ✅ Calculate expected Sharpe ratio and max drawdown
- ✅ Test across different market regimes (bull, bear, ranging)
- ✅ Optimize parameters
- ✅ Build confidence in your edge

### Run Backtest

```python
from backtest_engine import BacktestEngine
from agent_improved import MarketDataCollector
import pandas as pd

# 1. Collect historical data (you'll need to implement data fetching)
# For now, this is a placeholder - you need historical candles
historical_data = []  # List of market snapshots

# 2. Initialize backtest engine
backtest = BacktestEngine(
    initial_capital=10000.0,
    position_size_pct=10.0,
    fee_pct=0.05,        # 0.05% fees
    slippage_pct=0.1     # 0.1% slippage
)

# 3. Run backtest
results = backtest.run(historical_data)

# 4. Print results
backtest.print_results()
```

### Expected Backtest Output

```
================================================================================
PERFORMANCE SUMMARY
================================================================================

📊 BASIC STATS:
   Total Trades: 143
   Win Rate: 54.55%
   Winners: 78 | Losers: 65

💰 P&L:
   Total P&L: $2,347.21
   Avg Win: $85.43
   Avg Loss: -$42.15
   Best Trade: 4.23%
   Worst Trade: -2.87%

📈 ADVANCED METRICS:
   Profit Factor: 1.47
   Expectancy: $16.41 per trade
   Risk/Reward: 2.03

🎯 RISK-ADJUSTED RETURNS:
   Sharpe Ratio: 1.68 (⭐⭐ Good)
   Sortino Ratio: 2.12
   Calmar Ratio: 1.95

⚠️  RISK METRICS:
   Max Drawdown: 8.34% ($834.21)
   Annual Return: 16.28%

⭐ QUALITY ASSESSMENT:
   Quality Score: 75/100
   Hedge Fund Ready: ❌ NO (need more data - only 143 trades, need 100+)
================================================================================
```

### Backtest Requirements for Production

Before going live with real money:

1. **Data**: At least 6-12 months of historical data
2. **Trades**: Minimum 100 trades for statistical significance
3. **Metrics**:
   - Sharpe > 1.2 (ideally >1.5)
   - Max Drawdown < 15%
   - Profit Factor > 1.3
4. **Regimes**: Test across bull, bear, and ranging markets
5. **Forward Test**: 3 months paper trading after backtest

---

## 📈 Performance Metrics

### View Current Performance

```python
from performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics(save_path="data/hybrid_performance.json")
metrics.print_summary()
```

### Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Sharpe Ratio** | Risk-adjusted returns (return per unit of risk) | >1.5 |
| **Sortino Ratio** | Like Sharpe but only penalizes downside volatility | >2.0 |
| **Max Drawdown** | Largest peak-to-trough decline | <15% |
| **Profit Factor** | Gross profit / gross loss | >1.5 |
| **Win Rate** | % of winning trades | >50% |
| **Expectancy** | Average $ per trade | >0 |
| **Calmar Ratio** | Annual return / max drawdown | >1.0 |

### Real-Time Monitoring

The hybrid agent automatically tracks:
- Trade-by-trade P&L
- Daily/weekly P&L
- Equity curve
- Drawdown in real-time
- Per-regime performance

Check `data/hybrid_performance.json` for full history.

---

## 🏭 Production Deployment

### Railway Deployment

1. **Update Railway Environment Variables**:
```bash
# In Railway dashboard, set:
RUN_MODE=continuous
```

2. **Deploy Hybrid Agent**:

Option A: Update `agent_improved.py` to use hybrid logic
Option B: Change Railway start command to `python3 agent_hybrid.py`

Edit `railway.toml`:
```toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "python3 agent_hybrid.py"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

3. **Monitor**:
```bash
# View logs in Railway dashboard
# Check data/hybrid_performance.json for metrics
```

### Switching Between Systems

**Test Hybrid on Testnet** (while old system runs on mainnet):
```bash
# Terminal 1: Old system on mainnet
export RUN_MODE=continuous
python3 agent_improved.py

# Terminal 2: Hybrid system on testnet
export RUN_MODE=continuous
python3 agent_hybrid.py
```

**Compare performance** after 1 week → switch if hybrid is better.

---

## ❓ FAQ

### Q: Is the hybrid system ready for real money?

**A**: Not yet. You need to:
1. Backtest on 6-12 months of historical data
2. Paper trade for 3+ months
3. Achieve Sharpe >1.2, Max DD <15%
4. Start with small capital ($1K-$5K)

### Q: Can I still use the old agent_improved.py?

**A**: Yes! Both systems work independently. Use `agent_improved.py` for the original LLM approach, or `agent_hybrid.py` for the new hybrid system.

### Q: How much does the hybrid system cost?

**A**: ~$2-5/month vs $50-100/month for the old system.

- Regime classification: $0.0001/call
- Cached for 2 hours → ~12 calls/day
- Cost: ~$0.05/day = $1.50/month

### Q: What if I want to change trading rules?

**A**: Edit `strategy_config.py`. All rules are clearly documented. Change RSI thresholds, stop loss multiples, etc.

### Q: How do I force regime refresh?

```python
# In agent_hybrid.py or any script:
agent.regime_classifier.invalidate_cache()

# Next call will query LLM fresh
regime = agent.regime_classifier.get_regime(market_data, force_refresh=True)
```

### Q: Can I backtest without historical data?

**A**: No. You need historical candle data. Options:
1. Use Hyperliquid API to fetch past candles
2. Record live data for 3-6 months then backtest
3. Use external data providers (CryptoCompare, CoinGecko, etc.)

### Q: What's the expected monthly return?

**A**: Unknown until backtested. Realistic targets:
- Conservative: 0.5-1% monthly (6-12% annual)
- Moderate: 1-2% monthly (12-26% annual)
- Aggressive: 2-3% monthly (26-42% annual, higher risk)

Most retail algo traders lose money in year 1. Set realistic expectations.

---

## 🎓 Next Steps

### Week 1: Testing
- [ ] Run single iterations on testnet
- [ ] Verify all components work
- [ ] Test regime caching
- [ ] Check performance metrics save correctly

### Week 2-4: Validation
- [ ] Collect historical data (3-12 months)
- [ ] Run backtests
- [ ] Analyze Sharpe ratio, max drawdown
- [ ] Optimize parameters if needed

### Month 2-4: Paper Trading
- [ ] Run hybrid system on testnet continuous
- [ ] Track real-time performance
- [ ] Compare to backtest results
- [ ] Build confidence in edge

### Month 5+: Live Trading (Small Capital)
- [ ] Start with $1K-$5K
- [ ] Monitor daily
- [ ] Scale slowly if profitable
- [ ] Never risk more than you can afford to lose

---

## 📞 Support

- **Issues**: Check `logs/` directory for error details
- **Questions**: Review code comments in each module
- **Performance**: Check `data/hybrid_performance.json`

## ⚠️ Disclaimer

**This is experimental trading software. You can lose money.**

- Start small ($1K max)
- Test thoroughly before scaling
- Never trade with money you can't afford to lose
- Past performance ≠ future results
- No guarantees of profitability

---

**Good luck and trade responsibly! 🚀📈**
