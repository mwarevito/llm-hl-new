# Quick Start Guide - Improved Trading Agent

## ✅ All Critical Fixes Applied

The agent now has:
- ✅ Fixed initialization order
- ✅ Full state persistence
- ✅ Circuit breaker protection
- ✅ Proper risk management
- ✅ Data organization

---

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
cd /Users/tony/.claude-worktrees/smart-council-strategy/unruffled-hawking
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` file:
```bash
# LLM Provider (choose one)
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...

# Hyperliquid (Testnet)
HYPERLIQUID_ADDRESS=0x...
HYPERLIQUID_PRIVATE_KEY=0x...
```

### 3. Test Run (Single Iteration)
```bash
python3 agent_improved.py
```

This will:
- Fetch BTC market data
- Calculate indicators
- Get LLM decision
- Show result (no actual trade on testnet)

---

## ⚙️ Configuration

### Conservative Settings (Recommended):
```python
agent = LLMTradingAgent(
    symbol="BTC",
    position_size_pct=5.0,      # 5% of account per trade
    check_interval_seconds=300,  # Check every 5 minutes
    llm_provider="openai",
    llm_model="gpt-4o",
    max_daily_loss_pct=3.0,     # Stop after 3% daily loss
    max_spread_bps=30.0         # Max 0.3% spread
)
```

### Moderate Settings:
```python
agent = LLMTradingAgent(
    symbol="BTC",
    position_size_pct=10.0,     # 10% of account
    max_daily_loss_pct=5.0,     # Stop after 5% daily loss
    max_spread_bps=50.0         # Max 0.5% spread
)
```

---

## 🔄 Running Modes

### Mode 1: Single Test (Default)
```python
# In agent_improved.py
agent = LLMTradingAgent(...)
result = agent.run_once()
```

### Mode 2: Continuous Trading
```python
# In agent_improved.py (uncomment last line)
agent = LLMTradingAgent(...)
agent.run_loop()  # Runs continuously
```

Or run from command line:
```bash
python3 agent_improved.py --continuous
```

---

## 📊 Monitoring

### Check Logs
```bash
tail -f logs/llm_agent_improved_*.log
```

### Check Daily P&L State
```bash
cat data/risk_state.json
```

Example output:
```json
{
  "daily_pnl_usd": -15.50,
  "last_reset_time": "2025-12-01T10:30:00"
}
```

### Performance Summary
The agent prints performance summary every 10 iterations:
```
PERFORMANCE SUMMARY
==================
Total Trades: 5
Win Rate: 60.00%
Total P&L: $25.50
Average P&L: 1.25%
Best Trade: 3.50%
Worst Trade: -2.10%
```

---

## 🛡️ Safety Features

### 1. Circuit Breaker
Automatically stops trading when daily loss exceeds limit:
```
🛑 CIRCUIT BREAKER: Daily loss limit exceeded (-5.2% < -5.0%)
```

### 2. Spread Filter
Won't trade if spread is too wide:
```
⚠️ Spread too wide: 75.00 bps > 50.00 bps
```

### 3. Balance Check
Verifies sufficient balance before trading:
```
❌ Insufficient account balance: $50.00 < $100.00
```

### 4. Position Check
Prevents opening multiple positions:
```
⚠️ Position already exists: LONG. Skipping new position.
```

---

## 📁 Directory Structure

```
/unruffled-hawking/
├── agent_improved.py          # Main bot (IMPROVED VERSION)
├── agent.py                   # Original (has bugs)
├── data/
│   └── risk_state.json        # Daily P&L tracking
├── logs/
│   └── llm_agent_improved_*.log
├── backtest/                  # Future: backtesting data
├── .env                       # Your API keys (KEEP PRIVATE)
├── requirements.txt
├── IMPROVEMENTS.md            # Detailed changes doc
├── FIXES_APPLIED.md          # Critical fixes doc
└── QUICK_START.md            # This file
```

---

## 🧪 Testing Checklist

Before running with real money:

### Testnet Testing (2-4 weeks minimum):
- [ ] Run single iteration successfully
- [ ] Run continuous mode for 24 hours
- [ ] Verify TP/SL orders appear on Hyperliquid UI
- [ ] Test circuit breaker (simulate losses)
- [ ] Test bot restart mid-day (state persists)
- [ ] Monitor all trades and calculate metrics
- [ ] Verify spread filter works
- [ ] Check position sizing is correct

### Performance Analysis:
- [ ] Win rate > 50%
- [ ] Risk/reward ratio > 1.5
- [ ] Profitable after LLM API costs
- [ ] Max drawdown acceptable
- [ ] Daily loss limit never exceeded

---

## 🔍 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Failed to get market data"
- Check Hyperliquid API status
- Verify symbol name (use "BTC" not "BTCUSD")

### "OpenAI API error"
- Verify API key in .env
- Check you have credits

### "Failed to load risk state"
- Normal on first run (no state file exists yet)
- Creates `data/risk_state.json` on first trade

### Circuit breaker won't reset
- Delete `data/risk_state.json` to manually reset
- Or wait until next day (auto-resets at midnight)

---

## 📞 What to Do If...

### Bot crashes mid-trading:
✅ **Safe** - State is saved after each trade
- Restart bot
- It will load previous P&L from `data/risk_state.json`
- Circuit breaker limit still enforced

### Hit daily loss limit:
✅ **Protected** - Trading stopped automatically
- Bot will reject all new trades today
- Resets automatically at midnight
- Or delete `data/risk_state.json` to reset manually (use with caution!)

### TP/SL not being placed:
⚠️ **Check Hyperliquid API**
- Verify order types supported
- May need to adjust `_place_tp_sl_orders()` function
- Check Hyperliquid documentation for exact API format

### LLM making bad decisions:
🔧 **Tune the prompt**
- Edit `_create_prompt()` in agent_improved.py
- Adjust confidence threshold (default 0.75)
- Add more context or constraints

---

## 💰 Cost Estimation

### LLM API Costs:
- **GPT-4o**: ~$0.02 per decision
- **5-minute intervals**: 288 calls/day
- **Daily cost**: ~$5.76
- **Monthly cost**: ~$172

### Break-even calculation:
```
Need to profit > API costs to be worthwhile
If account = $10,000
Monthly return needed > 1.72% just to cover API costs
```

**Recommendation**: Start with hourly checks (24 calls/day = $0.48/day) to reduce costs while testing.

---

## 🎯 Next Steps

1. **Run tests** (see Testing Checklist above)
2. **Monitor for 2-4 weeks** on testnet
3. **Verify TP/SL orders** work on exchange
4. **Calculate metrics** (win rate, Sharpe ratio, max drawdown)
5. **Consider adding**:
   - Multi-timeframe analysis
   - Telegram notifications
   - Backtesting system

---

## ⚠️ Important Reminders

1. **Never skip testnet testing** - Run for weeks before live trading
2. **API costs add up** - Factor into profitability calculations
3. **LLM confidence is not true probability** - Don't over-trust it
4. **Circuit breaker is your friend** - Don't disable it
5. **TP/SL must work** - Verify on exchange UI before trusting
6. **Start small** - Use 5% position sizes initially
7. **Monitor daily** - Check logs and performance metrics

---

## 📚 Additional Resources

- **IMPROVEMENTS.md** - Detailed comparison of all changes
- **FIXES_APPLIED.md** - Critical bugs fixed
- **Hyperliquid Docs**: https://hyperliquid.gitbook.io/
- **OpenAI Pricing**: https://openai.com/pricing
- **Anthropic Pricing**: https://anthropic.com/pricing

---

**Good luck, and trade safely! 🚀**

Remember: This is still experimental. Never risk more than you can afford to lose.
