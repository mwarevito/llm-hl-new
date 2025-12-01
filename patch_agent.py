import os
import sys

TARGET_FILE = "/Users/tony/.claude-worktrees/smart-council-strategy/unruffled-hawking/agent_improved.py"

with open(TARGET_FILE, 'r') as f:
    content = f.read()

# 1. Add imports
if "import json" not in content:
    content = content.replace("import time", "import time\nimport json")

# 2. Add state file constant
if "STATE_FILE = " not in content:
    content = content.replace("class RiskManager:", "STATE_FILE = 'risk_state.json'\n\nclass RiskManager:")

# 3. Add load_state and save_state methods to RiskManager
risk_manager_methods = """
    def load_state(self):
        \"\"\"Load daily PnL state from file\"\"\"
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
        \"\"\"Save daily PnL state to file\"\"\"
        try:
            state = {
                'daily_pnl_usd': self.daily_pnl_usd,
                'last_reset_time': self.last_reset_time.isoformat()
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")
"""

if "def load_state(self):" not in content:
    # Insert before check_account_balance
    content = content.replace("    def check_account_balance", risk_manager_methods + "\n    def check_account_balance")

# 4. Call load_state in __init__
if "self.load_state()" not in content:
    content = content.replace("self.max_spread_bps = max_spread_bps", "self.max_spread_bps = max_spread_bps\n        self.load_state()")

# 5. Call save_state in record_trade
if "self.save_state()" not in content and "def record_trade" in content:
    # We need to find the end of record_trade method to append save_state
    # This is a bit tricky with string replacement, let's look for the log line
    content = content.replace("logger.info(f\"Daily P&L updated: ${self.daily_pnl_usd:.2f}\")", "logger.info(f\"Daily P&L updated: ${self.daily_pnl_usd:.2f}\")\n        self.save_state()")

# 6. Call save_state in check_daily_loss_limit (reset logic)
if "self.save_state()" not in content and "def check_daily_loss_limit" in content:
    content = content.replace("logger.info(\"New day detected - resetting daily P&L\")", "logger.info(\"New day detected - resetting daily P&L\")\n            self.save_state()")

with open(TARGET_FILE, 'w') as f:
    f.write(content)

print("Successfully patched agent_improved.py")
