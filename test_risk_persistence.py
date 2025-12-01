import json
import os
import sys
from datetime import datetime, timedelta

# Mock logger
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def success(self, msg): print(f"SUCCESS: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")

# Mock the agent_improved module to test RiskManager in isolation
# We'll just define the class here with the same logic for testing
STATE_FILE = 'risk_state.json'
logger = MockLogger()

class RiskManager:
    def __init__(self, max_daily_loss_pct=5.0, max_position_size_pct=10.0, max_spread_bps=50.0):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size_pct = max_position_size_pct
        self.max_spread_bps = max_spread_bps
        self.daily_pnl_usd = 0.0
        self.last_reset_time = datetime.now()
        self.load_state()

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
            state = {
                'daily_pnl_usd': self.daily_pnl_usd,
                'last_reset_time': self.last_reset_time.isoformat()
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def record_trade(self, pnl_usd):
        self.daily_pnl_usd += pnl_usd
        logger.info(f"Daily P&L updated: ${self.daily_pnl_usd:.2f}")
        self.save_state()

def test_persistence():
    print("--- Test 1: Save and Load ---")
    # Clean up
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    rm1 = RiskManager()
    rm1.record_trade(-50.0)  # Lost $50
    print(f"RM1 PnL: {rm1.daily_pnl_usd}")

    # Create new instance, should load state
    rm2 = RiskManager()
    print(f"RM2 PnL: {rm2.daily_pnl_usd}")
    
    if rm2.daily_pnl_usd == -50.0:
        print("PASS: State loaded correctly")
    else:
        print("FAIL: State not loaded")

    print("\n--- Test 2: Day Reset ---")
    # Manually modify file to be yesterday
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    state['last_reset_time'] = yesterday
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

    rm3 = RiskManager()
    print(f"RM3 PnL: {rm3.daily_pnl_usd}")
    
    if rm3.daily_pnl_usd == 0.0:
        print("PASS: State reset correctly for new day")
    else:
        print("FAIL: State not reset")

    # Cleanup
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

if __name__ == "__main__":
    test_persistence()
