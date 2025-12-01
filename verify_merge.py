#!/usr/bin/env python3
"""
Verification script for merged agent_improved.py
Run this to verify all features are working correctly
"""

import os
import sys
from agent_improved import RiskManager, LLMConfig

def test_state_persistence():
    """Test that state persistence works"""
    print("🧪 Test 1: State Persistence")

    # Clean up first
    if os.path.exists('data/risk_state.json'):
        os.remove('data/risk_state.json')

    # Create manager and record trade
    rm1 = RiskManager()
    rm1.record_trade(-25.0)

    # Create new instance - should load state
    rm2 = RiskManager()

    if rm2.daily_pnl_usd == -25.0:
        print("   ✅ PASS - State persisted correctly")
        return True
    else:
        print(f"   ❌ FAIL - Expected -25.0, got {rm2.daily_pnl_usd}")
        return False

def test_llm_config():
    """Test that LLMConfig works"""
    print("\n🧪 Test 2: LLMConfig Class")

    try:
        config = LLMConfig()

        # Just check that it initializes and has the expected attributes
        if hasattr(config, 'model') and hasattr(config, 'temperature') and hasattr(config, 'provider'):
            print(f"   ✅ PASS - LLMConfig works correctly (model={config.model}, temp={config.temperature})")
            return True
        else:
            print(f"   ❌ FAIL - Missing expected attributes")
            return False
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        return False

def test_llm_config_from_env():
    """Test that LLMConfig loads from environment"""
    print("\n🧪 Test 3: LLMConfig from Environment")

    try:
        os.environ['LLM_MODEL'] = 'test-model'
        os.environ['LLM_TEMPERATURE'] = '0.5'

        config = LLMConfig.from_env()

        if config.model == 'test-model' and config.temperature == 0.5:
            print("   ✅ PASS - Environment loading works")
            return True
        else:
            print(f"   ❌ FAIL - Expected test-model/0.5, got {config.model}/{config.temperature}")
            return False
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        return False
    finally:
        # Clean up env vars
        os.environ.pop('LLM_MODEL', None)
        os.environ.pop('LLM_TEMPERATURE', None)

def test_imports():
    """Test that all required modules are imported"""
    print("\n🧪 Test 4: Required Imports")

    try:
        import agent_improved

        # Check for key imports
        required = ['RiskManager', 'LLMConfig', 'MarketDataCollector',
                   'LLMTradingDecision', 'HyperliquidExecutor', 'LLMTradingAgent']

        for item in required:
            if not hasattr(agent_improved, item):
                print(f"   ❌ FAIL - Missing: {item}")
                return False

        print("   ✅ PASS - All classes available")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        return False

def test_directories():
    """Test that required directories exist"""
    print("\n🧪 Test 5: Directory Structure")

    required_dirs = ['data', 'logs', 'backtest']

    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"   ❌ FAIL - Missing directory: {dir_name}")
            return False

    print("   ✅ PASS - All directories exist")
    return True

def main():
    print("="*60)
    print("🔍 VERIFICATION SCRIPT FOR MERGED AGENT")
    print("="*60)
    print()

    # Run all tests
    results = []
    results.append(test_state_persistence())
    results.append(test_llm_config())
    results.append(test_llm_config_from_env())
    results.append(test_imports())
    results.append(test_directories())

    # Clean up test state file
    if os.path.exists('data/risk_state.json'):
        os.remove('data/risk_state.json')

    # Summary
    print()
    print("="*60)
    print("📊 SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print()

    if passed == total:
        print("✅ ALL TESTS PASSED!")
        print()
        print("Your merged agent is ready to use!")
        print()
        print("Next steps:")
        print("  1. Configure .env with your API keys")
        print("  2. Run: python3 agent_improved.py")
        print("  3. Test on testnet for 2-4 weeks")
        print("  4. Monitor logs and performance")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Please review the errors above and fix them.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
