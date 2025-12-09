"""
LLM Regime Classifier with Caching

This module uses LLM to classify market regime, trend, and volatility state.
Results are cached for 1-4 hours to reduce API calls and improve determinism.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from loguru import logger
from openai import OpenAI


class RegimeCache:
    """Cache for regime classifications to reduce API calls"""

    def __init__(self, cache_duration_hours: float = 2.0):
        self.cache_duration_hours = cache_duration_hours
        self.cache: Optional[Dict] = None
        self.cache_file = "data/regime_cache.json"

        # Load from disk if exists
        self._load_from_disk()

    def _load_from_disk(self):
        """Load cache from disk (persists across restarts)"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    valid_until = datetime.fromisoformat(data['valid_until'])

                    # Only use if still valid
                    if datetime.now() < valid_until:
                        self.cache = data
                        logger.info(f"Loaded regime cache from disk (valid until {valid_until})")
        except Exception as e:
            logger.warning(f"Failed to load regime cache: {e}")

    def _save_to_disk(self):
        """Save cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save regime cache: {e}")

    def is_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.cache:
            return False

        try:
            valid_until = datetime.fromisoformat(self.cache['valid_until'])
            return datetime.now() < valid_until
        except (KeyError, ValueError):
            return False

    def get(self) -> Optional[Dict]:
        """Get cached regime data if valid"""
        if self.is_valid():
            return self.cache
        return None

    def set(self, regime_data: Dict):
        """Set cache with expiration"""
        valid_until = datetime.now() + timedelta(hours=self.cache_duration_hours)
        regime_data['valid_until'] = valid_until.isoformat()
        regime_data['cached_at'] = datetime.now().isoformat()

        self.cache = regime_data
        self._save_to_disk()

        logger.info(f"Regime cached until {valid_until} ({self.cache_duration_hours}h)")

    def invalidate(self):
        """Force cache refresh"""
        self.cache = None
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("Regime cache invalidated")


class LLMRegimeClassifier:
    """
    Uses LLM to classify market regime with caching.

    This reduces API calls by 95% (only calls LLM every 1-4 hours)
    and makes the system more deterministic (same regime for multiple trades).
    """

    def __init__(self,
                 model: str = "gpt-4o-mini",
                 provider: str = "openai",
                 cache_duration_hours: float = 2.0):
        self.provider = provider
        self.model = model
        self.cache = RegimeCache(cache_duration_hours)

        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        logger.info(f"LLMRegimeClassifier initialized: {provider}/{model}, cache={cache_duration_hours}h")

    def get_regime(self, market_data: Dict, force_refresh: bool = False) -> Dict:
        """
        Get market regime classification (cached).

        Args:
            market_data: Market snapshot with indicators
            force_refresh: Force LLM query even if cache valid

        Returns:
            {
                'regime': 'MOMENTUM' | 'RANGING' | 'TRANSITIONAL',
                'trend': 'STRONG_BULL' | 'WEAK_BULL' | 'NEUTRAL' | 'WEAK_BEAR' | 'STRONG_BEAR',
                'volatility': 'HIGH' | 'NORMAL' | 'LOW',
                'risk_mode': 'AGGRESSIVE' | 'NORMAL' | 'DEFENSIVE',
                'confidence': 0.0-1.0,
                'reasoning': str,
                'valid_until': ISO timestamp,
                'cached_at': ISO timestamp
            }
        """

        # Check cache first
        if not force_refresh:
            cached = self.cache.get()
            if cached:
                logger.info(f"Using cached regime: {cached['regime']} / {cached['trend']}")
                return cached

        # Cache miss - query LLM
        logger.info("Cache expired - querying LLM for regime classification")
        regime_data = self._query_llm(market_data)

        # Cache result
        self.cache.set(regime_data)

        return regime_data

    def _query_llm(self, market_data: Dict) -> Dict:
        """Query LLM for regime classification"""

        prompt = self._create_prompt(market_data)

        if self.provider == "openai":
            response = self._query_openai(prompt)
        else:
            response = self._query_anthropic(prompt)

        parsed = self._parse_response(response)

        logger.info(f"LLM Regime: {parsed['regime']} / {parsed['trend']} (confidence: {parsed['confidence']:.2f})")
        logger.info(f"Reasoning: {parsed['reasoning']}")

        return parsed

    def _create_prompt(self, market_data: Dict) -> str:
        """Create focused prompt for regime classification only"""

        ind = market_data['indicators']

        prompt = f"""You are a market regime classifier. Your job is to classify the current market state, NOT to make trading decisions.

MARKET DATA FOR {market_data['symbol']}:
- Current Price: ${market_data['current_price']:.2f}

MULTI-TIMEFRAME ANALYSIS:
- 1h Regime: {ind['market_regime']} (Score: {ind.get('momentum_score', 0)})
- 4h Regime: {ind.get('regime_4h', 'N/A')} (Score: {ind.get('momentum_score_4h', 'N/A')})
- Alignment: {ind.get('regime_alignment', 'SINGLE_TF')}

TREND INDICATORS:
- Trend: {ind['trend']}
- Price vs MA(20): {ind['price_vs_ma20_pct']:.2f}%
- Price vs MA(50): {ind['price_vs_ma50_pct']:.2f}%
- Trend Strength: {ind['trend_strength_pct']:.2f}%

MOMENTUM:
- RSI(14): {ind['rsi_14']:.2f}
- MACD Histogram: {ind['macd_histogram']:.4f}
- 10h Returns: {ind['returns_10h_pct']:.2f}%
- 24h Returns: {ind['returns_24h_pct']:.2f}%

VOLATILITY:
- ATR: {ind['atr_pct']:.2f}% of price
- Bollinger Position: {ind['bollinger_position']*100:.1f}%
- Normalized Returns: {ind.get('normalized_returns', 0):.2f}x expected

VOLUME:
- Volume Ratio: {ind['volume_ratio']:.2f}x average

YOUR TASK:
Classify the market into these categories:

1. REGIME (Primary Classification):
   - MOMENTUM: Strong directional move, RSI can stay extended (70-80 normal in uptrend)
   - RANGING: Sideways choppy action, mean reversion works
   - TRANSITIONAL: Mixed signals, wait for clarity

2. TREND (Strength & Direction):
   - STRONG_BULL: Clear uptrend, price well above MAs, RSI >60
   - WEAK_BULL: Mild uptrend, price near MAs, RSI 50-60
   - NEUTRAL: No clear direction, price whipsawing
   - WEAK_BEAR: Mild downtrend, price near MAs, RSI 40-50
   - STRONG_BEAR: Clear downtrend, price well below MAs, RSI <40

3. VOLATILITY:
   - HIGH: ATR > 2.5% of price
   - NORMAL: ATR 1.0-2.5% of price
   - LOW: ATR < 1.0% of price

4. RISK_MODE (How aggressive should strategy be?):
   - AGGRESSIVE: Strong regime + trend alignment + high volume
   - NORMAL: Clear regime but some mixed signals
   - DEFENSIVE: Transitional or low volume or conflicting signals

Respond ONLY with valid JSON:
{{
    "regime": "MOMENTUM" or "RANGING" or "TRANSITIONAL",
    "trend": "STRONG_BULL" or "WEAK_BULL" or "NEUTRAL" or "WEAK_BEAR" or "STRONG_BEAR",
    "volatility": "HIGH" or "NORMAL" or "LOW",
    "risk_mode": "AGGRESSIVE" or "NORMAL" or "DEFENSIVE",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation (2-3 sentences)"
}}

Focus on regime classification accuracy. Deterministic trading rules will handle entry/exit logic.
"""

        return prompt

    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a market regime classifier. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low for consistency
                max_tokens=400
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
                max_tokens=400,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response with validation"""

        try:
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]

            data = json.loads(json_str)

            # Validate required fields
            required = ['regime', 'trend', 'volatility', 'risk_mode', 'confidence', 'reasoning']
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Validate enum values
            valid_regimes = ['MOMENTUM', 'RANGING', 'TRANSITIONAL']
            valid_trends = ['STRONG_BULL', 'WEAK_BULL', 'NEUTRAL', 'WEAK_BEAR', 'STRONG_BEAR']
            valid_volatility = ['HIGH', 'NORMAL', 'LOW']
            valid_risk = ['AGGRESSIVE', 'NORMAL', 'DEFENSIVE']

            if data['regime'] not in valid_regimes:
                logger.warning(f"Invalid regime '{data['regime']}', defaulting to TRANSITIONAL")
                data['regime'] = 'TRANSITIONAL'

            if data['trend'] not in valid_trends:
                logger.warning(f"Invalid trend '{data['trend']}', defaulting to NEUTRAL")
                data['trend'] = 'NEUTRAL'

            if data['volatility'] not in valid_volatility:
                data['volatility'] = 'NORMAL'

            if data['risk_mode'] not in valid_risk:
                data['risk_mode'] = 'NORMAL'

            # Clamp confidence
            data['confidence'] = max(0.0, min(1.0, float(data['confidence'])))

            return data

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response}")

            # Return safe default
            return {
                'regime': 'TRANSITIONAL',
                'trend': 'NEUTRAL',
                'volatility': 'NORMAL',
                'risk_mode': 'DEFENSIVE',
                'confidence': 0.0,
                'reasoning': f'Failed to parse LLM response: {str(e)}'
            }

    def invalidate_cache(self):
        """Force regime refresh on next call"""
        self.cache.invalidate()
