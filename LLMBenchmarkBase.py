from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
import requests
import time
import re


class LLMBenchmarkBase(Strategy):
    # #########################################################
    # OpenRouter API configuration
    # 
    # Create an account at https://openrouter.ai/
    # Create an API key at: https://openrouter.ai/settings/keys
    # 
    # Replace "xxxxxxx" with your API key.
    # Replace OPENROUTER_API_URL with another value in case you want to use a provider other than OpenRouter. 
    # 
    # #########################################################
    OPENROUTER_API_KEY = "xxxxxxx"
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    # #########################################################
    
    def __init__(self):
        super().__init__()
    
    @property
    def model_name(self):
        """Override this in child classes to specify the model"""
        raise NotImplementedError("Child class must specify model_name")
    
    def prepare_market_data(self, num_candles: int = 30) -> dict:
        """
        Extract recent candles and calculate technical indicators.
        Returns a dictionary with formatted market data.
        """
        # Get recent candles (last num_candles)
        candles = self.candles[-num_candles:]
        
        # Calculate technical indicators
        ema_9 = ta.ema(self.candles, 9)
        ema_21 = ta.ema(self.candles, 21)
        ema_50 = ta.ema(self.candles, 50)
        rsi_14 = ta.rsi(self.candles, 14)
        atr_14 = ta.atr(self.candles, 14)
        adx_14 = ta.adx(self.candles, 14)
        
        # Format candles data
        candles_data = []
        for candle in candles:
            candles_data.append({
                'timestamp': candle[0],
                'open': round(candle[1], 2),
                'high': round(candle[3], 2),
                'low': round(candle[4], 2),
                'close': round(candle[2], 2),
                'volume': round(candle[5], 2)
            })
        
        return {
            'current_price': round(self.price, 2),
            'candles': candles_data,
            'indicators': {
                'ema_9': round(ema_9, 2),
                'ema_21': round(ema_21, 2),
                'ema_50': round(ema_50, 2),
                'rsi_14': round(rsi_14, 2),
                'atr_14': round(atr_14, 4),
                'adx_14': round(adx_14, 2)
            },
            'position_status': 'long' if self.is_long else ('short' if self.is_short else 'close'),
            'position_pnl': self.position.pnl_percentage
        }
    
    def create_trading_prompt(self, market_data: dict) -> str:
        """
        Create a structured prompt for the LLM to make trading decisions.
        """
        candles_summary = "Last 5 candles:\n"
        for candle in market_data['candles'][-5:]:
            candles_summary += f"  Open: {candle['open']}, High: {candle['high']}, Low: {candle['low']}, Close: {candle['close']}, Volume: {candle['volume']}\n"
        
        prompt = f"""You are a professional cryptocurrency trader analyzing market data to make trading decisions.

Current Market Data:
- Current Price: {market_data['current_price']}
- Position Status: {market_data['position_status']}
- Position PNL: {market_data['position_pnl']}

Technical Indicators:
- EMA 9: {market_data['indicators']['ema_9']}
- EMA 21: {market_data['indicators']['ema_21']}
- EMA 50: {market_data['indicators']['ema_50']}
- RSI 14: {market_data['indicators']['rsi_14']}
- ATR 14: {market_data['indicators']['atr_14']}
- ADX 14: {market_data['indicators']['adx_14']}

{candles_summary}

Based on this market data and technical indicators, what trading action should be taken?

CRITICAL INSTRUCTIONS - READ CAREFULLY:
Your response must be EXACTLY ONE of these three words, with no additional text, punctuation, or explanation:

long
short
hold

DO NOT add any explanation, reasoning, or additional words.
DO NOT add punctuation marks, quotes, or any other characters.
DO NOT write sentences like "I think we should go long" or "The answer is hold".

CORRECT EXAMPLES:
- long
- short  
- hold

INCORRECT EXAMPLES:
- "long" (has quotes)
- Long position recommended (has extra words)
- I believe the answer is hold (has extra words)
- short. (has punctuation)

Your entire response must be exactly one of the three words: long, short, or hold"""

        return prompt
    
    def parse_llm_decision(self, response: str) -> str:
        """
        Parse the LLM response and validate it's one of the allowed actions.
        Returns "long", "short", or "hold". Defaults to "hold" for invalid responses.
        
        Uses multiple parsing strategies to extract the decision even if the LLM
        adds extra text or formatting.
        """
        if not response:
            self.log("WARNING: Empty response from LLM, defaulting to 'hold'")
            return "hold"

        # Clean up the response - remove extra whitespace, quotes, and convert to lowercase
        cleaned = response.strip().lower()
        cleaned = cleaned.strip('"\'`')  # Remove common quote characters
        
        valid_actions = ['long', 'short', 'hold']
        
        # Strategy 1: Check if the entire response is exactly one of our actions
        if cleaned in valid_actions:
            return cleaned
        
        # Strategy 2: Check if response starts with one of our actions (followed by space or punctuation)
        for action in valid_actions:
            if cleaned.startswith(action):
                # Make sure it's the word itself, not part of a longer word
                if len(cleaned) == len(action) or cleaned[len(action)] in [' ', '.', '!', '?', ',', '\n']:
                    return action
        
        # Strategy 3: Use regex to find the action as a standalone word
        # This ensures we match "long" in "I think long" but not in "longer"
        for action in valid_actions:
            pattern = r'\b' + action + r'\b'
            match = re.search(pattern, cleaned)
            if match:
                return action
        
        # Strategy 4: Check if any action appears anywhere (less strict fallback)
        for action in valid_actions:
            if action in cleaned:
                self.log(f"WARNING: Found '{action}' in response but not as standalone word. Response: '{response}'")
                return action
        
        # No valid action found, default to hold
        self.log(f"WARNING: Could not parse valid action from LLM response: '{response}'. Defaulting to 'hold'")
        return "hold"
    
    def call_openrouter(self, prompt: str) -> str:
        """
        Send request to OpenRouter API and return the response.
        Retries up to 3 times with a short pause between attempts.
        Returns "hold" if all retries fail.
        """
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }

                response = requests.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=data,
                    timeout=30
                )

                response.raise_for_status()
                result = response.json()

                return result['choices'][0]['message']['content']

            except Exception as e:
                if attempt < max_retries - 1:  # Don't log on the last attempt
                    print(f"OpenRouter API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"OpenRouter API error (final attempt): {str(e)}")
                    return "hold"
    
    @property
    @cached
    def llm_decision(self) -> str:
        """
        Get trading decision from LLM.
        Returns "long", "short", or "hold".
        """
        # Prepare market data
        market_data = self.prepare_market_data()

        # Create prompt
        prompt = self.create_trading_prompt(market_data)

        # Get LLM response
        response = self.call_openrouter(prompt)

        # Log the response
        self.log(f'LLM response: "{response}"')

        # Parse and validate decision
        decision = self.parse_llm_decision(response)

        # Log the decision
        self.log(f'LLM decision: "{decision}"')

        return decision

    # #########################################################
    # Jesse strategy methods
    # #########################################################
    
    def should_long(self) -> bool:
        """Enter long position if LLM says 'long' and we're not already long"""
        return self.llm_decision == "long"

    def should_short(self) -> bool:
        """Enter short position if LLM says 'short' and we're not already short"""
        return self.llm_decision == "short"

    def go_long(self):
        # Use 95% of available margin for the position
        qty = utils.size_to_qty(self.available_margin * 0.95, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        # Use 95% of available margin for the position
        qty = utils.size_to_qty(self.available_margin * 0.95, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price

    def on_open_position(self, order) -> None:
        # set SL/TP when position opens using actual entry price
        if self.is_long:
            sl = self.position.entry_price - (ta.atr(self.candles) * 2)
            tp = self.position.entry_price + (ta.atr(self.candles) * 2)
            self.stop_loss = self.position.qty, sl
            self.take_profit = self.position.qty, tp
        elif self.is_short:
            sl = self.position.entry_price + (ta.atr(self.candles) * 2)
            tp = self.position.entry_price - (ta.atr(self.candles) * 2)
            self.stop_loss = self.position.qty, sl
            self.take_profit = self.position.qty, tp
