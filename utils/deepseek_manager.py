"""
DeepSeek V4 API Manager
=======================
Professional wrapper for DeepSeek V4 API with system prompts, rate limiting,
response parsing, and fallback handling for crypto trading bot.

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from cachetools import TTLCache

# Configure logging
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class ResponseParsingError(APIError):
    """Raised when response parsing fails."""
    pass


class FallbackTriggered(Exception):
    """Raised when fallback handler is triggered."""
    pass


class ModelVersion(Enum):
    """DeepSeek model versions."""
    V4 = "deepseek-v4"
    V4_LITE = "deepseek-v4-lite"
    V4_PRO = "deepseek-v4-pro"


class ResponseFormat(Enum):
    """Response format options."""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests: int = 60
    window_seconds: int = 60
    max_concurrent: int = 5
    retry_after_seconds: int = 10


@dataclass
class APIConfig:
    """API configuration."""
    api_key: str
    base_url: str = "https://api.deepseek.com/v4"
    timeout_seconds: int = 30
    max_retries: int = 3
    model_version: ModelVersion = ModelVersion.V4_PRO
    response_format: ResponseFormat = ResponseFormat.JSON
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class SystemPrompt:
    """System prompt configuration."""
    role: str = "system"
    content: str = """
    You are an expert crypto trading assistant with deep knowledge of:
    - Technical analysis (TA) and chart patterns
    - Market microstructure and order flow
    - Risk management and position sizing
    - Multi-timeframe analysis
    - Binance exchange mechanics
    - DeFi and CeFi protocols
    
    Guidelines:
    - Provide concise, actionable insights
    - Always consider risk/reward ratios
    - Use proper risk management terminology
    - Reference specific technical indicators when relevant
    - Maintain professional trading discipline
    - Never guarantee profits or give financial advice
    """
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API-compatible dictionary."""
        return {"role": self.role, "content": self.content}


@dataclass
class Message:
    """Chat message structure."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API-compatible dictionary."""
        return {"role": self.role, "content": self.content}


class RateLimiter:
    """
    Token bucket rate limiter with burst support.
    
    Implements a sliding window rate limiter to prevent API abuse.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.tokens = config.max_requests
        self.last_refill = time.monotonic()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Acquire a token for API request.
        
        Returns:
            bool: True if token acquired, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            if elapsed >= self.config.window_seconds:
                self.tokens = self.config.max_requests
                self.last_refill = now
            else:
                # Partial refill
                refill_rate = self.config.max_requests / self.config.window_seconds
                new_tokens = int(elapsed * refill_rate)
                self.tokens = min(self.config.max_requests, self.tokens + new_tokens)
                self.last_refill = now
            
            if self.tokens > 0:
                self.tokens -= 1
                return True
            
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._semaphore.acquire()
        if not await self.acquire():
            raise RateLimitError("Rate limit exceeded")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._semaphore.release()


class ResponseParser:
    """
    Parse and validate API responses.
    
    Handles various response formats and error states.
    """
    
    @staticmethod
    def parse_json_response(response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from API.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ResponseParsingError: If parsing fails
        """
        try:
            # Clean response text (remove markdown code blocks if present)
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            raise ResponseParsingError(f"Failed to parse JSON response: {e}")
    
    @staticmethod
    def extract_content(response: Dict[str, Any]) -> str:
        """
        Extract content from API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Extracted content string
            
        Raises:
            ResponseParsingError: If content extraction fails
        """
        try:
            # Handle different response structures
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            elif "content" in response:
                return response["content"]
            elif "response" in response:
                return response["response"]
            else:
                raise ResponseParsingError("Unknown response structure")
        except (KeyError, IndexError, TypeError) as e:
            raise ResponseParsingError(f"Failed to extract content: {e}")
    
    @staticmethod
    def validate_response(response: Dict[str, Any]) -> bool:
        """
        Validate API response structure.
        
        Args:
            response: API response dictionary
            
        Returns:
            bool: True if valid
        """
        required_fields = ["id", "object", "created", "model"]
        return all(field in response for field in required_fields)


class FallbackHandler:
    """
    Handle API failures with graceful degradation.
    
    Provides fallback responses when API is unavailable.
    """
    
    def __init__(self, fallback_responses: Optional[Dict[str, str]] = None):
        """
        Initialize fallback handler.
        
        Args:
            fallback_responses: Custom fallback responses by scenario
        """
        self.fallback_responses = fallback_responses or {
            "rate_limit": "I'm currently experiencing high demand. Please try again in a moment.",
            "timeout": "The request timed out. Please check your connection and try again.",
            "auth_error": "Authentication failed. Please check your API key configuration.",
            "server_error": "The AI service is temporarily unavailable. Please try again later.",
            "default": "I encountered an unexpected error. Please try again."
        }
    
    def get_fallback_response(self, error: Exception) -> str:
        """
        Get appropriate fallback response based on error type.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Fallback response string
        """
        if isinstance(error, RateLimitError):
            return self.fallback_responses["rate_limit"]
        elif isinstance(error, asyncio.TimeoutError):
            return self.fallback_responses["timeout"]
        elif isinstance(error, AuthenticationError):
            return self.fallback_responses["auth_error"]
        elif isinstance(error, aiohttp.ClientError):
            return self.fallback_responses["server_error"]
        else:
            return self.fallback_responses["default"]


class DeepSeekManager:
    """
    DeepSeek V4 API wrapper with comprehensive features.
    
    Features:
    - System prompt management
    - Rate limiting with burst support
    - Response parsing and validation
    - Fallback handling
    - Retry logic with exponential backoff
    - Caching for common queries
    - Async/await support
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[APIConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        system_prompt: Optional[SystemPrompt] = None,
        fallback_handler: Optional[FallbackHandler] = None,
        cache_ttl: int = 300,
        cache_maxsize: int = 100
    ):
        """
        Initialize DeepSeek manager.
        
        Args:
            api_key: DeepSeek API key
            config: API configuration (optional)
            rate_limit_config: Rate limit configuration (optional)
            system_prompt: Custom system prompt (optional)
            fallback_handler: Custom fallback handler (optional)
            cache_ttl: Cache time-to-live in seconds
            cache_maxsize: Maximum cache size
        """
        self.config = config or APIConfig(api_key=api_key)
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.system_prompt = system_prompt or SystemPrompt()
        self.fallback_handler = fallback_handler or FallbackHandler()
        self.parser = ResponseParser()
        
        # Initialize cache
        self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Request tracking
        self._request_count = 0
        self._last_request_time: Optional[datetime] = None
        
        logger.info("DeepSeekManager initialized with model: %s", self.config.model_version.value)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create HTTP session.
        
        Returns:
            aiohttp.ClientSession instance
        """
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                    self._session = aiohttp.ClientSession(
                        timeout=timeout,
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
        return self._session
    
    def _generate_cache_key(self, messages: List[Message], **kwargs) -> str:
        """
        Generate cache key from messages and parameters.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        content = json.dumps([m.to_dict() for m in messages], sort_keys=True)
        params = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(f"{content}{params}".encode()).hexdigest()
    
    def _build_request_payload(
        self,
        messages: List[Message],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build API request payload.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
            
        Returns:
            Request payload dictionary
        """
        payload = {
            "model": self.config.model_version.value,
            "messages": [self.system_prompt.to_dict()] + [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            "stream": kwargs.get("stream", False)
        }
        
        # Add response format if specified
        if self.config.response_format == ResponseFormat.JSON:
            payload["response_format"] = {"type": "json_object"}
        
        return payload
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(
        self,
        payload: Dict[str, Any],
        endpoint: str = "/chat/completions"
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic.
        
        Args:
            payload: Request payload
            endpoint: API endpoint
            
        Returns:
            API response dictionary
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limited
            APIError: For other API errors
        """
        session = await self._get_session()
        url = f"{self.config.base_url}{endpoint}"
        
        async with self.rate_limiter:
            try:
                async with session.post(url, json=payload) as response:
                    self._request_count += 1
                    self._last_request_time = datetime.utcnow()
                    
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 10))
                        raise RateLimitError(f"Rate limited. Retry after {retry_after}s")
                    
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(f"API error {response.status}: {error_text}")
                    
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                logger.error("HTTP request failed: %s", str(e))
                raise
    
    async def chat(
        self,
        message: str,
        conversation_history: Optional[List[Message]] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat message to DeepSeek API.
        
        Args:
            message: User message
            conversation_history: Previous conversation messages
            use_cache: Whether to use cached responses
            **kwargs: Additional API parameters
            
        Returns:
            API response with parsed content
            
        Raises:
            FallbackTriggered: If fallback response is used
        """
        # Build message list
        messages = conversation_history or []
        messages.append(Message(role="user", content=message))
        
        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(messages, **kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug("Returning cached response")
                return cached_response
        
        # Build payload
        payload = self._build_request_payload(messages, **kwargs)
        
        try:
            # Make API request
            response = await self._make_request(payload)
            
            # Validate response
            if not self.parser.validate_response(response):
                raise ResponseParsingError("Invalid response structure")
            
            # Parse content
            content = self.parser.extract_content(response)
            
            # Parse JSON if expected
            if self.config.response_format == ResponseFormat.JSON:
                content = self.parser.parse_json_response(content)
            
            # Build result
            result = {
                "success": True,
                "content": content,
                "model": response.get("model"),
                "usage": response.get("usage", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache result
            if use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except (RateLimitError, AuthenticationError, APIError, 
                ResponseParsingError, asyncio.TimeoutError) as e:
            logger.error("API request failed: %s", str(e))
            
            # Get fallback response
            fallback_content = self.fallback_handler.get_fallback_response(e)
            
            result = {
                "success": False,
                "content": fallback_content,
                "error": str(e),
                "fallback": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            raise FallbackTriggered("Fallback response used") from e
    
    async def stream_chat(
        self,
        message: str,
        conversation_history: Optional[List[Message]] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        """
        Stream chat response from DeepSeek API.
        
        Args:
            message: User message
            conversation_history: Previous conversation messages
            callback: Optional callback for streaming chunks
            **kwargs: Additional API parameters
            
        Returns:
            Complete response text
        """
        messages = conversation_history or []
        messages.append(Message(role="user", content=message))
        
        payload = self._build_request_payload(messages, stream=True, **kwargs)
        
        session = await self._get_session()
        url = f"{self.config.base_url}/chat/completions"
        
        full_response = ""
        
        try:
            async with self.rate_limiter:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        raise APIError(f"Stream error: {response.status}")
                    
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                
                                try:
                                    chunk = json.loads(data)
                                    if "choices" in chunk:
                                        delta = chunk["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            full_response += content
                                            if callback:
                                                callback(content)
                                except json.JSONDecodeError:
                                    continue
            
            return full_response
            
        except Exception as e:
            logger.error("Stream failed: %s", str(e))
            fallback = self.fallback_handler.get_fallback_response(e)
            if callback:
                callback(fallback)
            return fallback
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using DeepSeek.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        prompt = f"""
        Analyze the sentiment of the following text for crypto trading context.
        Provide analysis in JSON format with fields:
        - sentiment: (bullish/bearish/neutral)
        - confidence: (0-100)
        - key_signals: (list of key signals found)
        - recommendation: (brief action recommendation)
        
        Text: {text}
        """
        
        try:
            result = await self.chat(prompt, use_cache=False)
            return result
        except FallbackTriggered:
            return {
                "success": False,
                "content": "Sentiment analysis unavailable",
                "fallback": True
            }
    
    async def generate_trading_signal(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading signal based on market data and indicators.
        
        Args:
            market_data: Current market data
            indicators: Technical indicators
            
        Returns:
            Trading signal with analysis
        """
        prompt = f"""
        Analyze the following market data and technical indicators to generate a trading signal.
        Provide analysis in JSON format with fields:
        - signal: (BUY/SELL/HOLD)
        - confidence: (0-100)
        - reasoning: (detailed reasoning)
        - risk_level: (LOW/MEDIUM/HIGH)
        - stop_loss: (suggested stop loss price)
        - take_profit: (suggested take profit price)
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Technical Indicators:
        {json.dumps(indicators, indent=2)}
        """
        
        try:
            result = await self.chat(prompt, use_cache=False)
            return result
        except FallbackTriggered:
            return {
                "success": False,
                "content": "Signal generation unavailable",
                "fallback": True
            }
    
    async def get_market_insight(
        self,
        symbol: str,
        timeframe: str,
        price_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get market insight for specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Analysis timeframe
            price_data: Price data for analysis
            
        Returns:
            Market insight with analysis
        """
        prompt = f"""
        Provide a comprehensive market insight for {symbol} on {timeframe} timeframe.
        Include analysis of:
        - Current trend direction and strength
        - Key support and resistance levels
        - Volume analysis
        - Momentum indicators
        - Potential entry/exit points
        
        Price Data:
        {json.dumps(price_data, indent=2)}
        
        Format response as JSON with appropriate fields.
        """
        
        try:
            result = await self.chat(prompt, use_cache=False)
            return result
        except FallbackTriggered:
            return {
                "success": False,
                "content": "Market insight unavailable",
                "fallback": True
            }
    
    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("DeepSeekManager session closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "request_count": self._request_count,
            "last_request_time": self._last_request_time.isoformat() if self._last_request_time else None,
            "cache_size": len(self.cache),
            "cache_maxsize": self.cache.maxsize,
            "model": self.config.model_version.value,
            "rate_limit_max": self.rate_limiter.config.max_requests,
            "rate_limit_window": self.rate_limiter.config.window_seconds
        }


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize manager
        manager = DeepSeekManager(
            api_key="your-api-key-here",
            config=APIConfig(
                api_key="your-api-key-here",
                model_version=ModelVersion.V4_PRO
            )
        )
        
        try:
            # Simple chat
            response = await manager.chat("What's the current market sentiment for BTC?")
            print("Chat Response:", response)
            
            # Streaming chat
            async def handle_chunk(chunk: str):
                print(chunk, end="", flush=True)
            
            print("\nStreaming response:")
            await manager.stream_chat(
                "Analyze ETH/USDT current trend",
                callback=handle_chunk
            )
            
            # Get stats
            print("\n\nManager Stats:", manager.get_stats())
            
        finally:
            await manager.close()
    
    # Run example
    asyncio.run(main())