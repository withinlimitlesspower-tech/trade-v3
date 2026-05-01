```py
"""
Binance API Client Module

Professional-grade Binance API wrapper with:
- Asynchronous HTTP client with automatic retry and rate limiting
- WebSocket price streams for real-time data
- Comprehensive klines/candlestick data fetching for all timeframes
- Built-in caching with TTL support
- Request deduplication and throttling
- Full error handling and logging

Author: Trading Bot Team
Version: 2.0.0
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

import aiohttp
import pandas as pd
import websockets
from aiohttp import ClientTimeout, TCPConnector
from cachetools import TTLCache
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure module logger
logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Supported trading timeframes with Binance interval codes."""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    @classmethod
    def get_all_timeframes(cls) -> List[str]:
        """Return all timeframe interval codes."""
        return [tf.value for tf in cls]


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class BinanceClientError(Exception):
    """Base exception for Binance client errors."""
    pass


class BinanceAPIError(BinanceClientError):
    """Raised when Binance API returns an error."""
    def __init__(self, status_code: int, error_code: int, message: str):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(f"API Error {error_code}: {message}")


class RateLimitExceeded(BinanceClientError):
    """Raised when rate limit is exceeded."""
    pass


class BinanceClient:
    """
    Asynchronous Binance API client with caching, WebSocket support,
    and comprehensive error handling.

    Features:
    - Automatic request retry with exponential backoff
    - Rate limit tracking and throttling
    - In-memory caching with configurable TTL
    - WebSocket price streams for real-time updates
    - Klines/candlestick data for all timeframes
    - Account information and order management
    """

    # Binance API endpoints
    BASE_URL = "https://api.binance.com"
    BASE_URL_TESTNET = "https://testnet.binance.vision"
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    WS_BASE_URL_TESTNET = "wss://testnet.binance.vision/ws"

    # Rate limiting constants
    RATE_LIMIT_WEIGHT = 1200  # Max weight per minute
    RATE_LIMIT_ORDERS = 50    # Max orders per 10 seconds

    # Default cache TTLs (in seconds)
    CACHE_TTL_PRICE = 2
    CACHE_TTL_KLINES = 60
    CACHE_TTL_ACCOUNT = 30
    CACHE_TTL_EXCHANGE_INFO = 300

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        cache_ttl: int = 60,
        max_retries: int = 3,
        request_timeout: int = 30,
    ):
        """
        Initialize Binance client.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            cache_ttl: Default cache TTL in seconds
            max_retries: Maximum number of retry attempts
            request_timeout: HTTP request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL
        self.ws_base_url = self.WS_BASE_URL_TESTNET if testnet else self.WS_BASE_URL

        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._ws_tasks: Dict[str, asyncio.Task] = {}

        # Rate limiting
        self._rate_limit_weight_used = 0
        self._rate_limit_weight_reset = 0
        self._rate_limit_order_used = 0
        self._rate_limit_order_reset = 0
        self._request_semaphore = asyncio.Semaphore(10)  # Max concurrent requests

        # Caching
        self._cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self._cache_ttl = cache_ttl

        # Retry configuration
        self._max_retries = max_retries
        self._request_timeout = request_timeout

        # WebSocket callbacks
        self._price_callbacks: Dict[str, List[Callable]] = {}
        self._depth_callbacks: Dict[str, List[Callable]] = {}
        self._kline_callbacks: Dict[str, List[Callable]] = {}

        logger.info(
            f"BinanceClient initialized (testnet={testnet}, "
            f"cache_ttl={cache_ttl}s, max_retries={max_retries})"
        )

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close all connections and cleanup resources."""
        # Close WebSocket connections
        for symbol, ws in self._ws_connections.items():
            try:
                await ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for {symbol}: {e}")

        # Cancel WebSocket tasks
        for symbol, task in self._ws_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("HTTP session closed")

        self._ws_connections.clear()
        self._ws_tasks.clear()
        self._session = None

    async def _ensure_session(self):
        """Ensure HTTP session exists and is active."""
        if self._session is None or self._session.closed:
            connector = TCPConnector(
                limit=20,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = ClientTimeout(total=self._request_timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_headers(),
            )
            logger.debug("Created new HTTP session")

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "TradingBot/2.0",
        }
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for authenticated requests.

        Args:
            params: Request parameters

        Returns:
            Hex-encoded signature string
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _handle_rate_limits(self, response: aiohttp.ClientResponse):
        """
        Track rate limit usage from response headers.

        Args:
            response: HTTP response from Binance API
        """
        weight_used = response.headers.get("X-MBX-USED-WEIGHT")
        weight_reset = response.headers.get("X-MBX-WEIGHT-RESET")
        order_used = response.headers.get("X-MBX-ORDER-COUNT")
        order_reset = response.headers.get("X-MBX-ORDER-COUNT-RESET")

        if weight_used:
            self._rate_limit_weight_used = int(weight_used)
        if weight_reset:
            self._rate_limit_weight_reset = int(weight_reset)
        if order_used:
            self._rate_limit_order_used = int(order_used)
        if order_reset:
            self._rate_limit_order_reset = int(order_reset)

        # Check if we're approaching rate limits
        if self._rate_limit_weight_used > self.RATE_LIMIT_WEIGHT * 0.9:
            logger.warning(
                f"Approaching rate limit: {self._rate_limit_weight_used}/"
                f"{self.RATE_LIMIT_WEIGHT} weight used"
            )

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to Binance API with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            params: Query/request parameters
            signed: Whether request requires signature
            retry_count: Current retry attempt number

        Returns:
            JSON response as dictionary

        Raises:
            BinanceAPIError: On API error response
            RateLimitExceeded: When rate limit is hit
            aiohttp.ClientError: On connection errors
        """
        await self._ensure_session()

        url = f"{self.base_url}{endpoint}"
        request_params = params or {}

        # Add timestamp and signature for signed requests
        if signed:
            request_params["timestamp"] = int(time.time() * 1000)
            request_params["signature"] = self._generate_signature(request_params)

        # Rate limiting semaphore
        async with self._request_semaphore:
            try:
                async with self._session.request(
                    method, url, params=request_params if method == "GET" else None,
                    json=request_params if method != "GET" else None,
                ) as response:
                    # Track rate limits
                    self._handle_rate_limits(response)

                    # Parse response
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"Rate limited, retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._request(
                            method, endpoint, params, signed, retry_count + 1
                        )

                    if response.status == 418:
                        raise RateLimitExceeded("IP has been auto-banned for rate limit abuse")

                    data = await response.json()

                    if response.status >= 400:
                        error_code = data.get("code", -1)
                        error_msg = data.get("msg", "Unknown error")
                        raise BinanceAPIError(response.status, error_code, error_msg)

                    return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if retry_count < self._max_retries:
                    wait_time = 2 ** retry_count
                    logger.warning(
                        f"Request failed (attempt {retry_count + 1}/{self._max_retries}): "
                        f"{e}. Retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                    return await self._request(
                        method, endpoint, params, signed, retry_count + 1
                    )
                raise BinanceClientError(f"Request failed after {self._max_retries} retries: {e}")

    # ==================== Public API Methods ====================

    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get current exchange trading rules and symbol information.

        Returns:
            Exchange info dictionary
        """
        cache_key = "exchange_info"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = await self._request("GET", "/api/v3/exchangeInfo")
        self._cache[cache_key] = data
        return data

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get trading rules and filters for a specific symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Symbol information dictionary or None if not found
        """
        exchange_info = await self.get_exchange_info()
        for s in exchange_info.get("symbols", []):
            if s["symbol"] == symbol.upper():
                return s
        return None

    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest price ticker for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data with price information
        """
        cache_key = f"ticker_price_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = await self._request(
            "GET", "/api/v3/ticker/price",
            params={"symbol": symbol.upper()}
        )
        self._cache[cache_key] = data
        return data

    async def get_all_ticker_prices(self) -> List[Dict[str, Any]]:
        """
        Get latest prices for all symbols.

        Returns:
            List of ticker data for all symbols
        """
        cache_key = "all_ticker_prices"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = await self._request("GET", "/api/v3/ticker/price")
        self._cache[cache_key] = data
        return data

    async def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24-hour rolling window ticker statistics.

        Args:
            symbol: Trading pair symbol

        Returns:
            24hr ticker statistics
        """
        cache_key = f"ticker_24hr_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = await self._request(
            "GET", "/api/v3/ticker/24hr",
            params={"symbol": symbol.upper()}
        )
        self._cache[cache_key] = data
        return data

    # ==================== Klines/Candlestick Data ====================

    async def get_klines(
        self,
        symbol: str,
        interval: Union[str, TimeFrame],
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get kline/candlestick data for a symbol.

        Args:
            symbol: Trading pair symbol
            interval: Timeframe interval (e.g., '1h', '4h', '1d')
            limit: Number of klines to fetch (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        # Convert TimeFrame enum to string if needed
        if isinstance(interval, TimeFrame):
            interval = interval.value

        # Build cache key
        cache_key = f"klines_{symbol}_{interval}_{limit}_{start_time}_{end_time}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Validate parameters
        symbol = symbol.upper()
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/api/v3/klines", params=params)

        # Parse into DataFrame
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        df = pd.DataFrame(data, columns=columns)

        # Convert numeric columns
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert timestamps
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Set index
        df.set_index("open_time", inplace=True)

        # Cache the result
        self._cache[cache_key] = df
        return df

    async def get_klines_for_all_timeframes(
        self,
        symbol: str,
        limit: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get kline data for all supported timeframes.

        Args:
            symbol: Trading pair symbol
            limit: Number of klines per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        tasks = {}
        for tf in TimeFrame:
            tasks[tf.value] = self.get_klines(symbol, tf.value, limit)

        results = {}
        for tf, task in tasks.items():
            try:
                results[tf] = await task
            except Exception as e:
                logger.error(f"Failed to fetch {tf} klines for {symbol}: {e}")
                results[tf] = pd.DataFrame()

        return results

    # ==================== Order Book ====================

    async def get_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get order book for a symbol.

        Args:
            symbol: Trading pair symbol
            limit: Depth (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Order book with bids and asks
        """
        cache_key = f"orderbook_{symbol}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = await self._request(
            "GET", "/api/v3/depth",
            params={"symbol": symbol.upper(), "limit": limit}
        )
        self._cache[cache_key] = data
        return data

    # ==================== Account Information ====================

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information.

        Requires valid API key with read permissions.

        Returns:
            Account information dictionary
        """
        cache_key = "account_info"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = await self._request("GET", "/api/v3/account", signed=True)
        self._cache[cache_key] = data
        return data

    async def get_asset_balance(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Get balance for a specific asset.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')

        Returns:
            Balance information or None if asset not found
        """
        account = await self.get_account_info()
        for balance in account.get("balances", []):
            if balance["asset"] == asset.upper():
                return balance
        return None

    async def get_all_balances(self) -> List[Dict[str, Any]]:
        """
        Get all non-zero asset balances.

        Returns:
            List of balances with non-zero amounts
        """
        account = await self.get_account_info()
        return [
            b for b in account.get("balances", [])
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        ]

    # ==================== Order Management ====================

    async def create_order(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        order_type: Union[str, OrderType],
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        new_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new order.

        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            order_type: Order type (MARKET, LIMIT, etc.)
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
            stop_price: Stop price (required for STOP_LOSS orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            new_client_order_id: Custom order ID

        Returns:
            Order response from Binance
        """
        # Convert enums to strings
        if isinstance(side, OrderSide):
            side = side.value
        if isinstance(order_type, OrderType):
            order_type = order_type.value

        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }

        # Add optional parameters based on order type
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            params["price"] = price
            params["timeInForce"] = time_in_force

        if order_type in ["STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT"]:
            if stop_price is None:
                raise ValueError("Stop price is required for stop/take profit orders")
            params["stopPrice"] = stop_price

        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id

        return await self._request("POST", "/api/v3/order", params=params, signed=True)

    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            symbol: Trading pair symbol
            order_id: Binance order ID
            client_order_id: Client custom order ID

        Returns:
            Cancelled order response
        """
        params = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")

        return await self._request("DELETE", "/api/v3/order", params=params, signed=True)

    async def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get order status.

        Args:
            symbol: Trading pair symbol
            order_id: Binance order ID
            client_order_id: Client custom order ID

        Returns:
            Order status information
        """
        params = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")

        return await self._request("GET", "/api/v3/order", params=params, signed=True)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        return await self._request("GET", "/api/v3/openOrders", params=params, signed=True)

    async def get_all_orders(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all orders for a symbol.

        Args:
            symbol: Trading pair symbol
            limit: Number of orders to return (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of orders
        """
        params = {
            "symbol": symbol.upper(),
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return await self._request("GET", "/api/v3/allOrders", params=params, signed=True)

    # ==================== WebSocket Streams ====================

    async def _connect_websocket(self, stream_name: str) -> websockets.WebSocketClientProtocol:
        """
        Connect to a WebSocket stream.

        Args:
            stream_name: Stream name (e.g., 'btcusdt@trade')

        Returns:
            WebSocket connection
        """
        ws_url = f"{self.ws_base_url}/{stream_name}"
        ws = await websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        )
        logger.info(f"Connected to WebSocket stream: {stream_name}")
        return ws

    async def _handle_websocket_stream(
        self,
        stream_name: str,
        callback: Callable[[Dict[str, Any]], None],
    ):
        """
        Handle WebSocket stream with reconnection logic.

        Args:
            stream_name: Stream name
            callback: Callback function for received data
        """
        while True:
            try:
                ws = await self._connect_websocket(stream_name)
                self._ws_connections[stream_name] = ws

                async for message in ws:
                    try:
                        data = json.loads(message)
                        callback(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from WebSocket: {e}")
                    except Exception as e:
                        logger.error(f"Error in WebSocket callback: {e}")

            except websockets.ConnectionClosed:
                logger.warning(f"WebSocket connection closed for {stream_name}, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error for {stream_name}: {e}")
                await asyncio.sleep(10)

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
    ):
        """
        Subscribe to real-time ticker updates for a symbol.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with ticker data
        """
        stream_name = f"{symbol.lower()}@ticker"
        if stream_name not in self._price_callbacks:
            self._price_callbacks[stream_name] = []
        self._price_callbacks[stream_name].append(callback)

        if stream_name not in self._ws_tasks:
            task = asyncio.create_task(
                self._handle_websocket_stream(stream_name, callback)
            )
            self._ws_tasks[stream_name] = task

    async def subscribe_depth(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        update_speed: str = "1000ms",
    ):
        """
        Subscribe to real-time depth updates for a symbol.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with depth data
            update_speed: Update speed (1000ms or 100ms)
        """
        stream_name = f"{symbol.lower()}@depth@{update_speed}"
        if stream_name not in self._depth_callbacks:
            self._depth_callbacks[stream_name] = []
        self._depth_callbacks[stream_name].append(callback)

        if stream_name not in self._ws_tasks:
            task = asyncio.create_task(
                self._handle_websocket_stream(stream_name, callback)
            )
            self._ws_tasks[stream_name] = task

    async def subscribe_klines(
        self,
        symbol: str,
        interval: Union[str, TimeFrame],
        callback: Callable[[Dict[str, Any]], None],
    ):
        """
        Subscribe to real-time kline updates for a symbol.

        Args:
            symbol: Trading pair symbol
            interval: Timeframe interval
            callback: Function to call with kline data
        """
        if isinstance(interval, TimeFrame):
            interval = interval.value

        stream_name = f"{symbol.lower()}@kline_{interval}"
        if stream_name not in self._kline_callbacks:
            self._kline_callbacks[stream_name] = []
        self._kline_callbacks[stream_name].append(callback)

        if stream_name not in self._ws_tasks:
            task = asyncio.create_task(
                self._handle_websocket_stream(stream_name, callback)
            )
            self._ws_tasks[stream_name] = task

    async def unsubscribe(self, stream_name: str):
        """
        Unsubscribe from a WebSocket stream.

        Args:
            stream_name: Stream name to unsubscribe from
        """
        if stream_name in self._ws_connections:
            await self._ws_connections[stream_name].close()
            del self._ws_connections[stream_name]

        if stream_name in self._ws_tasks:
            self._ws_tasks[stream_name].cancel()
            try:
                await self._ws_tasks[stream_name]
            except asyncio.CancelledError:
                pass
            del self._ws_tasks[stream_name]

        # Remove callbacks
        for callback_dict in [self._price_callbacks, self._depth_callbacks, self._kline_callbacks]:
            if stream_name in callback_dict:
                del callback_dict[stream_name]

        logger.info(f"Unsubscribed from stream: {stream_name}")

    # ==================== Utility Methods ====================

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self._cache),
            "maxsize": self._cache.maxsize,
            "ttl": self._cache.ttl,
            "currsize": self._cache.currsize,
        }

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status.

        Returns:
            Dictionary with rate limit information
        """
        return {
            "weight_used": self._rate_limit_weight_used,
            "weight_limit": self.RATE_LIMIT_WEIGHT,
            "weight_reset_in": max(0, self._rate_limit_weight_reset - int(time.time())),
            "order_used": self._rate_limit_order_used,
            "order_limit": self.RATE_LIMIT_ORDERS,
            "order_reset_in": max(0, self._rate_limit_order_reset - int(time.time())),
        }

    async def ping(self) -> bool:
        """
        Test connectivity to Binance API.

        Returns:
            True if connected, False otherwise
        """
        try:
            await self._request("GET", "/api/v3/ping")
            return True
        except Exception:
            return False

    async def get_server_time(self) -> int:
        """
        Get Binance server time.

        Returns:
            Server time in milliseconds
        """
        data = await self._request("GET", "/api/v3/time")
        return data["serverTime"]


# ==================== Factory Function ====================

def create_binance_client(
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
    **kwargs,
) -> BinanceClient:
    """
    Create and return a configured BinanceClient instance.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Use testnet if True
        **kwargs: Additional arguments for BinanceClient

    Returns:
        Configured BinanceClient instance
    """
    return BinanceClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        **kwargs,
    )


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        """Example usage of BinanceClient."""
        # Create client
        client = BinanceClient(testnet=True)

        # Test connectivity
        if await client.ping():
            logger.info("Connected to Binance API")
        else:
            logger.error("Failed to connect to Binance API")
            return

        # Get server time
        server_time = await client.get_server_time()
        logger.info(f"Server time: {server_time}")

        # Get exchange info
        exchange_info = await client.get_exchange_info()
        logger.info(f"Exchange info loaded: {len(exchange_info.get('symbols', []))} symbols")

        # Get ticker price
        btc_price = await client.get_ticker_price("BTCUSDT")
        logger.info(f"BTC/USDT Price: {btc_price}")

        # Get klines
        df = await client.get_klines("BTCUSDT", "1h", limit=10)
        logger.info(f"Klines data:\n{df.tail()}")

        # Get all timeframe klines
        all_tf = await client.get_klines_for_all_timeframes("BTCUSDT", limit=5)
        for tf, data in all_tf.items():
            if not data.empty:
                logger.info(f"{tf}: {len(data)} candles, last close: {data['close'].iloc[-1]}")

        # Cleanup
        await client.close()

    # Run example
    asyncio.run(m