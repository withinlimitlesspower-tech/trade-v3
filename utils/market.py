"""
Market Context Analysis Module
==============================
Provides comprehensive market analysis including BTC correlation,
sector classification, volume-based filtering, and market overview generation.

This module serves as the core market intelligence layer for the trading bot,
integrating with Binance API and DeepSeek V4 AI for enhanced analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy import stats

from utils.config import Config
from utils.cache import CacheManager
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class MarketSector(Enum):
    """Crypto market sectors for classification."""
    LAYER_1 = "Layer 1"
    LAYER_2 = "Layer 2"
    DEFI = "DeFi"
    NFT = "NFT/Gaming"
    MEME = "Meme"
    STABLECOIN = "Stablecoin"
    EXCHANGE = "Exchange Token"
    AI_BIG_DATA = "AI & Big Data"
    PRIVACY = "Privacy"
    INFRASTRUCTURE = "Infrastructure"
    METAVERSE = "Metaverse"
    OTHER = "Other"


@dataclass
class MarketOverview:
    """Comprehensive market overview data structure."""
    timestamp: datetime
    total_market_cap: float
    btc_dominance: float
    top_gainers: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]
    high_volume_coins: List[Dict[str, Any]]
    sector_performance: Dict[str, float]
    btc_correlation_matrix: Dict[str, float]
    market_sentiment: str
    fear_greed_index: Optional[int] = None
    total_volume_24h: float = 0.0
    active_coins: int = 0
    summary: str = ""


@dataclass
class CoinAnalysis:
    """Individual coin analysis data."""
    symbol: str
    price: float
    volume_24h: float
    market_cap: float
    sector: MarketSector
    btc_correlation: float
    price_change_24h: float
    volatility: float
    liquidity_score: float
    rank: int = 0
    is_top_volume: bool = False


class MarketAnalyzer:
    """
    Advanced market analysis engine for cryptocurrency markets.
    
    Features:
    - BTC correlation analysis for all tracked coins
    - Sector classification based on project type
    - Top volume coins filtering with configurable thresholds
    - Real-time market overview generation
    - Fear & Greed index integration
    - Market sentiment analysis
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        config: Config,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        Initialize the MarketAnalyzer.
        
        Args:
            binance_client: Authenticated Binance client instance
            config: Application configuration
            cache_manager: Optional cache manager for performance optimization
        """
        self.client = binance_client
        self.config = config
        self.cache = cache_manager or CacheManager()
        
        # Configuration parameters
        self.top_volume_limit = config.get('market.top_volume_limit', 50)
        self.min_volume_usdt = config.get('market.min_volume_usdt', 100000)
        self.correlation_window = config.get('market.correlation_window', 30)
        self.update_interval = config.get('market.update_interval', 60)
        
        # Sector classification mapping
        self.sector_map = self._initialize_sector_map()
        
        # Cache keys
        self.cache_keys = {
            'market_overview': 'market:overview',
            'top_volume': 'market:top_volume',
            'correlation': 'market:correlation',
            'sectors': 'market:sectors'
        }
        
        logger.info("MarketAnalyzer initialized successfully")

    def _initialize_sector_map(self) -> Dict[str, MarketSector]:
        """
        Initialize the sector classification mapping for known tokens.
        
        Returns:
            Dictionary mapping token symbols to their market sectors
        """
        return {
            # Layer 1
            'BTC': MarketSector.LAYER_1,
            'ETH': MarketSector.LAYER_1,
            'SOL': MarketSector.LAYER_1,
            'ADA': MarketSector.LAYER_1,
            'AVAX': MarketSector.LAYER_1,
            'DOT': MarketSector.LAYER_1,
            'MATIC': MarketSector.LAYER_1,
            'NEAR': MarketSector.LAYER_1,
            'FTM': MarketSector.LAYER_1,
            'ATOM': MarketSector.LAYER_1,
            
            # Layer 2
            'ARB': MarketSector.LAYER_2,
            'OP': MarketSector.LAYER_2,
            'MATIC': MarketSector.LAYER_2,
            'IMX': MarketSector.LAYER_2,
            'LRC': MarketSector.LAYER_2,
            
            # DeFi
            'UNI': MarketSector.DEFI,
            'AAVE': MarketSector.DEFI,
            'COMP': MarketSector.DEFI,
            'MKR': MarketSector.DEFI,
            'CRV': MarketSector.DEFI,
            'SUSHI': MarketSector.DEFI,
            'CAKE': MarketSector.DEFI,
            'LINK': MarketSector.DEFI,
            
            # NFT/Gaming
            'SAND': MarketSector.NFT,
            'MANA': MarketSector.NFT,
            'AXS': MarketSector.NFT,
            'ENJ': MarketSector.NFT,
            'CHZ': MarketSector.NFT,
            
            # Meme
            'DOGE': MarketSector.MEME,
            'SHIB': MarketSector.MEME,
            'PEPE': MarketSector.MEME,
            'FLOKI': MarketSector.MEME,
            
            # Exchange
            'BNB': MarketSector.EXCHANGE,
            'OKB': MarketSector.EXCHANGE,
            'FTT': MarketSector.EXCHANGE,
            'KCS': MarketSector.EXCHANGE,
            
            # AI & Big Data
            'FET': MarketSector.AI_BIG_DATA,
            'AGIX': MarketSector.AI_BIG_DATA,
            'OCEAN': MarketSector.AI_BIG_DATA,
            'GRT': MarketSector.AI_BIG_DATA,
            
            # Privacy
            'XMR': MarketSector.PRIVACY,
            'ZEC': MarketSector.PRIVACY,
            'DASH': MarketSector.PRIVACY,
            
            # Infrastructure
            'FIL': MarketSector.INFRASTRUCTURE,
            'ICP': MarketSector.INFRASTRUCTURE,
            'THETA': MarketSector.INFRASTRUCTURE,
            
            # Metaverse
            'APE': MarketSector.METAVERSE,
            'GALA': MarketSector.METAVERSE,
            'ILV': MarketSector.METAVERSE,
        }

    def classify_sector(self, symbol: str) -> MarketSector:
        """
        Classify a cryptocurrency into its market sector.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            MarketSector enum value
        """
        # Extract base symbol
        base_symbol = self._extract_base_symbol(symbol)
        
        # Check direct mapping
        if base_symbol in self.sector_map:
            return self.sector_map[base_symbol]
        
        # Attempt heuristic classification
        return self._heuristic_sector_classification(base_symbol)

    def _extract_base_symbol(self, symbol: str) -> str:
        """
        Extract the base cryptocurrency symbol from a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHBTC')
            
        Returns:
            Base symbol
        """
        # Remove common quote currencies
        for quote in ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']:
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return symbol[:-len(quote)]
        return symbol

    def _heuristic_sector_classification(self, symbol: str) -> MarketSector:
        """
        Heuristic-based sector classification for unknown tokens.
        
        Args:
            symbol: Base cryptocurrency symbol
            
        Returns:
            Best guess MarketSector
        """
        # Check for common patterns
        if any(keyword in symbol.lower() for keyword in ['swap', 'dex', 'lend', 'farm', 'yield']):
            return MarketSector.DEFI
        
        if any(keyword in symbol.lower() for keyword in ['game', 'play', 'hero', 'war']):
            return MarketSector.NFT
        
        if any(keyword in symbol.lower() for keyword in ['ai', 'gpt', 'brain', 'mind']):
            return MarketSector.AI_BIG_DATA
        
        if any(keyword in symbol.lower() for keyword in ['privacy', 'secret', 'anon']):
            return MarketSector.PRIVACY
        
        return MarketSector.OTHER

    async def get_btc_correlation(
        self,
        symbols: List[str],
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate BTC correlation for a list of symbols.
        
        Args:
            symbols: List of trading pair symbols
            window: Correlation calculation window in days
            
        Returns:
            Dictionary mapping symbols to their BTC correlation coefficient
        """
        window = window or self.correlation_window
        
        # Check cache first
        cache_key = f"{self.cache_keys['correlation']}:{hash(frozenset(symbols))}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Fetch historical data for BTC
            btc_data = await self._get_historical_prices('BTCUSDT', window)
            if not btc_data:
                logger.warning("Unable to fetch BTC historical data")
                return {}
            
            btc_returns = self._calculate_returns(btc_data)
            
            correlations = {}
            for symbol in symbols:
                try:
                    symbol_data = await self._get_historical_prices(symbol, window)
                    if symbol_data and len(symbol_data) == len(btc_data):
                        symbol_returns = self._calculate_returns(symbol_data)
                        
                        # Calculate Pearson correlation
                        correlation, _ = stats.pearsonr(btc_returns, symbol_returns)
                        correlations[symbol] = round(correlation, 4)
                    else:
                        correlations[symbol] = 0.0
                        
                except Exception as e:
                    logger.error(f"Error calculating correlation for {symbol}: {e}")
                    correlations[symbol] = 0.0
            
            # Cache results
            await self.cache.set(cache_key, correlations, ttl=3600)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error in BTC correlation calculation: {e}")
            return {}

    async def _get_historical_prices(
        self,
        symbol: str,
        days: int
    ) -> List[float]:
        """
        Fetch historical price data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days of historical data
            
        Returns:
            List of closing prices
        """
        try:
            klines = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.get_klines(
                    symbol=symbol,
                    interval=BinanceClient.KLINE_INTERVAL_1DAY,
                    limit=days
                )
            )
            
            return [float(kline[4]) for kline in klines]  # Closing prices
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return []

    def _calculate_returns(self, prices: List[float]) -> np.ndarray:
        """
        Calculate percentage returns from price data.
        
        Args:
            prices: List of price values
            
        Returns:
            Numpy array of percentage returns
        """
        if len(prices) < 2:
            return np.array([])
        
        prices_array = np.array(prices, dtype=float)
        returns = np.diff(prices_array) / prices_array[:-1]
        return returns

    async def get_top_volume_coins(
        self,
        limit: Optional[int] = None,
        min_volume: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top volume coins filtered by minimum volume threshold.
        
        Args:
            limit: Maximum number of coins to return
            min_volume: Minimum 24h volume in USDT
            
        Returns:
            List of dictionaries containing coin data sorted by volume
        """
        limit = limit or self.top_volume_limit
        min_volume = min_volume or self.min_volume_usdt
        
        # Check cache
        cache_key = f"{self.cache_keys['top_volume']}:{limit}:{min_volume}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Fetch ticker prices
            tickers = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_ticker
            )
            
            # Filter USDT pairs and sort by volume
            usdt_pairs = [
                ticker for ticker in tickers
                if ticker['symbol'].endswith('USDT')
                and float(ticker['quoteVolume']) >= min_volume
            ]
            
            # Sort by 24h volume descending
            sorted_pairs = sorted(
                usdt_pairs,
                key=lambda x: float(x['quoteVolume']),
                reverse=True
            )[:limit]
            
            # Enrich with additional data
            enriched_coins = []
            for ticker in sorted_pairs:
                coin_data = {
                    'symbol': ticker['symbol'],
                    'price': float(ticker['lastPrice']),
                    'volume_24h': float(ticker['quoteVolume']),
                    'price_change_24h': float(ticker['priceChangePercent']),
                    'high_24h': float(ticker['highPrice']),
                    'low_24h': float(ticker['lowPrice']),
                    'sector': self.classify_sector(ticker['symbol']).value,
                    'timestamp': datetime.now().isoformat()
                }
                enriched_coins.append(coin_data)
            
            # Cache results
            await self.cache.set(cache_key, enriched_coins, ttl=300)
            
            logger.info(f"Retrieved {len(enriched_coins)} top volume coins")
            return enriched_coins
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching top volume coins: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching top volume coins: {e}")
            return []

    async def get_sector_performance(
        self,
        top_coins: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for each market sector.
        
        Args:
            top_coins: Optional list of top coins data
            
        Returns:
            Dictionary mapping sector names to average performance
        """
        if not top_coins:
            top_coins = await self.get_top_volume_coins(limit=100)
        
        sector_performance = {}
        sector_counts = {}
        
        for coin in top_coins:
            sector = coin.get('sector', 'Other')
            price_change = coin.get('price_change_24h', 0)
            
            if sector not in sector_performance:
                sector_performance[sector] = 0.0
                sector_counts[sector] = 0
            
            sector_performance[sector] += price_change
            sector_counts[sector] += 1
        
        # Calculate averages
        for sector in sector_performance:
            if sector_counts[sector] > 0:
                sector_performance[sector] = round(
                    sector_performance[sector] / sector_counts[sector],
                    2
                )
        
        return sector_performance

    async def get_market_overview(self) -> MarketOverview:
        """
        Generate comprehensive market overview.
        
        Returns:
            MarketOverview dataclass with complete market analysis
        """
        # Check cache
        cached_overview = await self.cache.get(self.cache_keys['market_overview'])
        if cached_overview:
            return cached_overview
        
        try:
            # Fetch top volume coins
            top_coins = await self.get_top_volume_coins(limit=50)
            
            # Get BTC dominance and market cap
            btc_ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.get_ticker(symbol='BTCUSDT')
            )
            
            # Calculate total market cap (approximate from top coins)
            total_market_cap = sum(
                coin['price'] * await self._get_approximate_supply(coin['symbol'])
                for coin in top_coins[:20]  # Use top 20 for approximation
            )
            
            btc_market_cap = float(btc_ticker['lastPrice']) * 19_000_000  # Approximate BTC supply
            btc_dominance = (btc_market_cap / total_market_cap * 100) if total_market_cap > 0 else 0
            
            # Get top gainers and losers
            sorted_by_change = sorted(
                top_coins,
                key=lambda x: x['price_change_24h'],
                reverse=True
            )
            top_gainers = sorted_by_change[:5]
            top_losers = sorted_by_change[-5:][::-1]
            
            # Get sector performance
            sector_performance = await self.get_sector_performance(top_coins)
            
            # Calculate BTC correlations
            symbols = [coin['symbol'] for coin in top_coins[:20]]
            btc_correlations = await self.get_btc_correlation(symbols)
            
            # Calculate market sentiment
            market_sentiment = self._calculate_market_sentiment(top_coins)
            
            # Calculate total volume
            total_volume = sum(coin['volume_24h'] for coin in top_coins)
            
            # Generate summary
            summary = self._generate_market_summary(
                top_gainers,
                top_losers,
                sector_performance,
                btc_dominance,
                market_sentiment
            )
            
            overview = MarketOverview(
                timestamp=datetime.now(),
                total_market_cap=round(total_market_cap, 2),
                btc_dominance=round(btc_dominance, 2),
                top_gainers=top_gainers,
                top_losers=top_losers,
                high_volume_coins=top_coins[:10],
                sector_performance=sector_performance,
                btc_correlation_matrix=btc_correlations,
                market_sentiment=market_sentiment,
                total_volume_24h=round(total_volume, 2),
                active_coins=len(top_coins),
                summary=summary
            )
            
            # Cache overview
            await self.cache.set(
                self.cache_keys['market_overview'],
                overview,
                ttl=300
            )
            
            logger.info("Market overview generated successfully")
            return overview
            
        except Exception as e:
            logger.error(f"Error generating market overview: {e}")
            return MarketOverview(
                timestamp=datetime.now(),
                total_market_cap=0.0,
                btc_dominance=0.0,
                top_gainers=[],
                top_losers=[],
                high_volume_coins=[],
                sector_performance={},
                btc_correlation_matrix={},
                market_sentiment="neutral",
                summary="Unable to generate market overview"
            )

    async def _get_approximate_supply(self, symbol: str) -> float:
        """
        Get approximate circulating supply for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Approximate circulating supply
        """
        # This is a simplified approximation
        # In production, you'd use CoinGecko or similar API
        supply_estimates = {
            'BTCUSDT': 19_000_000,
            'ETHUSDT': 120_000_000,
            'BNBUSDT': 160_000_000,
            'SOLUSDT': 400_000_000,
            'ADAUSDT': 35_000_000_000,
            'DOTUSDT': 1_200_000_000,
            'AVAXUSDT': 350_000_000,
        }
        
        return supply_estimates.get(symbol, 1_000_000_000)

    def _calculate_market_sentiment(self, coins: List[Dict[str, Any]]) -> str:
        """
        Calculate overall market sentiment based on price movements.
        
        Args:
            coins: List of coin data dictionaries
            
        Returns:
            Market sentiment string ('bullish', 'bearish', 'neutral')
        """
        if not coins:
            return "neutral"
        
        # Calculate percentage of coins with positive price change
        positive_count = sum(
            1 for coin in coins
            if coin.get('price_change_24h', 0) > 0
        )
        
        positive_ratio = positive_count / len(coins)
        
        if positive_ratio > 0.6:
            return "bullish"
        elif positive_ratio < 0.4:
            return "bearish"
        else:
            return "neutral"

    def _generate_market_summary(
        self,
        gainers: List[Dict[str, Any]],
        losers: List[Dict[str, Any]],
        sector_performance: Dict[str, float],
        btc_dominance: float,
        sentiment: str
    ) -> str:
        """
        Generate a human-readable market summary.
        
        Args:
            gainers: Top gaining coins
            losers: Top losing coins
            sector_performance: Sector performance metrics
            btc_dominance: Current BTC dominance percentage
            sentiment: Market sentiment
            
        Returns:
            Formatted market summary string
        """
        summary_parts = []
        
        # Market sentiment
        summary_parts.append(f"Market sentiment is {sentiment.upper()}")
        
        # BTC dominance
        summary_parts.append(f"BTC dominance at {btc_dominance:.1f}%")
        
        # Top gainers
        if gainers:
            gainer_str = ", ".join(
                f"{g['symbol']}(+{g['price_change_24h']:.1f}%)"
                for g in gainers[:3]
            )
            summary_parts.append(f"Top gainers: {gainer_str}")
        
        # Top losers
        if losers:
            loser_str = ", ".join(
                f"{l['symbol']}({l['price_change_24h']:.1f}%)"
                for l in losers[:3]
            )
            summary_parts.append(f"Top losers: {loser_str}")
        
        # Best performing sector
        if sector_performance:
            best_sector = max(sector_performance, key=sector_performance.get)
            summary_parts.append(
                f"Best sector: {best_sector} ({sector_performance[best_sector]:.1f}%)"
            )
        
        return " | ".join(summary_parts)

    async def analyze_coin(self, symbol: str) -> Optional[CoinAnalysis]:
        """
        Perform comprehensive analysis on a single coin.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            CoinAnalysis dataclass with detailed metrics
        """
        try:
            # Fetch ticker data
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.get_ticker(symbol=symbol)
            )
            
            # Calculate BTC correlation
            btc_corr = await self.get_btc_correlation([symbol])
            correlation = btc_corr.get(symbol, 0.0)
            
            # Calculate volatility (using 30-day data)
            prices = await self._get_historical_prices(symbol, 30)
            volatility = np.std(self._calculate_returns(prices)) if len(prices) > 1 else 0.0
            
            # Calculate liquidity score (based on volume and spread)
            volume = float(ticker['quoteVolume'])
            liquidity_score = min(volume / 1_000_000, 10)  # Normalize to 0-10 scale
            
            # Classify sector
            sector = self.classify_sector(symbol)
            
            return CoinAnalysis(
                symbol=symbol,
                price=float(ticker['lastPrice']),
                volume_24h=volume,
                market_cap=float(ticker['lastPrice']) * await self._get_approximate_supply(symbol),
                sector=sector,
                btc_correlation=correlation,
                price_change_24h=float(ticker['priceChangePercent']),
                volatility=round(volatility, 4),
                liquidity_score=round(liquidity_score, 2)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing coin {symbol}: {e}")
            return None

    async def get_market_heatmap(self) -> Dict[str, Any]:
        """
        Generate market heatmap data for visualization.
        
        Returns:
            Dictionary with heatmap data organized by sector
        """
        top_coins = await self.get_top_volume_coins(limit=100)
        
        heatmap_data = {}
        for sector in MarketSector:
            sector_coins = [
                coin for coin in top_coins
                if coin.get('sector') == sector.value
            ]
            
            if sector_coins:
                heatmap_data[sector.value] = {
                    'coins': sector_coins,
                    'average_change': round(
                        sum(c['price_change_24h'] for c in sector_coins) / len(sector_coins),
                        2
                    ),
                    'total_volume': round(
                        sum(c['volume_24h'] for c in sector_coins),
                        2
                    ),
                    'count': len(sector_coins)
                }
        
        return heatmap_data

    async def refresh_data(self) -> bool:
        """
        Force refresh of all cached market data.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Clear all market-related cache
            for key in self.cache_keys.values():
                await self.cache.delete(key)
            
            # Regenerate market overview
            await self.get_market_overview()
            
            logger.info("Market data refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing market data: {e}")
            return False


# Factory function for creating MarketAnalyzer instances
def create_market_analyzer(
    binance_client: BinanceClient,
    config: Config,
    cache_manager: Optional[CacheManager] = None
) -> MarketAnalyzer:
    """
    Create and configure a MarketAnalyzer instance.
    
    Args:
        binance_client: Authenticated Binance client
        config: Application configuration
        cache_manager: Optional cache manager
        
    Returns:
        Configured MarketAnalyzer instance
    """
    return MarketAnalyzer(
        binance_client=binance_client,
        config=config,
        cache_manager=cache_manager
    )


# Export public API
__all__ = [
    'MarketAnalyzer',
    'MarketOverview',
    'CoinAnalysis',
    'MarketSector',
    'create_market_analyzer'
]