```py
"""
Crypto Trading Bot - Main Application
Professional crypto trading bot with Flask backend, DeepSeek V4 AI integration,
Binance API, multi-timeframe technical analysis, chat intelligence, and dark trading terminal UI.

Author: Trading Bot Team
Version: 1.0.0
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal, ROUND_DOWN
from functools import wraps
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import talib
import requests
from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, flash, send_from_directory
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.websockets import BinanceSocketManager
import ccxt
import websocket
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, JSON, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import redis
from celery import Celery
from celery.schedules import crontab

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
login_manager = LoginManager(app)
login_manager.login_view = 'login'
CORS(app)

# Initialize Celery
celery = Celery(app.name, broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))
celery.conf.update(app.config)

# ============================================================================
# Database Models
# ============================================================================

class User(UserMixin, db.Model):
    """User model for authentication and preferences."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    trading_enabled = Column(Boolean, default=False)
    max_risk_per_trade = Column(Float, default=2.0)
    preferred_exchange = Column(String(50), default='binance')
    
    # Relationships
    api_keys = relationship('APIKey', back_populates='user', lazy='dynamic')
    trades = relationship('Trade', back_populates='user', lazy='dynamic')
    signals = relationship('Signal', back_populates='user', lazy='dynamic')
    chat_history = relationship('ChatMessage', back_populates='user', lazy='dynamic')
    
    def set_password(self, password: str):
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Verify password against hash."""
        return check_password_hash(self.password_hash, password)


class APIKey(db.Model):
    """Store encrypted API keys for exchanges."""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    exchange = Column(String(50), nullable=False)
    api_key = Column(String(256), nullable=False)
    api_secret = Column(String(512), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='api_keys')


class Trade(db.Model):
    """Record of executed trades."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # BUY or SELL
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, etc.
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    fee_asset = Column(String(10))
    status = Column(String(20), default='PENDING')
    exchange_order_id = Column(String(100))
    exchange = Column(String(50), default='binance')
    signal_id = Column(Integer, ForeignKey('signals.id'))
    pnl = Column(Float)
    pnl_percent = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    user = relationship('User', back_populates='trades')
    signal = relationship('Signal', back_populates='trades')


class Signal(db.Model):
    """Trading signals generated by AI and technical analysis."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(20), nullable=False)  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    strength = Column(Float, default=0.0)  # 0-100
    timeframe = Column(String(10), nullable=False)
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    confidence = Column(Float, default=0.0)
    indicators = Column(JSON)
    ai_analysis = Column(Text)
    is_active = Column(Boolean, default=True)
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='signals')
    trades = relationship('Trade', back_populates='signal', lazy='dynamic')


class ChatMessage(db.Model):
    """Store chat history with AI assistant."""
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    context = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='chat_history')


class MarketData(db.Model):
    """Store historical market data for analysis."""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TechnicalIndicators:
    """Container for technical analysis indicators."""
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    atr: float = 0.0
    obv: float = 0.0
    volume_sma: float = 0.0
    support_levels: List[float] = None
    resistance_levels: List[float] = None


@dataclass
class SignalScore:
    """Scoring system for trading signals."""
    overall_score: float = 0.0
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    volume_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    risk_score: float = 0.0
    confidence_level: str = 'LOW'
    recommendation: str = 'HOLD'


# ============================================================================
# Exchange Client Manager
# ============================================================================

class ExchangeManager:
    """Manage connections to cryptocurrency exchanges."""
    
    def __init__(self):
        self.clients: Dict[str, Any] = {}
        self.websockets: Dict[str, Any] = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize exchange clients with API keys."""
        try:
            # Binance
            binance_api_key = os.getenv('BINANCE_API_KEY')
            binance_api_secret = os.getenv('BINANCE_API_SECRET')
            if binance_api_key and binance_api_secret:
                self.clients['binance'] = BinanceClient(binance_api_key, binance_api_secret)
                logger.info("Binance client initialized successfully")
            
            # Additional exchanges via CCXT
            for exchange_id in ['coinbase', 'kraken', 'ftx']:
                api_key = os.getenv(f'{exchange_id.upper()}_API_KEY')
                api_secret = os.getenv(f'{exchange_id.upper()}_API_SECRET')
                if api_key and api_secret:
                    exchange_class = getattr(ccxt, exchange_id)
                    self.clients[exchange_id] = exchange_class({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True
                    })
                    logger.info(f"{exchange_id.capitalize()} client initialized")
        
        except Exception as e:
            logger.error(f"Error initializing exchange clients: {e}")
    
    def get_client(self, exchange: str = 'binance') -> Any:
        """Get exchange client by name."""
        client = self.clients.get(exchange)
        if not client:
            raise ValueError(f"Exchange {exchange} not configured")
        return client
    
    def get_balance(self, exchange: str = 'binance') -> Dict[str, float]:
        """Get account balance from exchange."""
        try:
            client = self.get_client(exchange)
            if exchange == 'binance':
                account = client.get_account()
                balances = {}
                for balance in account['balances']:
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    if free > 0 or locked > 0:
                        balances[balance['asset']] = {
                            'free': free,
                            'locked': locked,
                            'total': free + locked
                        }
                return balances
            else:
                balance = client.fetch_balance()
                return {k: v for k, v in balance['total'].items() if v > 0}
        
        except Exception as e:
            logger.error(f"Error fetching balance from {exchange}: {e}")
            return {}
    
    def get_ticker(self, symbol: str, exchange: str = 'binance') -> Dict[str, Any]:
        """Get current ticker for symbol."""
        try:
            client = self.get_client(exchange)
            if exchange == 'binance':
                ticker = client.get_symbol_ticker(symbol=symbol)
                return ticker
            else:
                return client.fetch_ticker(symbol)
        
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}


# ============================================================================
# Technical Analysis Engine
# ============================================================================

class TechnicalAnalysisEngine:
    """Advanced technical analysis with multiple timeframe support."""
    
    TIMEFRAMES = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '12h': '12h',
        '1d': '1d',
        '1w': '1w'
    }
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.cache = {}
        self.cache_timeout = 60  # seconds
    
    def get_klines(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch and cache kline data."""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            client = self.exchange_manager.get_client('binance')
            klines = client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Cache the data
            self.cache[cache_key] = (df, time.time())
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate comprehensive technical indicators."""
        if df.empty:
            return TechnicalIndicators()
        
        try:
            indicators = TechnicalIndicators()
            
            # RSI
            indicators.rsi = talib.RSI(df['close'].values, timeperiod=14)[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            indicators.macd = macd[-1]
            indicators.macd_signal = macd_signal[-1]
            indicators.macd_histogram = macd_hist[-1]
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                df['close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            indicators.bollinger_upper = upper[-1]
            indicators.bollinger_middle = middle[-1]
            indicators.bollinger_lower = lower[-1]
            
            # Moving Averages
            indicators.sma_20 = talib.SMA(df['close'].values, timeperiod=20)[-1]
            indicators.sma_50 = talib.SMA(df['close'].values, timeperiod=50)[-1]
            indicators.sma_200 = talib.SMA(df['close'].values, timeperiod=200)[-1]
            indicators.ema_12 = talib.EMA(df['close'].values, timeperiod=12)[-1]
            indicators.ema_26 = talib.EMA(df['close'].values, timeperiod=26)[-1]
            
            # ATR
            indicators.atr = talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=14
            )[-1]
            
            # OBV
            indicators.obv = talib.OBV(df['close'].values, df['volume'].values)[-1]
            
            # Volume SMA
            indicators.volume_sma = talib.SMA(df['volume'].values, timeperiod=20)[-1]
            
            # Support and Resistance Levels
            indicators.support_levels = self._find_support_levels(df)
            indicators.resistance_levels = self._find_resistance_levels(df)
            
            return indicators
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return TechnicalIndicators()
    
    def _find_support_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[float]:
        """Identify support levels using pivot points."""
        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Simple pivot point calculation
            pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
            support1 = (2 * pivot) - highs[-1]
            support2 = pivot - (highs[-1] - lows[-1])
            support3 = lows[-1] - 2 * (highs[-1] - pivot)
            
            return sorted([support1, support2, support3])[:num_levels]
        
        except Exception as e:
            logger.error(f"Error finding support levels: {e}")
            return []
    
    def _find_resistance_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[float]:
        """Identify resistance levels using pivot points."""
        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Simple pivot point calculation
            pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
            resistance1 = (2 * pivot) - lows[-1]
            resistance2 = pivot + (highs[-1] - lows[-1])
            resistance3 = highs[-1] + 2 * (pivot - lows[-1])
            
            return sorted([resistance1, resistance2, resistance3], reverse=True)[:num_levels]
        
        except Exception as e:
            logger.error(f"Error finding resistance levels: {e}")
            return []
    
    def analyze_multi_timeframe(self, symbol: str) -> Dict[str, TechnicalIndicators]:
        """Analyze symbol across multiple timeframes."""
        results = {}
        
        for tf_name, tf_interval in self.TIMEFRAMES.items():
            df = self.get_klines(symbol, tf_interval)
            if not df.empty:
                indicators = self.calculate_indicators(df)
                results[tf_name] = indicators
        
        return results


# ============================================================================
# Signal Scoring System
# ============================================================================

class SignalScoring:
    """Comprehensive signal scoring system."""
    
    def __init__(self):
        self.weights = {
            'technical': 0.35,
            'sentiment': 0.20,
            'volume': 0.15,
            'momentum': 0.15,
            'volatility': 0.10,
            'risk': 0.05
        }
    
    def score_technical(self, indicators: TechnicalIndicators) -> float:
        """Score based on technical indicators."""
        score = 50.0  # Neutral starting point
        
        # RSI scoring
        if indicators.rsi < 30:
            score += 20  # Oversold - bullish
        elif indicators.rsi > 70:
            score -= 20  # Overbought - bearish
        elif 30 <= indicators.rsi <= 70:
            score += 10  # Neutral zone
        
        # MACD scoring
        if indicators.macd > indicators.macd_signal:
            score += 15  # Bullish crossover
        else:
            score -= 15  # Bearish crossover
        
        # Bollinger Bands scoring
        current_price = indicators.bollinger_middle
        if current_price < indicators.bollinger_lower:
            score += 15  # Price below lower band - potential bounce
        elif current_price > indicators.bollinger_upper:
            score -= 15  # Price above upper band - potential reversal
        
        # Moving Average scoring
        if indicators.sma_20 > indicators.sma_50:
            score += 10  # Golden cross potential
        if indicators.sma_50 > indicators.sma_200:
            score += 10  # Long-term bullish
        
        return max(0, min(100, score))
    
    def score_volume(self, df: pd.DataFrame) -> float:
        """Score based on volume analysis."""
        if df.empty:
            return 50.0
        
        score = 50.0
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            score += 20  # High volume confirms trend
        elif current_volume > avg_volume * 1.2:
            score += 10
        elif current_volume < avg_volume * 0.5:
            score -= 15  # Low volume suggests weak trend
        
        return max(0, min(100, score))
    
    def score_momentum(self, df: pd.DataFrame) -> float:
        """Score based on momentum indicators."""
        if df.empty:
            return 50.0
        
        score = 50.0
        
        # Rate of Change
        roc = ((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]) * 100
        
        if roc > 5:
            score += 20  # Strong positive momentum
        elif roc > 2:
            score += 10
        elif roc < -5:
            score -= 20  # Strong negative momentum
        elif roc < -2:
            score -= 10
        
        # Price vs SMA
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        if df['close'].iloc[-1] > sma_20 * 1.05:
            score += 10
        elif df['close'].iloc[-1] < sma_20 * 0.95:
            score -= 10
        
        return max(0, min(100, score))
    
    def score_volatility(self, df: pd.DataFrame) -> float:
        """Score based on volatility assessment."""
        if df.empty:
            return 50.0
        
        score = 50.0
        
        # Calculate volatility using standard deviation
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        if volatility > 0.5:
            score -= 20  # High volatility - risky
        elif volatility > 0.3:
            score -= 10
        elif volatility < 0.2:
            score += 15  # Low volatility - stable
        elif volatility < 0.1:
            score += 25  # Very low volatility
        
        return max(0, min(100, score))
    
    def score_risk(self, indicators: TechnicalIndicators) -> float:
        """Score based on risk assessment."""
        score = 50.0
        
        # ATR-based risk assessment
        if indicators.atr > 0:
            atr_percent = (indicators.atr / indicators.bollinger_middle) * 100
            if atr_percent > 5:
                score -= 20  # High risk
            elif atr_percent > 3:
                score -= 10
            elif atr_percent < 1:
                score += 15  # Low risk
        
        # Distance to support/resistance
        if indicators.support_levels and indicators.resistance_levels:
            nearest_support = min(indicators.support_levels, key=lambda x: abs(x - indicators.bollinger_middle))
            nearest_resistance = min(indicators.resistance_levels, key=lambda x: abs(x - indicators.bollinger_middle))
            
            distance_to_support = abs(indicators.bollinger_middle - nearest_support) / indicators.bollinger_middle * 100
            distance_to_resistance = abs(indicators.bollinger_middle - nearest_resistance) / indicators.bollinger_middle * 100
            
            if distance_to_support < 2:
                score += 10  # Close to support - potential bounce
            if distance_to_resistance < 2:
                score -= 10  # Close to resistance - potential rejection
        
        return max(0, min(100, score))
    
    def calculate_overall_score(self, indicators: TechnicalIndicators, df: pd.DataFrame) -> SignalScore:
        """Calculate comprehensive signal score."""
        score = SignalScore()
        
        # Calculate individual scores
        score.technical_score = self.score_technical(indicators)
        score.volume_score = self.score_volume(df)
        score.momentum_score = self.score_momentum(df)
        score.volatility_score = self.score_volatility(df)
        score.risk_score = self.score_risk(indicators)
        
        # Weighted average
        score.overall_score = (
            score.technical_score * self.weights['technical'] +
            score.volume_score * self.weights['volume'] +
            score.momentum_score * self.weights['momentum'] +
            score.volatility_score * self.weights['volatility'] +
            score.risk_score * self.weights['risk']
        )
        
        # Determine confidence level
        if score.overall_score >= 80:
            score.confidence_level = 'VERY_HIGH'
        elif score.overall_score >= 65:
            score.confidence_level = 'HIGH'
        elif score.overall_score >= 50:
            score.confidence_level = 'MEDIUM'
        elif score.overall_score >= 35:
            score.confidence_level = 'LOW'
        else:
            score.confidence_level = 'VERY_LOW'
        
        # Generate recommendation
        if score.overall_score >= 70:
            score.recommendation = 'STRONG_BUY'
        elif score.overall_score >= 55:
            score.recommendation = 'BUY'
        elif score.overall_score >= 45:
            score.recommendation = 'HOLD'
        elif score.overall_score >= 30:
            score.recommendation = 'SELL'
        else:
            score.recommendation = 'STRONG_SELL'
        
        return score


# ============================================================================
# DeepSeek AI Integration
# ============================================================================

class DeepSeekAIClient:
    """Integration with DeepSeek V4 AI for market analysis and chat."""
    
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')
        self.model = 'deepseek-chat-v4'
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
    
    def analyze_market(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered market analysis."""
        try:
            prompt = self._build_market_analysis_prompt(symbol, market_data)
            response = self._call_api(prompt)
            return self._parse_market_analysis(response)
        
        except Exception as e:
            logger.error(f"Error in AI market analysis: {e}")
            return {'error': str(e), 'analysis': 'Analysis unavailable'}
    
    def chat(self, message: str, context: Dict[str, Any] = None) -> str:
        """Chat with AI assistant about trading."""
        try:
            prompt = self._build_chat_prompt(message, context)
            response = self._call_api(prompt)
            return response.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        except Exception as e:
            logger.error(f"Error in AI chat: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    def _call_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to DeepSeek."""
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are an expert cryptocurrency trading assistant with deep knowledge of technical analysis, market dynamics, and risk management.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        response = self.session.post(
            f'{self.api_url}/chat/completions',
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def _build_market_analysis_prompt(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Build prompt for market analysis."""
        return f"""
        Analyze the following cryptocurrency market data for {symbol}:
        
        Current Price: {market_data.get('price', 'N/A')}
        24h Change: {market_data.get('change_24h', 'N/A')}%
        24h Volume: {market_data.get('volume_24h', 'N/A')}
        
        Technical Indicators:
        - RSI: {market_data.get('rsi', 'N/A')}
        - MACD: {market_data.get('macd', 'N/A')}
        - Bollinger Bands: Upper={market_data.get('bb_upper', 'N/A')}, Lower={market_data.get('bb_lower', 'N/A')}
        - Support Levels: {market_data.get('support_levels', 'N/A')}
        - Resistance Levels: {market_data.get('resistance_levels', 'N/A')}
        
        Market Sentiment: {market_data.get('sentiment', 'N/A')}
        
        Please provide:
        1. Technical analysis summary
        2. Key support and resistance levels
        3. Potential entry and exit points
        4. Risk assessment
        5. Trading recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        6. Confidence level and reasoning
        """
    
    def _build_chat_prompt(self, message: str, context: Dict[str, Any] = None) -> str:
        """Build prompt for chat interaction."""
        context_str = ""
        if context:
            context_str = f"""
            Current Context:
            - Active Symbols: {context.get('symbols', [])}
            - Portfolio Value: ${context.get('portfolio_value', 'N/A')}
            - Open Positions: {context.get('open_positions', 0)}
            - Recent Trades: {context.get('recent_trades', [])}
            """
        
        return f"""
        {context_str}
        
        User Message: {message}
        
        Please provide a helpful response with trading insights, analysis, or general cryptocurrency information.
        """
    
    def _parse_market_analysis(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response into structured analysis."""
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract structured data from response
            analysis = {
                'raw_analysis': content,
                'recommendation': self._extract_recommendation(content),
                'confidence': self._extract_confidence(content),
                'key_levels': self._extract_key_levels(content),
                'risk_assessment': self._extract_risk_assessment(content)
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error parsing AI analysis: {e}")
            return {'raw_analysis': str(response), 'error': str(e)}
    
    def _extract_recommendation(self, content: str) -> str:
        """Extract trading recommendation from AI response."""
        recommendations = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']
        content_upper = content.upper()
        
        for rec in recommendations:
            if rec in content_upper:
                return rec
        
        return 'HOLD'
    
    def _extract_confidence(self, content: str) -> str:
        """Extract confidence level from AI response."""
        confidence_levels = ['VERY HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY LOW']
        content_upper = content.upper()
        
        for level in confidence_levels:
            if level in content_upper:
                return level
        
        return 'MEDIUM'
    
    def _extract_key_levels(self, content: str) -> Dict[str, List[float]]:
        """Extract key price levels from AI response."""
        # Simple extraction - in production, use more sophisticated parsing
        levels = {'support': [], 'resistance': []}
        
        import re
        # Find numbers that could be price levels
        numbers = re.findall(r'\$?(\d+\.?\d*)', content)
        
        # Categorize based on context
        for num in numbers[:10]:  # Limit to first 10 numbers
            try:
                price = float(num)
                if 'support' in content.lower() and price > 0:
                    levels['support'].append(price)
                elif 'resistance' in content.lower() and price > 0:
                    levels['resistance'].append(price)
            except ValueError:
                continue
        
        return levels
    
    def _extract_risk_assessment(self, content: str) -> str:
        """Extract risk assessment from AI response."""
        risk_keywords = {
            'HIGH': ['high risk', 'risky', 'dangerous', 'volatile'],
            'MEDIUM': ['moderate risk', 'medium risk', 'balanced'],
            'LOW': ['low risk', 'safe', 'conservative', 'stable']
        }
        
        content_lower = content.lower()
        
        for level, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return level