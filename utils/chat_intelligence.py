```py
"""
Chat Intelligence Module - Intent Parser and Context Manager
============================================================

This module provides advanced chat intelligence capabilities for a crypto trading bot,
including multilingual intent parsing (Urdu/Hindi/English), context memory management,
conversation history tracking, and user preference learning.

Key Features:
- Multilingual intent parsing (English, Urdu, Hindi)
- Context-aware conversation memory
- User preference learning and adaptation
- Conversation history management with persistence
- Trading intent detection and parameter extraction
- Sentiment analysis for market-related queries
"""

import json
import re
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
import time

# Third-party imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None
    logging.warning("NumPy not available. Some features may be limited.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Falling back to basic similarity.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Enumeration of supported chat intents."""
    TRADE = "trade"
    QUERY = "query"
    COMMAND = "command"
    GREETING = "greeting"
    FAREWELL = "farewell"
    HELP = "help"
    SETTINGS = "settings"
    ANALYSIS = "analysis"
    ALERT = "alert"
    PORTFOLIO = "portfolio"
    MARKET = "market"
    UNKNOWN = "unknown"


class Language(Enum):
    """Supported languages for intent parsing."""
    ENGLISH = "en"
    URDU = "ur"
    HINDI = "hi"
    ROMANIZED = "romanized"


@dataclass
class ChatMessage:
    """Represents a single chat message with metadata."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    language: Language = Language.ENGLISH
    intent: IntentType = IntentType.UNKNOWN
    confidence: float = 0.0
    user_id: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """User preferences learned over time."""
    preferred_language: Language = Language.ENGLISH
    trading_pairs: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    common_intents: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_interaction: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0


@dataclass
class ConversationContext:
    """Maintains context for ongoing conversations."""
    current_intent: Optional[IntentType] = None
    last_trade_params: Optional[Dict[str, Any]] = None
    mentioned_pairs: List[str] = field(default_factory=list)
    mentioned_timeframes: List[str] = field(default_factory=list)
    sentiment: float = 0.0
    urgency: float = 0.0
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    context_window: deque = field(default_factory=lambda: deque(maxlen=10))


class MultilingualIntentParser:
    """
    Advanced intent parser supporting English, Urdu, Hindi, and Romanized text.
    
    Uses pattern matching, keyword extraction, and ML-based classification
    to determine user intent from chat messages.
    """
    
    # Intent patterns for different languages
    INTENT_PATTERNS = {
        IntentType.TRADE: {
            Language.ENGLISH: [
                r'\b(buy|sell|trade|order|purchase|acquire)\b',
                r'\b(enter|exit|position)\b',
                r'\b(long|short)\b',
                r'\b(limit|market|stop)\s+(order|loss|limit)\b',
            ],
            Language.URDU: [
                r'\b(خرید|فروخت|تجارت|آرڈر)\b',
                r'\b(لانگ|شارٹ)\b',
                r'\b(مارکیٹ|محدود)\s+(آرڈر|نقصان)\b',
            ],
            Language.HINDI: [
                r'\b(खरीदें|बेचें|व्यापार|ऑर्डर)\b',
                r'\b(लॉन्ग|शॉर्ट)\b',
                r'\b(मार्केट|लिमिट)\s+(ऑर्डर|लॉस)\b',
            ],
        },
        IntentType.QUERY: {
            Language.ENGLISH: [
                r'\b(what|how|why|when|where|which)\b',
                r'\b(tell|show|explain|describe)\b',
                r'\b(price|rate|value)\s+of\b',
                r'\?\s*$',
            ],
            Language.URDU: [
                r'\b(کیا|کیسے|کیوں|کب|کہاں)\b',
                r'\b(بتائیں|دکھائیں|وضاحت)\b',
                r'\?\s*$',
            ],
            Language.HINDI: [
                r'\b(क्या|कैसे|क्यों|कब|कहां)\b',
                r'\b(बताएं|दिखाएं|समझाएं)\b',
                r'\?\s*$',
            ],
        },
        IntentType.COMMAND: {
            Language.ENGLISH: [
                r'^\s*(set|change|update|configure|enable|disable)\b',
                r'^\s*(start|stop|pause|resume)\b',
                r'^\s*(show|display|list|get)\b',
            ],
            Language.URDU: [
                r'^\s*(سیٹ|تبدیل|اپڈیٹ|کنفیگر)\b',
                r'^\s*(شروع|روک|موقوف|جاری)\b',
            ],
            Language.HINDI: [
                r'^\s*(सेट|बदलें|अपडेट|कॉन्फ़िगर)\b',
                r'^\s*(शुरू|रोकें|रोक|जारी)\b',
            ],
        },
        IntentType.GREETING: {
            Language.ENGLISH: [
                r'^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b',
                r'^\s*(what\'s\s+up|sup|howdy)\b',
            ],
            Language.URDU: [
                r'^\s*(السلام\s+علیکم|ہیلو|ہائے)\b',
                r'^\s*(آداب|سلام)\b',
            ],
            Language.HINDI: [
                r'^\s*(नमस्ते|नमस्कार|हैलो|हाय)\b',
                r'^\s*(सुप्रभात|शुभ\s+संध्या)\b',
            ],
        },
        IntentType.FAREWELL: {
            Language.ENGLISH: [
                r'^\s*(bye|goodbye|see\s+you|farewell|take\s+care)\b',
                r'^\s*(later|cya|adios)\b',
            ],
            Language.URDU: [
                r'^\s*(خدا\s+حافظ|الوداع|پھر\s+ملیں\s+گے)\b',
            ],
            Language.HINDI: [
                r'^\s*(अलविदा|फिर\s+मिलेंगे|नमस्ते)\b',
            ],
        },
        IntentType.HELP: {
            Language.ENGLISH: [
                r'^\s*(help|support|guide|tutorial|documentation)\b',
                r'^\s*(how\s+to|what\s+can\s+you\s+do)\b',
            ],
            Language.URDU: [
                r'^\s*(مدد|سپورٹ|رہنمائی)\b',
            ],
            Language.HINDI: [
                r'^\s*(मदद|सहायता|गाइड)\b',
            ],
        },
        IntentType.ANALYSIS: {
            Language.ENGLISH: [
                r'\b(analyze|analysis|chart|pattern|trend)\b',
                r'\b(technical|fundamental)\s+analysis\b',
                r'\b(RSI|MACD|EMA|SMA|Bollinger|Fibonacci)\b',
            ],
            Language.URDU: [
                r'\b(تجزیہ|چارٹ|پیٹرن|رجحان)\b',
                r'\b(تکنیکی|بنیادی)\s+تجزیہ\b',
            ],
            Language.HINDI: [
                r'\b(विश्लेषण|चार्ट|पैटर्न|प्रवृत्ति)\b',
                r'\b(तकनीकी|मौलिक)\s+विश्लेषण\b',
            ],
        },
        IntentType.ALERT: {
            Language.ENGLISH: [
                r'\b(alert|notify|remind|warn|notification)\b',
                r'\b(set|create)\s+(alert|notification)\b',
                r'\b(when|if)\s+(price|value)\s+(reaches|hits|crosses)\b',
            ],
            Language.URDU: [
                r'\b(الرٹ|اطلاع|یاد\s+دہانی|انتباہ)\b',
            ],
            Language.HINDI: [
                r'\b(अलर्ट|सूचना|याद\s+दिलाना|चेतावनी)\b',
            ],
        },
        IntentType.PORTFOLIO: {
            Language.ENGLISH: [
                r'\b(portfolio|holdings|balance|positions)\b',
                r'\b(my\s+)?(assets|investments|trades)\b',
                r'\b(profit|loss|PnL|returns)\b',
            ],
            Language.URDU: [
                r'\b(پورٹ فولیو|ہولڈنگز|بیلنس|پوزیشنز)\b',
            ],
            Language.HINDI: [
                r'\b(पोर्टफोलियो|होल्डिंग्स|बैलेंस|पोज़ीशन)\b',
            ],
        },
        IntentType.MARKET: {
            Language.ENGLISH: [
                r'\b(market|price|rate|value|ticker)\b',
                r'\b(bitcoin|ethereum|bnb|crypto)\b',
                r'\b(volume|liquidity|volatility)\b',
            ],
            Language.URDU: [
                r'\b(مارکیٹ|قیمت|ریٹ|ویلیو)\b',
                r'\b(بٹ کوائن|ایتھریم|کرپٹو)\b',
            ],
            Language.HINDI: [
                r'\b(मार्केट|कीमत|रेट|वैल्यू)\b',
                r'\b(बिटकॉइन|एथेरियम|क्रिप्टो)\b',
            ],
        },
    }
    
    # Trading pair patterns
    TRADING_PAIR_PATTERNS = {
        Language.ENGLISH: r'\b([A-Z]{2,10}/[A-Z]{2,10})\b',
        Language.URDU: r'\b([A-Z]{2,10}/[A-Z]{2,10})\b',
        Language.HINDI: r'\b([A-Z]{2,10}/[A-Z]{2,10})\b',
    }
    
    # Timeframe patterns
    TIMEFRAME_PATTERNS = {
        Language.ENGLISH: r'\b(\d+[mhdw])\b',
        Language.URDU: r'\b(\d+[mhdw])\b',
        Language.HINDI: r'\b(\d+[mhdw])\b',
    }
    
    def __init__(self, use_ml: bool = True):
        """
        Initialize the multilingual intent parser.
        
        Args:
            use_ml: Whether to use ML-based classification (requires scikit-learn)
        """
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        self.vectorizer = None
        self.intent_classifier = None
        
        if self.use_ml:
            self._initialize_ml_components()
        
        logger.info(f"MultilingualIntentParser initialized (ML: {self.use_ml})")
    
    def _initialize_ml_components(self):
        """Initialize ML-based classification components."""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            # Training data would be loaded from a dataset in production
            logger.info("ML components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")
            self.use_ml = False
    
    def detect_language(self, text: str) -> Language:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected Language enum value
        """
        text_lower = text.lower().strip()
        
        # Check for Urdu script (Arabic-based)
        if re.search(r'[\u0600-\u06FF]', text):
            return Language.URDU
        
        # Check for Hindi script (Devanagari)
        if re.search(r'[\u0900-\u097F]', text):
            return Language.HINDI
        
        # Check for Romanized Urdu/Hindi indicators
        romanized_indicators = [
            'hai', 'hain', 'ho', 'ka', 'ki', 'ke', 'ko', 'se', 'mein',
            'aur', 'ya', 'nahi', 'tha', 'the', 'thi', 'thay'
        ]
        words = text_lower.split()
        romanized_count = sum(1 for word in words if word in romanized_indicators)
        
        if romanized_count >= 2:
            return Language.ROMANIZED
        
        return Language.ENGLISH
    
    def parse_intent(self, message: str, language: Optional[Language] = None) -> Tuple[IntentType, float]:
        """
        Parse the intent from a message.
        
        Args:
            message: Input message to parse
            language: Optional language override
            
        Returns:
            Tuple of (IntentType, confidence_score)
        """
        if not message or not message.strip():
            return IntentType.UNKNOWN, 0.0
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(message)
        
        message_lower = message.lower().strip()
        intent_scores = defaultdict(float)
        
        # Check patterns for each intent type
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            if language in patterns:
                for pattern in patterns[language]:
                    matches = re.findall(pattern, message_lower, re.IGNORECASE)
                    if matches:
                        # Weight by number of matches and pattern specificity
                        intent_scores[intent_type] += len(matches) * 0.3
        
        # Apply ML-based classification if available
        if self.use_ml and self.vectorizer and self.intent_classifier:
            try:
                ml_intent, ml_confidence = self._ml_classify(message_lower)
                if ml_intent != IntentType.UNKNOWN:
                    intent_scores[ml_intent] += ml_confidence * 0.4
            except Exception as e:
                logger.debug(f"ML classification failed: {e}")
        
        # Apply heuristic adjustments
        self._apply_heuristic_adjustments(message_lower, intent_scores, language)
        
        # Determine best intent
        if not intent_scores:
            return IntentType.UNKNOWN, 0.0
        
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        # Normalize confidence
        confidence = min(best_score, 1.0)
        
        # Apply threshold
        if confidence < 0.2:
            return IntentType.UNKNOWN, confidence
        
        return best_intent, confidence
    
    def _ml_classify(self, text: str) -> Tuple[IntentType, float]:
        """
        ML-based intent classification.
        
        Args:
            text: Preprocessed text to classify
            
        Returns:
            Tuple of (IntentType, confidence)
        """
        # This would use a trained model in production
        # For now, return unknown with low confidence
        return IntentType.UNKNOWN, 0.0
    
    def _apply_heuristic_adjustments(
        self, 
        text: str, 
        scores: Dict[IntentType, float],
        language: Language
    ):
        """Apply heuristic adjustments to intent scores."""
        
        # Boost trade intent if trading pairs are mentioned
        if self.extract_trading_pairs(text, language):
            scores[IntentType.TRADE] += 0.2
        
        # Boost analysis intent if technical indicators are mentioned
        analysis_indicators = ['rsi', 'macd', 'ema', 'sma', 'bollinger', 'fibonacci']
        if any(indicator in text for indicator in analysis_indicators):
            scores[IntentType.ANALYSIS] += 0.3
        
        # Boost query intent for questions
        if text.endswith('?'):
            scores[IntentType.QUERY] += 0.2
        
        # Boost command intent for imperative sentences
        imperative_words = ['show', 'give', 'tell', 'do', 'make', 'set', 'create']
        if text.split()[0] in imperative_words:
            scores[IntentType.COMMAND] += 0.15
    
    def extract_trading_pairs(self, text: str, language: Optional[Language] = None) -> List[str]:
        """
        Extract trading pairs from text.
        
        Args:
            text: Input text
            language: Language of the text
            
        Returns:
            List of trading pair strings
        """
        if language is None:
            language = self.detect_language(text)
        
        pattern = self.TRADING_PAIR_PATTERNS.get(language, self.TRADING_PAIR_PATTERNS[Language.ENGLISH])
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        # Normalize pairs to uppercase
        return [pair.upper() for pair in matches]
    
    def extract_timeframes(self, text: str, language: Optional[Language] = None) -> List[str]:
        """
        Extract timeframes from text.
        
        Args:
            text: Input text
            language: Language of the text
            
        Returns:
            List of timeframe strings
        """
        if language is None:
            language = self.detect_language(text)
        
        pattern = self.TIMEFRAME_PATTERNS.get(language, self.TIMEFRAME_PATTERNS[Language.ENGLISH])
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        # Normalize timeframes
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        return [tf.lower() for tf in matches if tf.lower() in valid_timeframes]
    
    def extract_trade_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract trading parameters from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        text_lower = text.lower()
        
        # Extract action (buy/sell)
        if re.search(r'\b(buy|خرید|खरीदें)\b', text_lower):
            params['action'] = 'buy'
        elif re.search(r'\b(sell|فروخت|बेचें)\b', text_lower):
            params['action'] = 'sell'
        
        # Extract order type
        if re.search(r'\b(limit|محدود|लिमिट)\b', text_lower):
            params['order_type'] = 'limit'
        elif re.search(r'\b(market|مارکیٹ|मार्केट)\b', text_lower):
            params['order_type'] = 'market'
        elif re.search(r'\b(stop|سٹاپ|स्टॉप)\b', text_lower):
            params['order_type'] = 'stop'
        
        # Extract quantity
        quantity_match = re.search(r'(\d+\.?\d*)\s*(btc|eth|bnb|usdt|coins?|tokens?)', text_lower)
        if quantity_match:
            params['quantity'] = float(quantity_match.group(1))
            params['asset'] = quantity_match.group(2).upper()
        
        # Extract price
        price_match = re.search(r'(?:at|@|price)\s*(\d+\.?\d*)', text_lower)
        if price_match:
            params['price'] = float(price_match.group(1))
        
        return params


class ContextMemoryManager:
    """
    Manages conversation context and memory for the chat system.
    
    Features:
    - Short-term conversation context
    - Long-term user preference learning
    - Conversation history persistence
    - Context window management
    """
    
    def __init__(self, max_history: int = 1000, context_window_size: int = 10):
        """
        Initialize the context memory manager.
        
        Args:
            max_history: Maximum number of messages to keep in history
            context_window_size: Size of the context window for current conversation
        """
        self.max_history = max_history
        self.context_window_size = context_window_size
        
        # Thread-safe storage
        self._lock = threading.RLock()
        
        # Conversation storage
        self.conversations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        
        # User preferences
        self.user_preferences: Dict[str, UserPreference] = defaultdict(UserPreference)
        
        # Active contexts
        self.active_contexts: Dict[str, ConversationContext] = {}
        
        # Session management
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ContextMemoryManager initialized (max_history={max_history})")
    
    def add_message(
        self,
        user_id: str,
        message: ChatMessage,
        session_id: Optional[str] = None
    ) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            user_id: Unique user identifier
            message: ChatMessage object to add
            session_id: Optional session identifier
        """
        if not user_id or not message.content:
            logger.warning("Invalid message data provided")
            return
        
        with self._lock:
            # Generate session ID if not provided
            if not session_id:
                session_id = f"{user_id}_{int(time.time())}"
            
            # Set message metadata
            message.user_id = user_id
            message.session_id = session_id
            
            # Add to conversation history
            self.conversations[user_id].append(message)
            
            # Update user preferences
            self._update_user_preferences(user_id, message)
            
            # Update active context
            self._update_active_context(user_id, message)
            
            # Update session info
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'user_id': user_id,
                    'start_time': datetime.now(),
                    'message_count': 0,
                    'last_intent': None
                }
            self.sessions[session_id]['message_count'] += 1
            self.sessions[session_id]['last_intent'] = message.intent.value
    
    def _update_user_preferences(self, user_id: str, message: ChatMessage):
        """Update user preferences based on message analysis."""
        preferences = self.user_preferences[user_id]
        
        # Update language preference
        if message.language != Language.ENGLISH:
            preferences.preferred_language = message.language
        
        # Update trading pairs
        parser = MultilingualIntentParser()
        pairs = parser.extract_trading_pairs(message.content, message.language)
        for pair in pairs:
            if pair not in preferences.trading_pairs:
                preferences.trading_pairs.append(pair)
        
        # Update timeframes
        timeframes = parser.extract_timeframes(message.content, message.language)
        for tf in timeframes:
            if tf not in preferences.timeframes:
                preferences.timeframes.append(tf)
        
        # Update intent frequency
        preferences.common_intents[message.intent.value] += 1
        
        # Update interaction metrics
        preferences.last_interaction = datetime.now()
        preferences.interaction_count += 1
    
    def _update_active_context(self, user_id: str, message: ChatMessage):
        """Update the active conversation context."""
        if user_id not in self.active_contexts:
            self.active_contexts[user_id] = ConversationContext()
        
        context = self.active_contexts[user_id]
        
        # Update context window
        context.context_window.append(message)
        
        # Update current intent
        context.current_intent = message.intent
        
        # Extract and store mentioned pairs and timeframes
        parser = MultilingualIntentParser()
        context.mentioned_pairs.extend(
            parser.extract_trading_pairs(message.content, message.language)
        )
        context.mentioned_timeframes.extend(
            parser.extract_timeframes(message.content, message.language)
        )
        
        # Keep only last 10 unique mentions
        context.mentioned_pairs = list(set(context.mentioned_pairs))[:10]
        context.mentioned_timeframes = list(set(context.mentioned_timeframes))[:10]
    
    def get_conversation_history(
        self,
        user_id: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[ChatMessage]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: Unique user identifier
            limit: Maximum number of messages to return
            since: Only return messages after this timestamp
            
        Returns:
            List of ChatMessage objects
        """
        with self._lock:
            if user_id not in self.conversations:
                return []
            
            messages = list(self.conversations[user_id])
            
            # Filter by timestamp
            if since:
                messages = [m for m in messages if m.timestamp >= since]
            
            # Apply limit
            if limit:
                messages = messages[-limit:]
            
            return messages
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """
        Get preferences for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            UserPreference object or None
        """
        with self._lock:
            return self.user_preferences.get(user_id)
    
    def get_active_context(self, user_id: str) -> Optional[ConversationContext]:
        """
        Get active conversation context for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            ConversationContext object or None
        """
        with self._lock:
            return self.active_contexts.get(user_id)
    
    def clear_conversation(self, user_id: str) -> bool:
        """
        Clear conversation history for a user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if user_id in self.conversations:
                self.conversations[user_id].clear()
                if user_id in self.active_contexts:
                    del self.active_contexts[user_id]
                return True
            return False
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of a chat session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dictionary or None
        """
        with self._lock:
            return self.sessions.get(session_id)
    
    def export_conversations(self, user_id: str, filepath: str) -> bool:
        """
        Export conversation history to a JSON file.
        
        Args:
            user_id: Unique user identifier
            filepath: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                if user_id not in self.conversations:
                    return False
                
                messages = list(self.conversations[user_id])
                export_data = []
                
                for msg in messages:
                    export_data.append({
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'language': msg.language.value,
                        'intent': msg.intent.value,
                        'confidence': msg.confidence,
                        'metadata': msg.metadata
                    })
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Exported {len(export_data)} messages to {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export conversations: {e}")
            return False
    
    def import_conversations(self, user_id: str, filepath: str) -> bool:
        """
        Import conversation history from a JSON file.
        
        Args:
            user_id: Unique user identifier
            filepath: Path to import file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with self._lock:
                for item in import_data:
                    msg = ChatMessage(
                        content=item['content'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        language=Language(item['language']),
                        intent=IntentType(item['intent']),
                        confidence=item.get('confidence', 0.0),
                        user_id=user_id,
                        metadata=item.get('metadata', {})
                    )
                    self.conversations[user_id].append(msg)
                
                logger.info(f"Imported {len(import_data)} messages from {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to import conversations: {e}")
            return False


class UserPreferenceLearner:
    """
    Learns and adapts to user preferences over time.
    
    Features:
    - Pattern recognition in user behavior
    - Adaptive response generation
    - Preference prediction
    - Anomaly detection
    """
    
    def __init__(self, memory_manager: ContextMemoryManager):
        """
        Initialize the preference learner.
        
        Args:
            memory_manager: ContextMemoryManager instance
        """
        self.memory_manager = memory_manager
        self.learning_rate = 0.1
        self.forget_threshold = timedelta(days=30)
        
        logger.info("UserPreferenceLearner initialized")
    
    def learn_from_interaction(self, user_id: str, message: ChatMessage) -> None:
        """
        Learn from a user interaction.
        
        Args:
            user_id: Unique user identifier
            message: ChatMessage from the interaction
        """
        preferences = self.memory_manager.get_user_preferences(user_id)
        if not preferences:
            return
        
        # Update interaction patterns
        self._update_time_patterns(user_id, message)
        self._update_topic_preferences(user_id, message)
        self._update_communication_style(user_id, message)
    
    def _update_time_patterns(self, user_id: str, message: ChatMessage):
        """Update learned time patterns for user activity."""
        # This would track peak activity times, preferred trading hours, etc.
        pass
    
    def _update_topic_preferences(self, user_id: str, message: ChatMessage):
        """Update topic preferences based on message content."""
        preferences = self.memory_manager.get_user_preferences(user_id)
        if not preferences:
            return
        
        # Update intent frequency with exponential moving average
        intent_key = message.intent.value
        current_count = preferences.common_intents.get(intent_key, 0)
        preferences.common_intents[intent_key] = int(
            current_count * (1 - self.learning_rate) + 1 * self.learning_rate
        )
    
    def _update_communication_style(self, user_id: str, message: ChatMessage):
        """Update learned communication style preferences."""
        # This would track formality level, technical jargon usage, etc.
        pass
    
    def predict_preferred_response(
        self,
        user_id: str,
        message: ChatMessage
    ) -> Dict[str, Any]:
        """
        Predict the preferred response style for a user.
        
        Args:
            user_id: Unique user identifier
            message: Current message
            
        Returns:
            Dictionary with response preferences
        """
        preferences = self.memory_manager.get_user_preferences(user_id)
        
        prediction = {
            'language': preferences.preferred_language.value if preferences else 'en',
            'formality': self._predict_formality(user_id),
            'detail_level': self._predict_detail_level(user_id),
            'response_type': self._predict_response_type(user_id, message)
        }
        
        return prediction
    
    def _predict_formality(self, user_id: str) -> str:
        """Predict the formality level for responses."""
        # Analyze past messages for formality indicators
        history = self.memory_manager.get_conversation_history(user_id, limit=50)
        
        formal_indicators = 0
        informal_indicators = 0
        
        for msg in history:
            content = msg.content.lower()
            if any(word in content for word in ['please', 'thank', 'would', 'could']):
                formal_indicators += 1
            if any(word in content for word in ['yeah', 'nah', 'gonna', 'wanna']):
                informal_indicators += 1
        
        if formal_indicators > informal_indicators:
            return 'formal'
        elif informal_indicators > formal_indicators:
            return 'informal'
        return 'neutral'
    
    def _predict_detail_level(self, user_id: str) -> str:
        """Predict the preferred detail level for responses."""
        history = self.memory_manager.get_conversation_history(user_id, limit=50)
        
        # Analyze message length and complexity
        avg_length = sum(len(msg.content) for msg in history) / max(len(history), 1)
        
        if avg_length > 200:
            return 'detailed'
        elif avg_length < 50:
            return 'concise'
        return 'balanced'
    
    def _predict_response_type(
        self,
        user_id: str,
        message: ChatMessage
    ) -> str:
        """Predict the type of response needed."""
        context = self.memory_manager.get_active_context(user_id)
        
        if context and context.requires_clarification:
            return 'clarification'
        
        if message.intent == IntentType.TRADE:
            return 'execution'
        elif message.intent ==