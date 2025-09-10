"""
Mock Web Searcher Tool for Educational Demonstrations

This module provides a drop-in replacement for the WebSearcherTool that doesn't
require any API keys. It returns realistic, dynamically generated mock data for
common search queries.

"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fairlib.core.interfaces.tools import AbstractTool

class MockWebSearcherTool(AbstractTool):
    """
    A mock web searcher tool that simulates real web search results.
    
    This tool provides realistic responses for common query types without
    requiring any API keys or external services. It's designed to be a
    drop-in replacement for WebSearcherTool in educational and testing contexts.
    
    Features:
        - Dynamic data generation (prices change between runs)
        - Temporal awareness (dates, "yesterday", "last week")
        - Multiple domain support (crypto, stocks, weather, news, general)
        - Configurable response delay to simulate network latency
        - Extensible mock data system
    """
    
    name = "web_search"
    description = "Search the web for current information (MOCK MODE)"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MockWebSearcherTool.
        
        Args:
            config: Optional configuration dict (maintained for compatibility)
                   Accepts all WebSearcherTool config params but ignores most
        """
        self.config = config or {}
        self.response_delay = self.config.get('mock_delay', 0.5)  # Simulate network delay
        
        # Initialize dynamic data
        self._initialize_mock_data()
        
    def _initialize_mock_data(self):
        """Initialize realistic mock data with some randomization."""
        
        # Base prices with realistic ranges
        self.base_data = {
            "crypto": {
                "bitcoin": {
                    "symbol": "BTC",
                    "base_price": 45000,
                    "volatility": 5000,
                    "market_cap": "880B",
                },
                "ethereum": {
                    "symbol": "ETH", 
                    "base_price": 2500,
                    "volatility": 400,
                    "market_cap": "300B",
                },
                "solana": {
                    "symbol": "SOL",
                    "base_price": 110,
                    "volatility": 20,
                    "market_cap": "47B",
                },
                "cardano": {
                    "symbol": "ADA",
                    "base_price": 0.65,
                    "volatility": 0.15,
                    "market_cap": "23B",
                }
            },
            "stocks": {
                "AAPL": {"name": "Apple Inc.", "base_price": 180, "volatility": 10},
                "GOOGL": {"name": "Alphabet Inc.", "base_price": 145, "volatility": 8},
                "MSFT": {"name": "Microsoft Corp.", "base_price": 380, "volatility": 15},
                "TSLA": {"name": "Tesla Inc.", "base_price": 250, "volatility": 30},
                "AMZN": {"name": "Amazon.com Inc.", "base_price": 170, "volatility": 12},
                "NVDA": {"name": "NVIDIA Corp.", "base_price": 500, "volatility": 40},
                "META": {"name": "Meta Platforms Inc.", "base_price": 350, "volatility": 25},
            },
            "weather_conditions": [
                "Sunny", "Partly Cloudy", "Cloudy", "Clear", "Overcast",
                "Light Rain", "Scattered Showers"
            ],
            "news_templates": [
                "Tech giants report strong quarterly earnings amid AI boom",
                "Federal Reserve signals potential rate changes in upcoming meeting",
                "Breakthrough in renewable energy storage technology announced",
                "Global markets react to latest economic indicators",
                "Major cybersecurity update recommended for all users",
                "Climate summit reaches unprecedented agreement on emissions",
                "Revolutionary medical treatment shows promising trial results",
            ]
        }
        
        # Generate session-specific variations
        self._generate_session_prices()
    
    def _generate_session_prices(self):
        """Generate prices for this session with random variations."""
        self.current_prices = {}
        
        # Generate crypto prices
        self.current_prices['crypto'] = {}
        for crypto, data in self.base_data['crypto'].items():
            variation = random.uniform(-data['volatility'], data['volatility'])
            price = max(data['base_price'] + variation, data['base_price'] * 0.5)
            change_24h = random.uniform(-7, 7)
            self.current_prices['crypto'][crypto] = {
                'price': price,
                'change_24h': change_24h,
                'symbol': data['symbol'],
                'market_cap': data['market_cap']
            }
        
        # Generate stock prices
        self.current_prices['stocks'] = {}
        for symbol, data in self.base_data['stocks'].items():
            variation = random.uniform(-data['volatility'], data['volatility'])
            price = max(data['base_price'] + variation, data['base_price'] * 0.8)
            change_pct = random.uniform(-3, 3)
            self.current_prices['stocks'][symbol] = {
                'price': price,
                'change_pct': change_pct,
                'name': data['name']
            }
    
    def use(self, tool_input: str) -> str:
        """
        Performs a mock web search with the given query.
        
        Args:
            tool_input: The search query string
            
        Returns:
            JSON string with search results matching WebSearcherTool format
        """
        # Simulate network delay
        if self.response_delay > 0:
            time.sleep(self.response_delay)
        
        query_lower = tool_input.lower()
        
        # Get appropriate mock data based on query type
        if self._is_crypto_query(query_lower):
            mock_data = self._handle_crypto_query(query_lower)
        elif self._is_stock_query(query_lower):
            mock_data = self._handle_stock_query(query_lower)
        elif self._is_weather_query(query_lower):
            mock_data = self._handle_weather_query(query_lower)
        elif self._is_news_query(query_lower):
            mock_data = self._handle_news_query(query_lower)
        elif self._is_date_time_query(query_lower):
            mock_data = self._handle_date_time_query(query_lower)
        else:
            mock_data = self._handle_general_query(tool_input)
        
        # Format results to match WebSearcherTool output
        results = []
        
        # Create primary search result
        results.append({
            "title": f"Search Results: {tool_input[:50]}",
            "url": f"https://example.com/search?q={tool_input.replace(' ', '+')}",
            "display_url": "example.com",
            "snippet": mock_data[:200] if len(mock_data) > 200 else mock_data,
            "source": "Mock Search",
            "relevance_score": 100
        })
        
        # Add secondary result for realism
        if len(mock_data) > 200:
            results.append({
                "title": f"Additional Information - {tool_input[:40]}",
                "url": f"https://info.example.org/?q={tool_input.replace(' ', '+')}",
                "display_url": "info.example.org",
                "snippet": mock_data[200:400] if len(mock_data) > 400 else mock_data[200:],
                "source": "Mock Database",
                "relevance_score": 85
            })
        
        # Return JSON string matching WebSearcherTool format
        response = {
            "query": tool_input,
            "enhanced_query": tool_input,
            "search_type": "general",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
            
        return json.dumps(response, indent=2)
    
    def _is_crypto_query(self, query: str) -> bool:
        """Check if query is about cryptocurrency."""
        crypto_terms = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
                       'solana', 'sol', 'cardano', 'ada', 'blockchain', 'defi']
        return any(term in query for term in crypto_terms)
    
    def _is_stock_query(self, query: str) -> bool:
        """Check if query is about stocks."""
        stock_terms = ['stock', 'share', 'nasdaq', 'dow jones', 's&p', 'market']
        stock_symbols = [s.lower() for s in self.base_data['stocks'].keys()]
        company_names = ['apple', 'google', 'alphabet', 'microsoft', 'tesla', 
                        'amazon', 'nvidia', 'meta', 'facebook']
        return any(term in query for term in stock_terms + stock_symbols + company_names)
    
    def _is_weather_query(self, query: str) -> bool:
        """Check if query is about weather."""
        weather_terms = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 
                        'cloudy', 'humidity', 'wind']
        return any(term in query for term in weather_terms)
    
    def _is_news_query(self, query: str) -> bool:
        """Check if query is about news."""
        news_terms = ['news', 'headline', 'latest', 'breaking', 'current events',
                     'today', 'recent']
        return any(term in query for term in news_terms)
    
    def _is_date_time_query(self, query: str) -> bool:
        """Check if query is about date/time."""
        date_terms = ['date', 'time', 'day', 'month', 'year', 'today', 'tomorrow',
                     'yesterday', 'week', 'weekend']
        return any(term in query for term in date_terms)
    
    def _handle_crypto_query(self, query: str) -> str:
        """Handle cryptocurrency-related queries."""
        results = []
        
        # Check which cryptos are mentioned
        for crypto, data in self.current_prices['crypto'].items():
            if crypto in query or data['symbol'].lower() in query:
                price_str = f"${data['price']:,.2f}" if data['price'] > 1 else f"${data['price']:.4f}"
                change_indicator = "📈" if data['change_24h'] > 0 else "📉"
                
                results.append(
                    f"{data['symbol']} Price: {price_str} USD\n"
                    f"24h Change: {change_indicator} {data['change_24h']:+.2f}%\n"
                    f"Market Cap: ${data['market_cap']}"
                )
        
        # Default to Bitcoin if no specific crypto mentioned
        if not results:
            btc_data = self.current_prices['crypto']['bitcoin']
            price_str = f"${btc_data['price']:,.2f}"
            change_indicator = "📈" if btc_data['change_24h'] > 0 else "📉"
            results.append(
                f"Bitcoin (BTC) Price: {price_str} USD\n"
                f"24h Change: {change_indicator} {btc_data['change_24h']:+.2f}%\n"
                f"Market Cap: ${btc_data['market_cap']}\n"
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        data = "\n\n".join(results)
        return data
    
    def _handle_stock_query(self, query: str) -> str:
        """Handle stock market queries."""
        results = []
        
        for symbol, data in self.current_prices['stocks'].items():
            if symbol.lower() in query or data['name'].lower() in query.lower():
                change_indicator = "🟢" if data['change_pct'] > 0 else "🔴"
                results.append(
                    f"{symbol} ({data['name']})\n"
                    f"Price: ${data['price']:.2f} USD\n"
                    f"Change: {change_indicator} {data['change_pct']:+.2f}%"
                )
        
        if not results:
            # Show market overview
            results.append("📊 Market Overview:\n")
            for symbol, data in list(self.current_prices['stocks'].items())[:3]:
                results.append(f"• {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)")
        
        data = "\n\n".join(results)
        return data
    
    def _handle_weather_query(self, query: str) -> str:
        """Handle weather-related queries."""
        temp_f = random.randint(45, 85)
        temp_c = round((temp_f - 32) * 5/9)
        condition = random.choice(self.base_data['weather_conditions'])
        humidity = random.randint(30, 80)
        wind_speed = random.randint(5, 20)
        
        # Extract location if mentioned
        location = "your location"
        common_cities = ['new york', 'london', 'tokyo', 'paris', 'sydney', 'los angeles']
        for city in common_cities:
            if city in query:
                location = city.title()
                break
        
        weather_data = (
            f"🌡️ Weather for {location}:\n"
            f"Temperature: {temp_f}°F ({temp_c}°C)\n"
            f"Condition: {condition}\n"
            f"Humidity: {humidity}%\n"
            f"Wind: {wind_speed} mph\n"
            f"Updated: {datetime.now().strftime('%I:%M %p')}"
        )
        
        return weather_data
    
    def _handle_news_query(self, query: str) -> str:
        """Handle news-related queries."""
        num_headlines = random.randint(3, 5)
        headlines = random.sample(self.base_data['news_templates'], num_headlines)
        
        news_data = f"📰 Latest News ({datetime.now().strftime('%Y-%m-%d')}):\n\n"
        for i, headline in enumerate(headlines, 1):
            time_ago = random.randint(1, 12)
            news_data += f"{i}. {headline}\n   ({time_ago} hours ago)\n\n"
        
        return news_data.strip()
    
    def _handle_date_time_query(self, query: str) -> str:
        """Handle date/time queries."""
        now = datetime.now()
        
        if "yesterday" in query:
            date = now - timedelta(days=1)
            result = f"Yesterday was {date.strftime('%A, %B %d, %Y')}"
        elif "tomorrow" in query:
            date = now + timedelta(days=1)
            result = f"Tomorrow will be {date.strftime('%A, %B %d, %Y')}"
        elif "time" in query:
            result = f"Current time: {now.strftime('%I:%M:%S %p %Z')}"
        else:
            result = f"Today is {now.strftime('%A, %B %d, %Y')}\nCurrent time: {now.strftime('%I:%M:%S %p')}"
        
        return result
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries that don't fit other categories."""
        results = [
            f"🔍 Search Results for: '{query}'\n",
            f"Found several relevant sources:\n",
            f"• Wikipedia: General information about {query}",
            f"• Recent articles: Multiple perspectives on {query}",
            f"• Expert opinions: Various viewpoints regarding {query}",
            f"\n💡 Tip: Try searching for more specific terms for detailed results."
        ]
        
        data = "\n".join(results)
        return data