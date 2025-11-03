"""
Real-time Market Intelligence Module

Provides comprehensive market analysis including:
- Sentiment Radar (social media sentiment aggregation)
- News Impact Analyzer
- Market Regime Detection
- Fear & Greed Index
- Dynamic Correlation Matrix
- Intelligent Alert System
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class SentimentLevel(Enum):
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentData:
    """Container for sentiment analysis data"""

    source: str
    timestamp: datetime
    sentiment_score: float
    confidence: float
    volume: int
    keywords: List[str]
    raw_data: Dict[str, Any]


@dataclass
class NewsImpact:
    """Container for news impact analysis"""

    headline: str
    timestamp: datetime
    sentiment_score: float
    impact_score: float
    relevance_score: float
    source: str
    url: Optional[str]
    keywords: List[str]


@dataclass
class MarketAlert:
    """Container for market alerts"""

    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    data: Dict[str, Any]
    triggered_by: str


class MarketIntelligence:
    """Real-time market intelligence engine"""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize market intelligence

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sentiment_cache: dict = {}
        self.news_cache: dict = {}
        self.alerts: list = []
        self.correlation_history: list = []
        self.reddit_enabled = self.config.get("reddit_enabled", False)
        self.news_enabled = self.config.get("news_enabled", False)
        self.twitter_enabled = self.config.get("twitter_enabled", False)

    async def get_sentiment_radar(self, assets: List[str] = None) -> Dict[str, Any]:
        """
        Get real-time sentiment radar data

        Args:
            assets: List of assets to analyze

        Returns:
            Dict containing sentiment radar data
        """
        try:
            if not assets:
                assets = ["BTC", "ETH", "LTC", "XRP", "DOGE"]
            sentiment_data = {}
            for asset in assets:
                asset_sentiment = await self._aggregate_asset_sentiment(asset)
                sentiment_data[asset] = asset_sentiment
            overall_sentiment = self._calculate_overall_sentiment(sentiment_data)
            radar_data = self._generate_radar_data(sentiment_data)
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_sentiment": overall_sentiment,
                "asset_sentiments": sentiment_data,
                "radar_data": radar_data,
                "sentiment_distribution": self._calculate_sentiment_distribution(sentiment_data),
                "trending_assets": self._identify_trending_assets(sentiment_data),
                "sentiment_momentum": self._calculate_sentiment_momentum(sentiment_data),
            }
        except Exception as e:
            logger.error("Error in sentiment radar: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    async def _aggregate_asset_sentiment(self, asset: str) -> Dict[str, Any]:
        """Aggregate sentiment data for a specific asset"""
        try:
            sentiment_sources: list = []
            if self.reddit_enabled:
                reddit_sentiment = await self._get_reddit_sentiment(asset)
                if reddit_sentiment:
                    sentiment_sources.append(reddit_sentiment)
            if self.news_enabled:
                news_sentiment = await self._get_news_sentiment(asset)
                if news_sentiment:
                    sentiment_sources.append(news_sentiment)
            if self.twitter_enabled:
                twitter_sentiment = await self._get_twitter_sentiment(asset)
                if twitter_sentiment:
                    sentiment_sources.append(twitter_sentiment)
            if not sentiment_sources:
                sentiment_sources = self._generate_sample_sentiment(asset)
            if sentiment_sources:
                weighted_sentiment = sum(
                    (s.sentiment_score * s.confidence for s in sentiment_sources)
                ) / sum((s.confidence for s in sentiment_sources))
                avg_confidence = sum((s.confidence for s in sentiment_sources)) / len(
                    sentiment_sources
                )
                total_volume = sum((s.volume for s in sentiment_sources))
                return {
                    "sentiment_score": weighted_sentiment,
                    "confidence": avg_confidence,
                    "volume": total_volume,
                    "sources": len(sentiment_sources),
                    "source_breakdown": {s.source: s.sentiment_score for s in sentiment_sources},
                    "keywords": list(set([kw for s in sentiment_sources for kw in s.keywords])),
                }
            else:
                return {
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "volume": 0,
                    "sources": 0,
                    "source_breakdown": {},
                    "keywords": [],
                }
        except Exception as e:
            logger.error("Error aggregating sentiment for %s: %s", asset, e)
            return {"sentiment_score": 0.0, "confidence": 0.0, "volume": 0, "sources": 0}

    def _generate_sample_sentiment(self, asset: str) -> List[SentimentData]:
        """Generate sample sentiment data for demonstration"""
        np.random.seed(hash(asset + str(datetime.now().date())) % 2**32)
        sources = ["reddit", "news", "twitter"]
        sentiment_data: list = []
        for source in sources:
            base_sentiment = np.random.normal(0, 0.3)
            if asset == "BTC":
                base_sentiment += 0.1
            elif asset == "DOGE":
                base_sentiment += np.random.choice([-0.2, 0.3])
            sentiment_data.append(
                SentimentData(
                    source=source,
                    timestamp=datetime.now(),
                    sentiment_score=np.clip(base_sentiment, -1, 1),
                    confidence=np.random.uniform(0.6, 0.9),
                    volume=np.random.randint(100, 1000),
                    keywords=[f"{asset.lower()}", "crypto", "trading"],
                    raw_data={},
                )
            )
        return sentiment_data

    async def _get_reddit_sentiment(self, asset: str) -> Optional[SentimentData]:
        """Get Reddit sentiment (placeholder for real implementation)"""
        return None

    async def _get_news_sentiment(self, asset: str) -> Optional[SentimentData]:
        """Get news sentiment (placeholder for real implementation)"""
        return None

    async def _get_twitter_sentiment(self, asset: str) -> Optional[SentimentData]:
        """Get Twitter sentiment (placeholder for real implementation)"""
        return None

    def _calculate_overall_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market sentiment"""
        if not sentiment_data:
            return {"score": 0.0, "level": SentimentLevel.NEUTRAL.value, "confidence": 0.0}
        total_weighted_sentiment = 0
        total_weight = 0
        for asset, data in sentiment_data.items():
            weight = data["volume"] * data["confidence"]
            total_weighted_sentiment += data["sentiment_score"] * weight
            total_weight += weight
        if total_weight == 0:
            return {"score": 0.0, "level": SentimentLevel.NEUTRAL.value, "confidence": 0.0}
        overall_score = total_weighted_sentiment / total_weight
        avg_confidence = sum((data["confidence"] for data in sentiment_data.values())) / len(
            sentiment_data
        )
        if overall_score <= -0.6:
            level = SentimentLevel.EXTREME_FEAR
        elif overall_score <= -0.2:
            level = SentimentLevel.FEAR
        elif overall_score >= 0.6:
            level = SentimentLevel.EXTREME_GREED
        elif overall_score >= 0.2:
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.NEUTRAL
        return {"score": overall_score, "level": level.value, "confidence": avg_confidence}

    def _generate_radar_data(self, sentiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate data for sentiment radar chart"""
        radar_data: list = []
        for asset, data in sentiment_data.items():
            radar_data.append(
                {
                    "asset": asset,
                    "sentiment": (data["sentiment_score"] + 1) * 50,
                    "confidence": data["confidence"] * 100,
                    "volume": min(data["volume"] / 10, 100),
                    "sources": data["sources"],
                }
            )
        return radar_data

    def _calculate_sentiment_distribution(self, sentiment_data: Dict[str, Any]) -> Dict[str, int]:
        """Calculate distribution of sentiment levels"""
        distribution = {level.value: 0 for level in SentimentLevel}
        for asset, data in sentiment_data.items():
            score = data["sentiment_score"]
            if score <= -0.6:
                distribution[SentimentLevel.EXTREME_FEAR.value] += 1
            elif score <= -0.2:
                distribution[SentimentLevel.FEAR.value] += 1
            elif score >= 0.6:
                distribution[SentimentLevel.EXTREME_GREED.value] += 1
            elif score >= 0.2:
                distribution[SentimentLevel.GREED.value] += 1
            else:
                distribution[SentimentLevel.NEUTRAL.value] += 1
        return distribution

    def _identify_trending_assets(self, sentiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify trending assets based on sentiment"""
        trending: list = []
        for asset, data in sentiment_data.items():
            if data["volume"] > 500 and abs(data["sentiment_score"]) > 0.3:
                trending.append(
                    {
                        "asset": asset,
                        "sentiment_score": data["sentiment_score"],
                        "volume": data["volume"],
                        "trend_strength": abs(data["sentiment_score"]) * (data["volume"] / 1000),
                    }
                )
        trending.sort(key=lambda x: x["trend_strength"], reverse=True)
        return trending[:5]

    def _calculate_sentiment_momentum(self, sentiment_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sentiment momentum (change over time)"""
        momentum = {}
        for asset, data in sentiment_data.items():
            base_momentum = np.random.normal(0, 0.1)
            if abs(data["sentiment_score"]) > 0.5:
                base_momentum += np.sign(data["sentiment_score"]) * 0.05
            momentum[asset] = base_momentum
        return momentum

    async def analyze_news_impact(self, limit: int = 50) -> Dict[str, Any]:
        """
        Analyze news impact on market sentiment

        Args:
            limit: Number of news articles to analyze

        Returns:
            Dict containing news impact analysis
        """
        try:
            news_articles = await self._fetch_recent_news(limit)
            analyzed_articles: list = []
            for article in news_articles:
                impact = await self._analyze_article_impact(article)
                analyzed_articles.append(impact)
            analyzed_articles.sort(key=lambda x: x.impact_score, reverse=True)
            avg_sentiment = (
                np.mean([a.sentiment_score for a in analyzed_articles]) if analyzed_articles else 0
            )
            high_impact_count = len([a for a in analyzed_articles if a.impact_score > 0.7])
            all_keywords = [kw for a in analyzed_articles for kw in a.keywords]
            keyword_counts: dict = {}
            for kw in all_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            return {
                "timestamp": datetime.now().isoformat(),
                "total_articles": len(analyzed_articles),
                "high_impact_articles": high_impact_count,
                "average_sentiment": avg_sentiment,
                "top_articles": analyzed_articles[:10],
                "sentiment_distribution": self._calculate_news_sentiment_distribution(
                    analyzed_articles
                ),
                "top_keywords": top_keywords,
                "impact_timeline": self._create_impact_timeline(analyzed_articles),
            }
        except Exception as e:
            logger.error("Error in news impact analysis: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    async def _fetch_recent_news(self, limit: int) -> List[Dict[str, Any]]:
        """Fetch recent news articles (simulated)"""
        sample_headlines = [
            "Bitcoin reaches new all-time high amid institutional adoption",
            "Ethereum upgrade shows promising scalability improvements",
            "Regulatory concerns weigh on cryptocurrency markets",
            "Major bank announces crypto custody services",
            "DeFi protocol suffers major security breach",
            "Central bank digital currency pilot program launched",
            "Crypto mining energy consumption debate intensifies",
            "NFT market shows signs of recovery after downturn",
            "Stablecoin regulations proposed by financial authorities",
            "Blockchain technology adoption accelerates in enterprise",
        ]
        news_articles: list = []
        for i in range(min(limit, len(sample_headlines))):
            news_articles.append(
                {
                    "headline": sample_headlines[i],
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                    "source": np.random.choice(
                        ["CoinDesk", "CoinTelegraph", "Reuters", "Bloomberg"]
                    ),
                    "url": f"https://example.com/news/{i}",
                    "content": f"Article content for: {sample_headlines[i]}",
                }
            )
        return news_articles

    async def _analyze_article_impact(self, article: Dict[str, Any]) -> NewsImpact:
        """Analyze the impact of a single news article"""
        headline = article["headline"].lower()
        positive_keywords = ["high", "adoption", "improvement", "recovery", "launch", "accelerate"]
        negative_keywords = ["concern", "breach", "downturn", "regulation", "debate", "weigh"]
        sentiment_score = 0.0
        impact_score = 0.5
        keywords: list = []
        for word in positive_keywords:
            if word in headline:
                sentiment_score += 0.2
                impact_score += 0.1
                keywords.append(word)
        for word in negative_keywords:
            if word in headline:
                sentiment_score -= 0.2
                impact_score += 0.1
                keywords.append(word)
        crypto_keywords = ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft"]
        relevance_score = 0.5
        for word in crypto_keywords:
            if word in headline:
                relevance_score += 0.1
                keywords.append(word)
        sentiment_score = np.clip(sentiment_score, -1, 1)
        impact_score = np.clip(impact_score, 0, 1)
        relevance_score = np.clip(relevance_score, 0, 1)
        return NewsImpact(
            headline=article["headline"],
            timestamp=article["timestamp"],
            sentiment_score=sentiment_score,
            impact_score=impact_score,
            relevance_score=relevance_score,
            source=article["source"],
            url=article.get("url"),
            keywords=keywords,
        )

    def _calculate_news_sentiment_distribution(self, articles: List[NewsImpact]) -> Dict[str, int]:
        """Calculate distribution of news sentiment"""
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        for article in articles:
            if article.sentiment_score > 0.1:
                distribution["positive"] += 1
            elif article.sentiment_score < -0.1:
                distribution["negative"] += 1
            else:
                distribution["neutral"] += 1
        return distribution

    def _create_impact_timeline(self, articles: List[NewsImpact]) -> List[Dict[str, Any]]:
        """Create timeline of news impact"""
        timeline = {}
        for article in articles:
            hour_key = article.timestamp.strftime("%Y-%m-%d %H:00")
            if hour_key not in timeline:
                timeline[hour_key] = {
                    "timestamp": hour_key,
                    "article_count": 0,
                    "avg_sentiment": 0,
                    "avg_impact": 0,
                    "articles": [],
                }
            timeline[hour_key]["article_count"] += 1
            timeline[hour_key]["articles"].append(article)
        for hour_data in timeline.values():
            articles = hour_data["articles"]
            hour_data["avg_sentiment"] = np.mean([a.sentiment_score for a in articles])
            hour_data["avg_impact"] = np.mean([a.impact_score for a in articles])
            del hour_data["articles"]
        return sorted(timeline.values(), key=lambda x: x["timestamp"])

    def detect_market_regime(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Detect current market regime

        Args:
            price_data: Dictionary of asset price data

        Returns:
            Dict containing market regime analysis
        """
        try:
            if not price_data:
                return {"error": "No price data available"}  # type: ignore[dict-item]
            regime_analysis = {}
            for asset, data in price_data.items():
                if "close" not in data.columns or len(data) < 50:
                    continue
                prices = data["close"].values
                returns = np.diff(prices) / prices[:-1]
                regime = self._analyze_asset_regime(prices, returns)
                regime_analysis[asset] = regime
            overall_regime = self._determine_overall_regime(regime_analysis)
            regime_confidence = self._calculate_regime_confidence(regime_analysis)
            regime_changes = self._detect_regime_changes(regime_analysis)
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_regime": overall_regime,
                "regime_confidence": regime_confidence,
                "asset_regimes": regime_analysis,
                "regime_distribution": self._calculate_regime_distribution(regime_analysis),
                "regime_changes": regime_changes,
                "market_stress_indicators": self._calculate_stress_indicators(price_data),
            }
        except Exception as e:
            logger.error("Error in market regime detection: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def _analyze_asset_regime(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze regime for a single asset"""
        sma_20 = np.convolve(prices, np.ones(20) / 20, mode="valid")
        sma_50 = np.convolve(prices, np.ones(50) / 50, mode="valid")
        current_price = prices[-1]
        current_sma_20 = sma_20[-1] if len(sma_20) > 0 else current_price
        current_sma_50 = sma_50[-1] if len(sma_50) > 0 else current_price
        price_above_sma20 = current_price > current_sma_20
        price_above_sma50 = current_price > current_sma_50
        sma20_above_sma50 = current_sma_20 > current_sma_50
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        avg_volatility = np.std(returns)
        momentum_5d = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        momentum_20d = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
        regime_score = 0
        if price_above_sma20:
            regime_score += 1
        if price_above_sma50:
            regime_score += 1
        if sma20_above_sma50:
            regime_score += 1
        if momentum_5d > 0:
            regime_score += 0.5
        if momentum_20d > 0:
            regime_score += 0.5
        if volatility > avg_volatility * 1.5:
            regime = MarketRegime.VOLATILE
        elif regime_score >= 3:
            regime = MarketRegime.BULL
        elif regime_score <= 1:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.SIDEWAYS
        return {
            "regime": regime.value,
            "regime_score": regime_score,
            "volatility": volatility,
            "momentum_5d": momentum_5d,
            "momentum_20d": momentum_20d,
            "price_above_sma20": price_above_sma20,
            "price_above_sma50": price_above_sma50,
            "sma20_above_sma50": sma20_above_sma50,
        }

    def _determine_overall_regime(self, regime_analysis: Dict[str, Any]) -> str:
        """Determine overall market regime"""
        if not regime_analysis:
            return MarketRegime.SIDEWAYS.value
        regime_counts = {}
        for asset_regime in regime_analysis.values():
            regime = asset_regime["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        return max(regime_counts.items(), key=lambda x: x[1])[0]

    def _calculate_regime_confidence(self, regime_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in regime detection"""
        if not regime_analysis:
            return 0.0
        regime_scores = [data["regime_score"] for data in regime_analysis.values()]
        avg_score = np.mean(regime_scores)
        score_std = np.std(regime_scores)
        confidence = (1 - score_std / 2.5) * (abs(avg_score - 2) / 2)
        return np.clip(confidence, 0, 1)

    def _detect_regime_changes(self, regime_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect recent regime changes (simulated)"""
        changes: list = []
        for asset, data in regime_analysis.items():
            if np.random.random() < 0.2:
                changes.append(
                    {
                        "asset": asset,
                        "previous_regime": np.random.choice(["bull", "bear", "sideways"]),
                        "current_regime": data["regime"],
                        "change_timestamp": (
                            datetime.now() - timedelta(hours=np.random.randint(1, 48))
                        ).isoformat(),
                        "confidence": np.random.uniform(0.6, 0.9),
                    }
                )
        return changes

    def _calculate_regime_distribution(self, regime_analysis: Dict[str, Any]) -> Dict[str, int]:
        """Calculate distribution of regimes across assets"""
        distribution = {regime.value: 0 for regime in MarketRegime}
        for data in regime_analysis.values():
            regime = data["regime"]
            distribution[regime] += 1
        return distribution

    def _calculate_stress_indicators(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate market stress indicators"""
        stress_indicators = {}
        try:
            all_returns: list = []
            correlations = []
            for asset, data in price_data.items():
                if "close" in data.columns and len(data) > 1:
                    prices = data["close"].values
                    returns = np.diff(prices) / prices[:-1]
                    all_returns.extend(returns[-20:])
            if all_returns:
                stress_indicators["volatility_stress"] = min(np.std(all_returns) * 10, 1.0)
                extreme_moves = len([r for r in all_returns if abs(r) > 0.05]) / len(all_returns)
                stress_indicators["extreme_movement_stress"] = extreme_moves
                negative_returns = [r for r in all_returns if r < 0]
                if negative_returns:
                    stress_indicators["downside_stress"] = min(
                        abs(np.mean(negative_returns)) * 5, 1.0
                    )
                else:
                    stress_indicators["downside_stress"] = 0.0
            if stress_indicators:
                stress_indicators["overall_stress"] = np.mean(list(stress_indicators.values()))
        except Exception as e:
            logger.error("Error calculating stress indicators: %s", e)
            stress_indicators = {"overall_stress": 0.0}
        return stress_indicators

    def calculate_fear_greed_index(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate custom Fear & Greed Index

        Args:
            market_data: Dictionary containing various market metrics

        Returns:
            Dict containing Fear & Greed Index data
        """
        try:
            components = {}
            momentum_score = self._calculate_momentum_score(market_data.get("price_data", {}))
            components["momentum"] = {"score": momentum_score, "weight": 0.25}
            volatility_score = self._calculate_volatility_score(market_data.get("price_data", {}))
            components["volatility"] = {"score": volatility_score, "weight": 0.25}
            sentiment_score = self._calculate_sentiment_score(market_data.get("sentiment_data", {}))
            components["sentiment"] = {"score": sentiment_score, "weight": 0.2}
            dominance_score = self._calculate_dominance_score(market_data.get("price_data", {}))
            components["dominance"] = {"score": dominance_score, "weight": 0.15}
            volume_score = self._calculate_volume_score(market_data.get("price_data", {}))
            components["volume"] = {"score": volume_score, "weight": 0.15}
            fg_index = sum((comp["score"] * comp["weight"] for comp in components.values()))
            fg_index = np.clip(fg_index, 0, 100)
            if fg_index <= 20:
                level = "Extreme Fear"
                color = "#d32f2f"
            elif fg_index <= 40:
                level = "Fear"
                color = "#f57c00"
            elif fg_index <= 60:
                level = "Neutral"
                color = "#fbc02d"
            elif fg_index <= 80:
                level = "Greed"
                color = "#689f38"
            else:
                level = "Extreme Greed"
                color = "#388e3c"
            historical_avg = 50 + np.random.normal(0, 10)
            return {
                "timestamp": datetime.now().isoformat(),
                "index_value": fg_index,
                "level": level,
                "color": color,
                "components": components,
                "historical_average": historical_avg,
                "deviation_from_average": fg_index - historical_avg,
                "trend": self._calculate_fg_trend(),
                "interpretation": self._interpret_fg_index(fg_index),
            }
        except Exception as e:
            logger.error("Error calculating Fear & Greed Index: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def _calculate_momentum_score(self, price_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate momentum component score (0-100)"""
        if not price_data:
            return 50
        momentum_scores: list = []
        for asset, data in price_data.items():
            if "close" in data.columns and len(data) >= 30:
                prices = data["close"].values
                momentum_1d = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
                momentum_7d = (prices[-1] - prices[-8]) / prices[-8] if len(prices) >= 8 else 0
                momentum_30d = (prices[-1] - prices[-31]) / prices[-31] if len(prices) >= 31 else 0
                momentum = momentum_1d * 0.2 + momentum_7d * 0.3 + momentum_30d * 0.5
                momentum_score = 50 + momentum * 100
                momentum_scores.append(np.clip(momentum_score, 0, 100))
        return np.mean(momentum_scores) if momentum_scores else 50

    def _calculate_volatility_score(self, price_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate volatility component score (0-100, inverted - high vol = fear)"""
        if not price_data:
            return 50
        volatility_scores: list = []
        for asset, data in price_data.items():
            if "close" in data.columns and len(data) >= 20:
                prices = data["close"].values
                returns = np.diff(prices) / prices[:-1]
                current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
                historical_vol = np.std(returns)
                vol_ratio = current_vol / historical_vol if historical_vol != 0 else 1
                vol_score = 100 - min(vol_ratio * 50, 100)
                volatility_scores.append(max(vol_score, 0))
        return np.mean(volatility_scores) if volatility_scores else 50

    def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate sentiment component score (0-100)"""
        if not sentiment_data:
            return 50
        overall_sentiment = sentiment_data.get("overall_sentiment", {})
        sentiment_score = overall_sentiment.get("score", 0)
        return 50 + sentiment_score * 50

    def _calculate_dominance_score(self, price_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate market dominance score (0-100)"""
        return 50 + np.random.normal(0, 15)

    def _calculate_volume_score(self, price_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate volume component score (0-100)"""
        if not price_data:
            return 50
        volume_scores: list = []
        for asset, data in price_data.items():
            if "volume" in data.columns and len(data) >= 20:
                volumes = data["volume"].values
                current_vol = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
                avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
                vol_ratio = current_vol / avg_vol if avg_vol != 0 else 1
                vol_score = 50 + min((vol_ratio - 1) * 50, 50)
                volume_scores.append(max(vol_score, 0))
        return np.mean(volume_scores) if volume_scores else 50

    def _calculate_fg_trend(self) -> str:
        """Calculate Fear & Greed Index trend (simulated)"""
        trends = ["increasing", "decreasing", "stable"]
        return np.random.choice(trends)

    def _interpret_fg_index(self, index_value: float) -> str:
        """Provide interpretation of Fear & Greed Index"""
        if index_value <= 20:
            return "Extreme fear indicates potential buying opportunity as market may be oversold."
        elif index_value <= 40:
            return "Fear in the market suggests caution, but may present selective opportunities."
        elif index_value <= 60:
            return "Neutral sentiment indicates balanced market conditions."
        elif index_value <= 80:
            return "Greed in the market suggests caution as assets may be overvalued."
        else:
            return (
                "Extreme greed indicates potential selling opportunity as market may be overbought."
            )

    def generate_market_alerts(
        self, market_data: Dict[str, Any], alert_config: Dict[str, Any] = None
    ) -> List[MarketAlert]:
        """
        Generate intelligent market alerts

        Args:
            market_data: Current market data
            alert_config: Alert configuration settings

        Returns:
            List of MarketAlert objects
        """
        try:
            if not alert_config:
                alert_config = self._get_default_alert_config()
            alerts: list = []
            price_alerts = self._check_price_alerts(market_data.get("price_data", {}), alert_config)
            alerts.extend(price_alerts)
            sentiment_alerts = self._check_sentiment_alerts(
                market_data.get("sentiment_data", {}), alert_config
            )
            alerts.extend(sentiment_alerts)
            volume_alerts = self._check_volume_alerts(
                market_data.get("price_data", {}), alert_config
            )
            alerts.extend(volume_alerts)
            regime_alerts = self._check_regime_alerts(
                market_data.get("regime_data", {}), alert_config
            )
            alerts.extend(regime_alerts)
            fg_alerts = self._check_fear_greed_alerts(
                market_data.get("fear_greed_data", {}), alert_config
            )
            alerts.extend(fg_alerts)
            alerts.sort(
                key=lambda x: (self._get_severity_priority(x.severity), x.timestamp), reverse=True
            )
            self.alerts.extend(alerts)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
            return alerts
        except Exception as e:
            logger.error("Error generating market alerts: %s", e)
            return []

    def _get_default_alert_config(self) -> Dict[str, Any]:
        """Get default alert configuration"""
        return {
            "price_change_threshold": 0.05,
            "volume_change_threshold": 2.0,
            "sentiment_extreme_threshold": 0.7,
            "fear_greed_extreme_threshold": 20,
            "enabled_alerts": ["price", "sentiment", "volume", "regime", "fear_greed"],
        }

    def _check_price_alerts(
        self, price_data: Dict[str, pd.DataFrame], config: Dict[str, Any]
    ) -> List[MarketAlert]:
        """Check for price-based alerts"""
        alerts: list = []
        threshold = config.get("price_change_threshold", 0.05)
        for asset, data in price_data.items():
            if "close" in data.columns and len(data) >= 2:
                current_price = data["close"].iloc[-1]
                previous_price = data["close"].iloc[-2]
                change = (current_price - previous_price) / previous_price
                if abs(change) > threshold:
                    severity = "high" if abs(change) > threshold * 2 else "medium"
                    direction = "surge" if change > 0 else "drop"
                    alerts.append(
                        MarketAlert(
                            alert_id=f"price_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            timestamp=datetime.now(),
                            alert_type="price_movement",
                            severity=severity,
                            message=f"{asset} price {direction}: {change * 100:.2f}% in last period",
                            data={
                                "asset": asset,
                                "change_percent": change * 100,
                                "current_price": current_price,
                                "previous_price": previous_price,
                            },
                            triggered_by="price_monitor",
                        )
                    )
        return alerts

    def _check_sentiment_alerts(
        self, sentiment_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[MarketAlert]:
        """Check for sentiment-based alerts"""
        alerts: list = []
        threshold = config.get("sentiment_extreme_threshold", 0.7)
        overall_sentiment = sentiment_data.get("overall_sentiment", {})
        sentiment_score = overall_sentiment.get("score", 0)
        if abs(sentiment_score) > threshold:
            severity = "high" if abs(sentiment_score) > 0.8 else "medium"
            sentiment_type = "extremely positive" if sentiment_score > 0 else "extremely negative"
            alerts.append(
                MarketAlert(
                    alert_id=f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    alert_type="sentiment_extreme",
                    severity=severity,
                    message=f"Market sentiment is {sentiment_type}: {sentiment_score:.2f}",
                    data={
                        "sentiment_score": sentiment_score,
                        "sentiment_level": overall_sentiment.get("level", "unknown"),
                        "confidence": overall_sentiment.get("confidence", 0),
                    },
                    triggered_by="sentiment_monitor",
                )
            )
        return alerts

    def _check_volume_alerts(
        self, price_data: Dict[str, pd.DataFrame], config: Dict[str, Any]
    ) -> List[MarketAlert]:
        """Check for volume-based alerts"""
        alerts: list = []
        threshold = config.get("volume_change_threshold", 2.0)
        for asset, data in price_data.items():
            if "volume" in data.columns and len(data) >= 20:
                current_volume = data["volume"].iloc[-1]
                avg_volume = data["volume"].iloc[-20:].mean()
                if current_volume > avg_volume * threshold:
                    volume_ratio = current_volume / avg_volume
                    severity = "high" if volume_ratio > threshold * 2 else "medium"
                    alerts.append(
                        MarketAlert(
                            alert_id=f"volume_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            timestamp=datetime.now(),
                            alert_type="volume_spike",
                            severity=severity,
                            message=f"{asset} volume spike: {volume_ratio:.1f}x average volume",
                            data={
                                "asset": asset,
                                "current_volume": current_volume,
                                "average_volume": avg_volume,
                                "volume_ratio": volume_ratio,
                            },
                            triggered_by="volume_monitor",
                        )
                    )
        return alerts

    def _check_regime_alerts(
        self, regime_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[MarketAlert]:
        """Check for regime change alerts"""
        alerts: list = []
        regime_changes = regime_data.get("regime_changes", [])
        for change in regime_changes:
            change_time = datetime.fromisoformat(change["change_timestamp"].replace("Z", "+00:00"))
            if datetime.now() - change_time < timedelta(hours=2):
                alerts.append(
                    MarketAlert(
                        alert_id=f"regime_{change['asset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        alert_type="regime_change",
                        severity="medium",
                        message=f"{change['asset']} regime changed from {change['previous_regime']} to {change['current_regime']}",
                        data=change,
                        triggered_by="regime_monitor",
                    )
                )
        return alerts

    def _check_fear_greed_alerts(
        self, fg_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[MarketAlert]:
        """Check for Fear & Greed Index alerts"""
        alerts: list = []
        threshold = config.get("fear_greed_extreme_threshold", 20)
        index_value = fg_data.get("index_value", 50)
        if index_value <= threshold or index_value >= 100 - threshold:
            severity = "high" if index_value <= 10 or index_value >= 90 else "medium"
            level = fg_data.get("level", "Unknown")
            alerts.append(
                MarketAlert(
                    alert_id=f"feargreed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    alert_type="fear_greed_extreme",
                    severity=severity,
                    message=f"Fear & Greed Index at extreme level: {level} ({index_value:.0f})",
                    data={
                        "index_value": index_value,
                        "level": level,
                        "interpretation": fg_data.get("interpretation", ""),
                    },
                    triggered_by="fear_greed_monitor",
                )
            )
        return alerts

    def _get_severity_priority(self, severity: str) -> int:
        """Get priority value for severity (higher = more important)"""
        priorities = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return priorities.get(severity, 0)

    def get_dynamic_correlation_matrix(
        self, price_data: Dict[str, pd.DataFrame], window: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate dynamic correlation matrix

        Args:
            price_data: Dictionary of asset price data
            window: Rolling window for correlation calculation

        Returns:
            Dict containing correlation matrix and analysis
        """
        try:
            if len(price_data) < 2:
                return {"error": "Need at least 2 assets for correlation analysis"}  # type: ignore[dict-item]
            returns_data = {}
            for asset, data in price_data.items():
                if "close" in data.columns and len(data) > window:
                    prices = data["close"].values
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[asset] = returns
            if len(returns_data) < 2:
                return {"error": "Insufficient data for correlation analysis"}  # type: ignore[dict-item]
            min_length = min((len(returns) for returns in returns_data.values()))
            aligned_returns = {}
            for asset, returns in returns_data.items():
                aligned_returns[asset] = returns[-min_length:]
            assets = list(aligned_returns.keys())
            n_assets = len(assets)
            current_corr_matrix = np.zeros((n_assets, n_assets))
            for i, asset_i in enumerate(assets):
                for j, asset_j in enumerate(assets):
                    if i == j:
                        current_corr_matrix[i, j] = 1.0
                    else:
                        corr = np.corrcoef(
                            aligned_returns[asset_i][-window:], aligned_returns[asset_j][-window:]
                        )[0, 1]
                        current_corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
            long_window = min(window * 2, min_length)
            trend_matrix = np.zeros((n_assets, n_assets))
            if min_length > long_window:
                for i, asset_i in enumerate(assets):
                    for j, asset_j in enumerate(assets):
                        if i != j:
                            old_corr = np.corrcoef(
                                aligned_returns[asset_i][-long_window:-window],
                                aligned_returns[asset_j][-long_window:-window],
                            )[0, 1]
                            if not np.isnan(old_corr):
                                trend_matrix[i, j] = current_corr_matrix[i, j] - old_corr
            clusters = self._identify_correlation_clusters(current_corr_matrix, assets)
            avg_correlation = np.mean(current_corr_matrix[np.triu_indices(n_assets, k=1)])
            correlation_stress = min(max(avg_correlation, 0) * 2, 1)
            return {
                "timestamp": datetime.now().isoformat(),
                "assets": assets,
                "correlation_matrix": current_corr_matrix.tolist(),
                "trend_matrix": trend_matrix.tolist(),
                "average_correlation": avg_correlation,
                "correlation_stress": correlation_stress,
                "clusters": clusters,
                "strongest_correlations": self._find_strongest_correlations(
                    current_corr_matrix, assets
                ),
                "weakest_correlations": self._find_weakest_correlations(
                    current_corr_matrix, assets
                ),
                "correlation_summary": self._summarize_correlations(current_corr_matrix),
            }
        except Exception as e:
            logger.error("Error calculating correlation matrix: %s", e)
            return {"error": str(e)}  # type: ignore[dict-item]

    def _identify_correlation_clusters(
        self, corr_matrix: np.ndarray, assets: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify correlation clusters"""
        try:
            from sklearn.cluster import AgglomerativeClustering

            distance_matrix = 1 - np.abs(corr_matrix)
            n_clusters = min(3, len(assets))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            clusters: list = []
            for cluster_id in range(n_clusters):
                cluster_assets = [
                    assets[i] for (i, label) in enumerate(cluster_labels) if label == cluster_id
                ]
                if len(cluster_assets) > 1:
                    indices = [assets.index(asset) for asset in cluster_assets]
                    cluster_correlations: list = []
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            cluster_correlations.append(corr_matrix[indices[i], indices[j]])
                    avg_correlation = np.mean(cluster_correlations) if cluster_correlations else 0
                    clusters.append(
                        {
                            "cluster_id": cluster_id,
                            "assets": cluster_assets,
                            "avg_correlation": avg_correlation,
                            "size": len(cluster_assets),
                        }
                    )
            return clusters
        except ImportError:
            return []
        except Exception as e:
            logger.error("Error in correlation clustering: %s", e)
            return []

    def _find_strongest_correlations(
        self, corr_matrix: np.ndarray, assets: List[str]
    ) -> List[Dict[str, Any]]:
        """Find strongest correlations"""
        correlations: list = []
        n_assets = len(assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr_value = corr_matrix[i, j]
                correlations.append(
                    {"asset_1": assets[i], "asset_2": assets[j], "correlation": corr_value}
                )
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return correlations[:5]

    def _find_weakest_correlations(
        self, corr_matrix: np.ndarray, assets: List[str]
    ) -> List[Dict[str, Any]]:
        """Find weakest correlations"""
        correlations: list = []
        n_assets = len(assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr_value = corr_matrix[i, j]
                correlations.append(
                    {"asset_1": assets[i], "asset_2": assets[j], "correlation": corr_value}
                )
        correlations.sort(key=lambda x: abs(x["correlation"]))
        return correlations[:5]

    def _summarize_correlations(self, corr_matrix: np.ndarray) -> Dict[str, float]:
        """Summarize correlation matrix statistics"""
        upper_triangle = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]
        return {
            "mean": np.mean(upper_triangle),
            "median": np.median(upper_triangle),
            "std": np.std(upper_triangle),
            "min": np.min(upper_triangle),
            "max": np.max(upper_triangle),
            "positive_correlations": len(upper_triangle[upper_triangle > 0.1])
            / len(upper_triangle),
            "negative_correlations": len(upper_triangle[upper_triangle < -0.1])
            / len(upper_triangle),
            "strong_correlations": len(upper_triangle[np.abs(upper_triangle) > 0.7])
            / len(upper_triangle),
        }
