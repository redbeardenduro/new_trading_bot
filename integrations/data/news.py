"""
Integration module for fetching news articles from NewsAPI and performing
sentiment analysis using shared utilities (SentimentAnalyzer with FinBERT model
preference, SentimentTracker), driven by central config. Implements the
ISentimentSource interface. Includes detailed logging of analyzed articles
and graceful handling of API rate limits.
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

try:
    from core.config import BotConfig
    BotConfigType: Any = BotConfig
except ImportError:
    import logging
    logging.error('CRITICAL: Failed to import BotConfig. NewsAPI may not function.')
    BotConfigType = None
try:
    from common.common_logger import CACHE_DIR, get_logger
    logger = get_logger('news_api')
except Exception as e:
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger('news_api_fallback')
    logger.error('Failed to get central logger: %s. Using basic fallback logger.', e)
    CACHE_DIR = Path('.') / 'data' / 'cache'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    from core.interfaces import ISentimentSource
    ISentimentSourceType: Any = ISentimentSource
except ImportError:
    logger.error('Failed to import ISentimentSource interface.')
    ISentimentSourceType = object
sentiment_utils_available = False
try:
    from utils.sentiment_utils import (SentimentAnalyzer, SentimentTracker,
                                       adjust_sentiment_by_volume,
                                       calculate_time_weighted_sentiment,
                                       get_sentiment_statistics,
                                       normalize_sentiment_score)
    sentiment_utils_available = True
except ImportError:
    logger.error('CRITICAL: Failed to import sentiment utilities from utils.sentiment_utils. News analysis will be impaired.')
    SentimentTracker = type('SentimentTracker', (), {})  # type: ignore[assignment]
    SentimentAnalyzer = type('SentimentAnalyzer', (), {})  # type: ignore[assignment]

    def calculate_time_weighted_sentiment(items: Any, **kwargs: Any) -> float:  # type: ignore[misc]
        return 0.0

    def adjust_sentiment_by_volume(s: Any, i: Any, b: Any) -> Any:  # type: ignore[misc]
        return s

    def get_sentiment_statistics(items: Any) -> dict:  # type: ignore[misc]
        return {}

    def normalize_sentiment_score(s: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return s

class NewsAPIRateLimitError(Exception):
    """Custom exception raised when NewsAPI rate limit is hit."""
    pass

def log_news_sentiment(symbol: str, articles_to_log: List[Dict]) -> None:
    """
    Logs analyzed news articles to a JSON file, appending only new entries
    and deduplicating based on URL.

    Args:
        symbol (str): The crypto symbol (e.g., 'BTC').
        articles_to_log (List[Dict]): A list of dictionaries, each containing
                                      analyzed article data. Expected keys:
                                      'headline', 'source_name', 'published_at',
                                      'url', 'sentiment_label', 'confidence_score', 'model'.
    """
    if not symbol or not isinstance(symbol, str):
        logger.error('Cannot log news sentiment: Invalid symbol provided.')
        return
    if not articles_to_log or not isinstance(articles_to_log, list):
        logger.debug('No new articles provided to log for %s.', symbol)
        return
    log_file = CACHE_DIR / f'{symbol.upper()}_News_sentiment.json'
    logger.debug('Logging %s analyzed news articles for %s to %s', len(articles_to_log), symbol, log_file)
    existing_logs: list = []
    existing_urls = set()
    if log_file.exists():
        try:
            with log_file.open('r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    loaded_data = json.loads(content)
                    if isinstance(loaded_data, list):
                        existing_logs = loaded_data
                        for entry in existing_logs:
                            if isinstance(entry, dict) and 'url' in entry:
                                existing_urls.add(entry['url'])
                    else:
                        logger.warning('Existing log file %s is not a list. Overwriting.', log_file)
        except json.JSONDecodeError:
            logger.error('Error decoding JSON from %s. Log file might be corrupted. Overwriting.', log_file)
            existing_logs = []  # type: ignore[assignment]
            existing_urls = set()
        except OSError as e:
            logger.error('OS error reading log file %s: %s. Will attempt to overwrite.', log_file, e)
            existing_logs = []  # type: ignore[assignment]
            existing_urls = set()
        except Exception as e:
            logger.error('Unexpected error loading log file %s: %s. Will attempt to overwrite.', log_file, e, exc_info=True)
            existing_logs = []  # type: ignore[assignment]
            existing_urls = set()
    new_entries_to_add: list = []
    log_timestamp = datetime.now(timezone.utc).isoformat()
    for article in articles_to_log:
        if not isinstance(article, dict):
            continue
        url = article.get('url')
        if url and url not in existing_urls:
            log_entry = {'timestamp': log_timestamp, 'symbol': symbol.upper(), 'headline': article.get('headline'), 'source_name': article.get('source_name'), 'published_at': article.get('published_at'), 'url': url, 'sentiment_label': article.get('sentiment_label'), 'confidence_score': article.get('confidence_score'), 'model': article.get('model')}
            if all((log_entry.get(k) is not None for k in ['headline', 'url', 'sentiment_label', 'model'])):
                new_entries_to_add.append(log_entry)
                existing_urls.add(url)
            else:
                logger.warning('Skipping article log due to missing essential fields: %s', url)
    if not new_entries_to_add:
        logger.info('No new unique articles to log for %s.', symbol)
        return
    all_logs = existing_logs + new_entries_to_add
    temp_file = log_file.with_suffix(f'.tmp_{time.time_ns()}')
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with temp_file.open('w', encoding='utf-8') as f:
            json.dump(all_logs, f, indent=2)
        temp_file.replace(log_file)
        logger.info('Successfully appended %s new news sentiment entries for %s to %s', len(new_entries_to_add), symbol, log_file.name)
    except (OSError, TypeError, ValueError) as e:
        logger.error('Error saving news sentiment log to %s: %s', log_file, e, exc_info=True)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.error('Unexpected error saving news sentiment log: %s', e, exc_info=True)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass

class NewsAPI(ISentimentSourceType):
    """Client for fetching news articles from NewsAPI using centralized config."""

    def __init__(self, config: Any) -> None:
        """Initializes the NewsAPI client using BotConfig."""
        if BotConfigType is None:
            raise ValueError('BotConfig is not available for NewsAPI initialization.')
        if not sentiment_utils_available:
            raise ImportError('Sentiment utilities could not be imported. NewsAPI cannot function.')
        self.config = config
        self.client_config = config.news_client
        self.sentiment_config = config.sentiment
        self.api_key: Optional[str] = config.get('api_credentials.news.api_key')
        self.base_url: str = 'https://newsapi.org/v2'
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: int = getattr(self.client_config, 'cache_expiry_seconds', 1800)
        self.debug_mode = getattr(self.client_config, 'debug_mode', False)
        self.rate_limited = False
        self.rate_limit_until: Optional[datetime] = None
        self.rate_limit_reset_hours = 12
        self.sentiment_analyzer = SentimentAnalyzer(model_type='finbert')
        tracker_hist_len = getattr(self.sentiment_config, 'history_length', 30)
        self.sentiment_tracker = SentimentTracker(history_length=tracker_hist_len)
        if not self.api_key:
            logger.error('NewsAPI key not found or empty in config. News fetching disabled.')
        else:
            logger.info('NewsAPI client initialized. Debug Mode: %s', self.debug_mode)

    def _check_rate_limit(self) -> None:
        """Checks if the rate limit is currently active and clears it if the estimated time has passed."""
        if self.rate_limited and self.rate_limit_until:
            if datetime.now(timezone.utc) > self.rate_limit_until:
                logger.info('Estimated NewsAPI rate limit period passed. Clearing rate limit flag.')
                self.rate_limited = False
                self.rate_limit_until = None
            else:
                wait_seconds = (self.rate_limit_until - datetime.now(timezone.utc)).total_seconds()
                logger.warning('NewsAPI is rate limited. Try again in %.1f hours.', wait_seconds / 3600)

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Tuple[Optional[Dict], Optional[Dict]]]:
        """
        Internal helper to make requests to NewsAPI with enhanced error handling.
        Returns a tuple: (json_response, headers) or (None, None) on failure.
        Raises NewsAPIRateLimitError on 429 status code.
        """
        if not self.api_key:
            logger.error('Cannot make NewsAPI request: API key missing.')
            return (None, None)
        if not isinstance(params, dict):
            logger.error('Cannot make NewsAPI request: Invalid params type (%s).', type(params))
            return (None, None)
        url = f'{self.base_url}/{endpoint}'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        timeout = getattr(self.client_config, 'request_timeout_seconds', 15)
        full_url_for_log = f'{url}?{urlencode(params)}'
        if self.debug_mode:
            logger.debug('NewsAPI Request: URL=%s', url)
            logger.debug('NewsAPI Request: Params=%s', params)
            logger.debug('NewsAPI Request: Full URL (for manual check) = %s', full_url_for_log)
            logger.debug('NewsAPI Request: Timeout=%ss', timeout)
        else:
            logger.debug('Making NewsAPI request to %s with query: %s', endpoint, params.get('q', 'N/A'))
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response_headers = dict(response.headers)
            if self.debug_mode:
                rate_limit = response_headers.get('X-RateLimit-Limit')
                rate_remaining = response_headers.get('X-RateLimit-Remaining')
                rate_reset = response_headers.get('X-RateLimit-Reset')
                logger.debug('NewsAPI Response: Status Code=%s', response.status_code)
                logger.debug('NewsAPI Response Headers: RateLimit=%s, Remaining=%s, Reset=%s', rate_limit, rate_remaining, rate_reset)
            response.raise_for_status()
            try:
                json_response = response.json()
                if not isinstance(json_response, dict):
                    logger.error('NewsAPI error: Invalid JSON response format (not a dict) for %s', url)
                    return (None, response_headers)
                if self.rate_limited:
                    logger.info('Successful NewsAPI request received. Clearing rate limit flag.')
                    self.rate_limited = False
                    self.rate_limit_until = None
                return (json_response, response_headers)
            except json.JSONDecodeError as json_err:
                logger.error("NewsAPI error: Failed to decode JSON response from %s. Content: '%s...' Error: %s", url, response.text[:200], json_err)
                return (None, response_headers)
        except requests.exceptions.Timeout:
            logger.error('NewsAPI request timed out (%ss) for URL: %s', timeout, url)
            return (None, None)
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            error_content = http_err.response.text[:500]
            error_detail = ''
            try:
                error_json = http_err.response.json()
                if isinstance(error_json, dict):
                    error_detail = f" Code: {error_json.get('code', 'N/A')}, Message: {error_json.get('message', 'N/A')}"
            except json.JSONDecodeError:
                pass
            if status_code == 429:
                logger.error('NewsAPI RATE LIMIT HIT (429) for %s.%s Message: %s', url, error_detail, error_content)
                self.rate_limited = True
                self.rate_limit_until = datetime.now(timezone.utc) + timedelta(hours=self.rate_limit_reset_hours)
                logger.warning('Setting rate limit flag. Estimated reset: %s', self.rate_limit_until.isoformat())
                raise NewsAPIRateLimitError(f'Rate limit hit: {error_detail}')
            else:
                logger.error("NewsAPI HTTP error: %s for %s.%s Response Hint: '%s...'", status_code, url, error_detail, error_content)
                return (None, response_headers)
        except requests.exceptions.ConnectionError as conn_err:
            logger.error('NewsAPI connection error for %s: %s', url, conn_err)
            return (None, None)
        except requests.exceptions.RequestException as req_err:
            logger.error('NewsAPI request failed for %s: %s', url, req_err, exc_info=True)
            return (None, None)
        except Exception as e:
            logger.error('NewsAPI unexpected error during request for %s: %s', url, e, exc_info=True)
            raise

    def _check_cache(self, cache_key: str, allow_stale_on_ratelimit: bool=False) -> Optional[Any]:
        """Check local instance cache. Optionally return stale data if rate limited."""
        entry = self.cache.get(cache_key)
        if entry is None or not isinstance(entry, dict):
            return None
        now = time.time()
        is_expired = now >= entry.get('timestamp', 0) + self.cache_expiry
        if not is_expired:
            logger.debug('NewsAPI cache HIT: %s', cache_key)
            return entry.get('data')
        elif allow_stale_on_ratelimit and self.rate_limited:
            logger.warning('NewsAPI cache STALE but using due to rate limit: %s', cache_key)
            return entry.get('data')
        else:
            logger.debug('NewsAPI cache %s: %s', 'EXPIRED' if is_expired else 'MISS', cache_key)
            if is_expired:
                self.cache.pop(cache_key, None)
            return None

    def _update_cache(self, cache_key: str, data: Any) -> None:
        """Update local instance cache."""
        if data is None:
            return
        self.cache[cache_key] = {'timestamp': time.time(), 'data': data}
        if len(self.cache) > 100:
            try:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].get('timestamp', 0))
                self.cache.pop(oldest_key, None)
                logger.debug('Cleaned NewsAPI cache (removed oldest entry)')
            except ValueError:
                pass
            except Exception as e:
                logger.warning('Error during cache cleanup: %s', e)

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection by fetching a small number of headlines."""
        if not self.api_key:
            return (False, 'API key missing.')
        self._check_rate_limit()
        if self.rate_limited:
            return (False, f'Connection test skipped: Currently rate limited until approx {self.rate_limit_until.isoformat()}')  # type: ignore[union-attr]
        logger.info('Testing NewsAPI connection...')
        params = {'q': 'market', 'pageSize': 1, 'language': 'en'}
        try:
            (data, headers) = self._make_request('everything', params)  # type: ignore[misc]
            if data is None:
                rate_info = ''
                if headers and self.debug_mode:
                    rem = headers.get('X-RateLimit-Remaining', 'N/A')
                    lim = headers.get('X-RateLimit-Limit', 'N/A')
                    rate_info = f' (Rate Limit Remaining: {rem}/{lim})'
                return (False, f'Connection failed: Request error or timeout.{rate_info}')
            elif isinstance(data, dict) and data.get('status') == 'ok':
                logger.info('NewsAPI connection successful.')
                return (True, 'Connection successful.')
            else:
                status = data.get('status', 'N/A') if isinstance(data, dict) else 'Invalid Response'
                code = data.get('code', 'N/A') if isinstance(data, dict) else 'N/A'
                message = data.get('message', 'Unknown error') if isinstance(data, dict) else 'Response not a dictionary'
                error_msg = f'Connection failed. Status: {status}, Code: {code}, Message: {message}'
                logger.error(error_msg)
                return (False, error_msg)
        except NewsAPIRateLimitError as rle:
            return (False, f'Connection test failed: Rate Limited ({rle})')
        except Exception as e:
            logger.error('Connection test failed unexpectedly: %s', e, exc_info=True)
            return (False, f'Connection failed: Unexpected error - {e}')

    def get_sentiment_analysis(self, crypto_symbol: str, **kwargs) -> Dict:  # type: ignore[no-untyped-def]
        """
        Fetches news articles for the symbol and analyzes their sentiment.
        Uses cached data as fallback if rate limited.

        Args:
            crypto_symbol (str): The crypto symbol (e.g., 'BTC').
            **kwargs: Optional arguments, e.g., days (int), max_results (int).

        Returns:
            Dict: Dictionary with sentiment analysis results, including 'source_name'.
        """
        default_result = {'sentiment': 'neutral', 'strength': 0.0, 'confidence': 0.0, 'posts_analyzed': 0, 'source_name': 'NewsAPI', 'error': None}
        if not self.api_key:
            default_result['error'] = 'NewsAPI client not initialized or API key missing.'
            return default_result
        days_back = kwargs.get('days', getattr(self.client_config, 'search_days_back', 1))
        max_results = kwargs.get('max_results', getattr(self.client_config, 'max_results_per_query', 20))
        news_articles = self.get_crypto_news(crypto_symbol, days=days_back, max_results=max_results)
        if not news_articles:
            if self.rate_limited:
                default_result['error'] = 'Rate limited and no suitable cached news data found.'
            else:
                logger.warning('No relevant news articles found for %s.', crypto_symbol)
            return default_result
        analysis_result = self._analyze_news_sentiment(news_articles, crypto_symbol)
        analysis_result['source_name'] = 'NewsAPI'
        return analysis_result

    def get_crypto_news(self, crypto_symbol: str, days: Optional[int]=None, max_results: Optional[int]=None) -> List[Dict]:
        """
        Internal: Fetch relevant news articles for a crypto symbol using config defaults.
        Handles request errors and returns an empty list on failure.
        Implements fallback query strategies and handles rate limiting.
        """
        if not self.api_key:
            return []
        asset_upper = crypto_symbol.upper()
        days_back = days if days is not None else getattr(self.client_config, 'search_days_back', 1)
        num_results = max_results if max_results is not None else getattr(self.client_config, 'max_results_per_query', 20)
        try:
            days_back = max(1, int(days_back))
            num_results = max(1, int(num_results))
        except (ValueError, TypeError):
            days_back = 1
            num_results = 20
        cache_key = f'news_{asset_upper}_{days_back}_{num_results}'
        self._check_rate_limit()
        cached_data = self._check_cache(cache_key, allow_stale_on_ratelimit=True)
        if cached_data is not None and isinstance(cached_data, list):
            if self.rate_limited:
                logger.warning('Returning cached news for %s due to active rate limit.', asset_upper)
            return cached_data
        if self.rate_limited:
            logger.error('Cannot fetch news for %s: Rate limited and no usable cache found.', asset_upper)
            return []
        logger.info('Fetching news for %s (Initial Days: %s, Max: %s)', asset_upper, days_back, num_results)
        (articles_attempt_1, articles_attempt_2, articles_attempt_3) = (None, None, None)
        fetch_error = None
        try:
            attempt_description = f'keywords, {days_back}-day range'
            logger.debug('NewsAPI Attempt 1 (%s) for %s...', attempt_description, asset_upper)
            articles_attempt_1 = self._fetch_news_attempt(asset_upper, days_back, num_results, attempt_description)
            if articles_attempt_1 is not None and len(articles_attempt_1) > 0:
                logger.info('Success on Attempt 1 (%s) for %s.', attempt_description, asset_upper)
                self._update_cache(cache_key, articles_attempt_1)
                return articles_attempt_1
            if articles_attempt_1 is not None:
                primary_keyword = None
                all_keywords_map = self.config.get('news_client.asset_keywords', {})
                asset_keywords = all_keywords_map.get(asset_upper)
                if asset_keywords and isinstance(asset_keywords, list) and asset_keywords:
                    primary_keyword = asset_keywords[0]
                if primary_keyword:
                    attempt_description = f"primary keyword '{primary_keyword}', {days_back}-day range"
                    logger.warning('NewsAPI Attempt 1 got 0 results for %s. Retrying with simpler query.', asset_upper)
                    logger.debug('NewsAPI Attempt 2 (%s) for %s...', attempt_description, asset_upper)
                    articles_attempt_2 = self._fetch_news_attempt(asset_upper, days_back, num_results, attempt_description, use_primary_keyword_only=True)
                    if articles_attempt_2 is not None and len(articles_attempt_2) > 0:
                        logger.info('Success on Attempt 2 (%s) for %s.', attempt_description, asset_upper)
                        self._update_cache(cache_key, articles_attempt_2)
                        return articles_attempt_2
                else:
                    logger.debug('Skipping NewsAPI Attempt 2: No primary keyword found.')
            if articles_attempt_1 is not None and (primary_keyword is None or (articles_attempt_2 is not None and len(articles_attempt_2) == 0)):
                fallback_days = 3
                if fallback_days > days_back:
                    attempt_description = f'keywords, {fallback_days}-day range'
                    logger.warning('NewsAPI attempts 1 & 2 got 0 results for %s. Retrying with wider date range.', asset_upper)
                    logger.debug('NewsAPI Attempt 3 (%s) for %s...', attempt_description, asset_upper)
                    articles_attempt_3 = self._fetch_news_attempt(asset_upper, fallback_days, num_results, attempt_description)
                    if articles_attempt_3 is not None and len(articles_attempt_3) > 0:
                        logger.info('Success on Attempt 3 (%s) for %s.', attempt_description, asset_upper)
                        self._update_cache(cache_key, articles_attempt_3)
                        return articles_attempt_3
                else:
                    logger.debug('Skipping NewsAPI Attempt 3: Fallback days not greater than initial days.')
        except NewsAPIRateLimitError:
            fetch_error = 'Rate Limited'
            cached_data_fallback = self._check_cache(cache_key, allow_stale_on_ratelimit=True)
            if cached_data_fallback:
                logger.warning('Returning cached news for %s after hitting rate limit during fetch attempts.', asset_upper)
                return cached_data_fallback
        except Exception as e:
            fetch_error = f'Unexpected error: {e}'
            logger.error('Unexpected error during news fetch attempts for %s: %s', asset_upper, e, exc_info=True)
        if fetch_error:
            logger.error('Failed to fetch news for %s: %s', asset_upper, fetch_error)
        else:
            logger.warning('No news articles found for %s after all attempts.', asset_upper)
        self._update_cache(cache_key, [])
        return []

    def _fetch_news_attempt(self, asset_upper: str, days_back: int, num_results: int, attempt_description: str, use_primary_keyword_only: bool=False) -> Optional[List[Dict]]:
        """
        Performs a single news fetch attempt with specific parameters.
        Returns list of articles on success, empty list for 0 results, None on request error.
        Raises NewsAPIRateLimitError if rate limited.
        """
        try:
            to_date = datetime.now(timezone.utc)
            from_date = to_date - timedelta(days=days_back)
            from_date_str = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            to_date_str = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        except OverflowError:
            logger.error('Date calculation overflow for days_back=%s. Using default 1 day.', days_back)
            to_date = datetime.now(timezone.utc)
            from_date = to_date - timedelta(days=1)
            from_date_str = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            to_date_str = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        query = ''
        all_keywords_map = self.config.get('news_client.asset_keywords', {})
        asset_keywords = all_keywords_map.get(asset_upper)
        if use_primary_keyword_only:
            if asset_keywords and isinstance(asset_keywords, list) and asset_keywords:
                query = f'"{asset_keywords[0]}"'
        else:
            if not asset_keywords or not isinstance(asset_keywords, list):
                asset_keywords = all_keywords_map.get('DEFAULT', [])
                if not asset_keywords:
                    asset_keywords = [asset_upper, self._get_full_name(asset_upper)]
                    logger.debug('Using default symbol/name keywords for %s: %s', asset_upper, asset_keywords)
            query = ' OR '.join([f'"{term}"' for term in asset_keywords if term])
        if not query:
            logger.error('Could not construct valid query for %s (attempt: %s).', asset_upper, attempt_description)
            return None
        params = {'q': query, 'from': from_date_str, 'to': to_date_str, 'language': 'en', 'sortBy': 'relevancy', 'pageSize': min(num_results, 100)}
        (data, _) = self._make_request('everything', params)  # type: ignore[misc]
        processed_articles: list = []
        if data and isinstance(data, dict):
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                if isinstance(articles, list):
                    for article in articles:
                        if not isinstance(article, dict):
                            continue
                        title = article.get('title')
                        desc = article.get('description')
                        published_at = article.get('publishedAt')
                        if not title or not isinstance(title, str):
                            continue
                        if not desc or not isinstance(desc, str):
                            desc = ''
                        if not published_at or not isinstance(published_at, str):
                            continue
                        processed_articles.append({'title': title.strip(), 'description': desc.strip(), 'url': str(article.get('url', '')).strip(), 'published_at': published_at, 'source': str(article.get('source', {}).get('name', '')).strip()})
                    logger.info('News attempt (%s) for %s returned %s articles (found %s total).', attempt_description, asset_upper, len(processed_articles), data.get('totalResults', 0))
                    return processed_articles
                else:
                    logger.warning("NewsAPI response for %s (attempt: %s) status OK, but 'articles' field missing or not a list.", asset_upper, attempt_description)
                    return None
            else:
                logger.error('NewsAPI attempt (%s) failed for %s. Status: %s, Code: %s', attempt_description, asset_upper, data.get('status', 'N/A'), data.get('code', 'N/A'))
                return None
        else:
            logger.error('Failed news attempt (%s) for %s. No response from API.', attempt_description, asset_upper)
            return None

    def _analyze_news_sentiment(self, news_data: List[Dict], crypto_symbol: Optional[str]=None) -> Dict:
        """Internal: Analyze sentiment of news articles using config thresholds and common utils."""
        default_result = {'sentiment': 'neutral', 'strength': 0.0, 'confidence': 0.0, 'posts_analyzed': 0, 'error': None}
        if not self.sentiment_analyzer:
            logger.error('SentimentAnalyzer not initialized in NewsAPI.')
            default_result['error'] = 'Analyzer not available'
            return default_result
        if not isinstance(news_data, list) or not news_data:
            return default_result
        bullish_threshold = getattr(self.sentiment_config, 'bullish_threshold', 0.05)
        bearish_threshold = getattr(self.sentiment_config, 'bearish_threshold', -0.05)
        baseline_count = getattr(self.sentiment_config, 'volume_baseline_count', 20)
        trend_window = getattr(self.sentiment_config, 'trend_window', 5)
        items_with_sentiment: list = []
        analyzed_articles_for_log = []
        total_weighted_sentiment = 0.0
        total_confidence = 0.0
        analyzed_count = 0
        for article in news_data:
            if not isinstance(article, dict):
                continue
            title = article.get('title', '')
            desc = article.get('description', '')
            text = f'{title}. {desc}'.strip()
            if not text or text == '.':
                continue
            try:
                analysis = self.sentiment_analyzer.analyze(text)
                if analysis is None or not isinstance(analysis, dict):
                    continue
                sentiment = analysis.get('score', 0.0)
                confidence = analysis.get('confidence', 0.0)
                if not isinstance(sentiment, (float, int)) or not -1.0 <= sentiment <= 1.0:
                    continue
                if not isinstance(confidence, (float, int)) or not 0.0 <= confidence <= 1.0:
                    continue
                article_copy = article.copy()
                article_copy['sentiment'] = sentiment
                article_copy['confidence'] = confidence
                items_with_sentiment.append(article_copy)
                analyzed_count += 1
                total_weighted_sentiment += sentiment * confidence
                total_confidence += confidence
                label = 'bullish' if sentiment > bullish_threshold else 'bearish' if sentiment < bearish_threshold else 'neutral'
                model_used = getattr(self.sentiment_analyzer, 'model_type', 'unknown')
                log_data = {'headline': article.get('title'), 'source_name': article.get('source'), 'published_at': article.get('published_at'), 'url': article.get('url'), 'sentiment_label': label, 'confidence_score': confidence, 'model': model_used}
                analyzed_articles_for_log.append(log_data)
            except Exception as e:
                url = article.get('url', 'N/A')
                logger.error("Error analyzing sentiment for article '%s': %s", url, e, exc_info=True)
                continue
        if crypto_symbol and analyzed_articles_for_log:
            try:
                log_news_sentiment(crypto_symbol, analyzed_articles_for_log)
            except Exception as log_err:
                logger.error('Failed to log news sentiment data for %s: %s', crypto_symbol, log_err, exc_info=True)
        num_items = analyzed_count
        avg_conf = total_confidence / num_items if num_items > 0 else 0.0
        overall_strength = total_weighted_sentiment / total_confidence if total_confidence > 1e-09 else 0.0
        time_weighted = calculate_time_weighted_sentiment(items_with_sentiment, timestamp_key='published_at')
        volume_adjusted = adjust_sentiment_by_volume(time_weighted, num_items, baseline_count)
        stats = get_sentiment_statistics(items_with_sentiment)
        sentiment_label = 'bullish' if overall_strength > bullish_threshold else 'bearish' if overall_strength < bearish_threshold else 'neutral'
        trend_stats = None
        if crypto_symbol and self.sentiment_tracker:
            try:
                self.sentiment_tracker.add_sentiment(crypto_symbol, overall_strength, avg_conf)
                trend_stats = self.sentiment_tracker.get_statistics(crypto_symbol, window=trend_window)
            except Exception as tracker_e:
                logger.warning('Error interacting with SentimentTracker for %s: %s', crypto_symbol, tracker_e)
        model_used = getattr(self.sentiment_analyzer, 'model_type', 'unknown')
        logger.info('News Sentiment (%s): Label=%s, Strength=%.3f, AvgConf=%.3f, Items=%s, Model=%s', crypto_symbol or 'N/A', sentiment_label, overall_strength, avg_conf, num_items, model_used)
        return {'sentiment': sentiment_label, 'strength': overall_strength, 'time_weighted': time_weighted, 'volume_adjusted': volume_adjusted, 'confidence': avg_conf, 'posts_analyzed': num_items, 'statistics': stats, 'trend_stats': trend_stats, 'error': None}

    def _get_full_name(self, crypto_symbol: str) -> str:
        """Simple helper to get full crypto name."""
        crypto_names_lower = {'btc': 'Bitcoin', 'eth': 'Ethereum', 'xrp': 'Ripple', 'ltc': 'Litecoin', 'doge': 'Dogecoin', 'dot': 'Polkadot', 'sol': 'Solana', 'ada': 'Cardano'}
        symbol_lower = str(crypto_symbol).lower() if crypto_symbol else ''
        return crypto_names_lower.get(symbol_lower, crypto_symbol)