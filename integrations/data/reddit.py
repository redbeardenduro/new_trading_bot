"""
Integration module for fetching Reddit posts and performing sentiment analysis,
driven by central configuration. Implements the ISentimentSource interface.

Uses the PRAW library to interact with the Reddit API and performs sentiment
analysis using shared utilities (SentimentAnalyzer with RoBERTa model preference,
SentimentTracker). Includes rate limiting, caching, and structured logging
of analyzed post sentiments.
"""
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from core.config import BotConfig
    BotConfigType: Any = BotConfig
except ImportError:
    import logging
    logging.error('CRITICAL: Failed to import BotConfig. RedditAPI may not function.')
    BotConfigType = None
try:
    import praw
    import prawcore
    praw_available = True
except ImportError:
    print('WARNING: PRAW library not found. Reddit integration disabled. Run: pip install praw')
    praw = None
    prawcore = None
    praw_available = False

    class PrawcoreException(Exception):
        pass

    class OAuthException(PrawcoreException):
        pass

    class Forbidden(PrawcoreException):
        pass

    class NotFound(PrawcoreException):
        pass

    class RequestException(PrawcoreException):
        pass

    class Redirect(PrawcoreException):
        pass
try:
    from common.common_logger import CACHE_DIR, get_logger
    logger = get_logger('reddit_api')
except Exception as e:
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger('reddit_api_fallback')
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
                                       normalize_sentiment_score,
                                       preprocess_text)
    sentiment_utils_available = True
except ImportError:
    logger.error('CRITICAL: Failed to import sentiment utilities from utils.sentiment_utils. Reddit analysis will be impaired.')

    class SentimentTracker:

        def __init__(self, history_length: int=0) -> None:
            pass

        def add_sentiment(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_statistics(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {}

    class SentimentAnalyzer:

        def __init__(self, model_type: str='') -> None:
            pass

        def analyze(self, t: str) -> Dict[str, Any]:
            return {'score': 0.0, 'confidence': 0.0}

    def calculate_time_weighted_sentiment(items: List[Any], **kwargs: Any) -> float:
        return 0.0

    def adjust_sentiment_by_volume(s: float, i: int, b: int) -> float:
        return s

    def get_sentiment_statistics(items: List[Any]) -> Dict[str, Any]:
        return {}

    def normalize_sentiment_score(s: float, **kwargs: Any) -> float:
        return s

    def preprocess_text(t: str) -> str:
        return t

def preprocess_post_text(post: Union[Dict, Any]) -> str:
    """Extracts and combines title and selftext from a PRAW post object or dict."""
    title = ''
    selftext = ''
    if isinstance(post, dict):
        title = post.get('title', '')
        selftext = post.get('selftext', '')
    elif hasattr(post, 'title') and hasattr(post, 'selftext'):
        title = getattr(post, 'title', '') or ''
        selftext = getattr(post, 'selftext', '') or ''
    else:
        logger.debug('Unknown type for post preprocessing: %s', type(post))
    full_text = f'{str(title)}. {str(selftext)}'.strip()
    if full_text == '.':
        return ''
    full_text = ' '.join(full_text.split())
    return full_text

def extract_subreddit_from_permalink(permalink: Optional[str]) -> Optional[str]:
    """Extracts the subreddit name from a Reddit permalink using regex."""
    if not permalink or not isinstance(permalink, str):
        return None
    match = re.search('/r/([^/]+)/', permalink)
    if match:
        return match.group(1)
    logger.debug('Could not extract subreddit from permalink: %s', permalink)
    return None

class RedditRateLimiter:
    """Simple rate limiter for Reddit API requests using delays."""

    def __init__(self, min_delay_sec: float=1.1):
        self._min_delay = max(0.1, min_delay_sec)
        self._last_request_time: float = 0.0
        self._lock = threading.RLock()
        logger.info('RedditRateLimiter initialized: Min Delay=%ss', self._min_delay)

    def wait_if_needed(self) -> None:
        """Wait until the minimum delay since the last request has passed."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            wait_time = self._min_delay - elapsed
            if wait_time > 0:
                logger.debug('Reddit rate limit delay: Waiting %.2fs.', wait_time)
                time.sleep(wait_time)
            self._last_request_time = time.time()

def log_reddit_sentiment(symbol: str, analyzed_posts: List[Dict]) -> None:
    """
    Logs analyzed Reddit posts to a JSON file, appending only new entries
    and deduplicating based on URL.

    Args:
        symbol (str): The crypto symbol (e.g., 'BTC').
        analyzed_posts (List[Dict]): A list of dictionaries, each containing
                                      analyzed post data. Expected keys:
                                      'subreddit', 'post_title', 'post_url',
                                      'model', 'sentiment_label', 'confidence_score'.
    """
    if not symbol or not isinstance(symbol, str):
        logger.error('Cannot log Reddit sentiment: Invalid symbol provided.')
        return
    if not analyzed_posts or not isinstance(analyzed_posts, list):
        logger.debug('No new Reddit posts provided to log for %s.', symbol)
        return
    log_file = CACHE_DIR / f'{symbol.upper()}_Reddit_sentiment.json'
    logger.debug('Logging %s analyzed Reddit posts for %s to %s', len(analyzed_posts), symbol, log_file)
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
                            if isinstance(entry, dict) and entry.get('post_url'):
                                existing_urls.add(entry['post_url'])
                    else:
                        logger.warning('Existing log file %s is not a list. Overwriting.', log_file)
        except json.JSONDecodeError:
            logger.error('Error decoding JSON from %s. Log file might be corrupted. Overwriting.', log_file)
            existing_logs: list = []
            existing_urls = set()
        except OSError as e:
            logger.error('OS error reading log file %s: %s. Will attempt to overwrite.', log_file, e)
            existing_logs: list = []
            existing_urls = set()
        except Exception as e:
            logger.error('Unexpected error loading log file %s: %s. Will attempt to overwrite.', log_file, e, exc_info=True)
            existing_logs: list = []
            existing_urls = set()
    new_entries_to_add: list = []
    log_timestamp = datetime.now(timezone.utc).isoformat()
    for post in analyzed_posts:
        if not isinstance(post, dict):
            continue
        url = post.get('post_url')
        if url and url not in existing_urls:
            log_entry = {'timestamp': log_timestamp, 'symbol': symbol.upper(), 'subreddit': post.get('subreddit'), 'post_title': post.get('post_title'), 'post_url': url, 'model': post.get('model'), 'sentiment_label': post.get('sentiment_label'), 'confidence_score': post.get('confidence_score')}
            if all((log_entry.get(k) is not None for k in ['post_title', 'post_url', 'model', 'sentiment_label'])):
                new_entries_to_add.append(log_entry)
                existing_urls.add(url)
            else:
                logger.warning('Skipping Reddit post log due to missing essential fields: %s', url)
    if not new_entries_to_add:
        logger.info('No new unique Reddit posts to log for %s.', symbol)
        return
    all_logs = existing_logs + new_entries_to_add
    temp_file = log_file.with_suffix(f'.tmp_{time.time_ns()}')
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with temp_file.open('w', encoding='utf-8') as f:
            json.dump(all_logs, f, indent=2)
        temp_file.replace(log_file)
        logger.info('Successfully appended %s new Reddit sentiment entries for %s to %s', len(new_entries_to_add), symbol, log_file.name)
    except (OSError, TypeError, ValueError) as e:
        logger.error('Error saving Reddit sentiment log to %s: %s', log_file, e, exc_info=True)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.error('Unexpected error saving Reddit sentiment log: %s', e, exc_info=True)
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass

class RedditAPI(ISentimentSourceType):
    """Client for fetching Reddit posts and analyzing sentiment using BotConfig."""

    def __init__(self, config: Any) -> None:
        """Initializes the PRAW client using loaded credentials from BotConfig."""
        if BotConfigType is None:
            raise ValueError('BotConfig is not available for RedditAPI initialization.')
        if not sentiment_utils_available:
            raise ImportError('Sentiment utilities could not be imported. RedditAPI cannot function.')
        if not praw_available:
            raise ImportError('PRAW library is not installed. RedditAPI cannot function.')
        self.config = config
        self.client_config = config.reddit_client
        self.sentiment_config = config.sentiment
        self.reddit: Optional[praw.Reddit] = None
        self.rate_limiter = RedditRateLimiter()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: int = getattr(self.client_config, 'cache_expiry_seconds', 1800)
        self.sentiment_analyzer = SentimentAnalyzer(model_type='roberta')
        tracker_hist_len = getattr(self.sentiment_config, 'history_length', 30)
        self.sentiment_tracker = SentimentTracker(history_length=tracker_hist_len)
        try:
            creds = config.api_credentials.reddit
            req_keys = ['client_id', 'client_secret', 'user_agent', 'username', 'password']
            if not all((getattr(creds, k, None) for k in req_keys)):
                logger.error('Reddit credentials missing or incomplete in config.')
                raise ValueError('Reddit credentials missing or incomplete.')
            self.reddit = praw.Reddit(client_id=creds.client_id, client_secret=creds.client_secret, username=creds.username, password=creds.password, user_agent=creds.user_agent, check_for_async=False)
            authenticated_user = self.reddit.user.me()
            if authenticated_user is None:
                logger.critical('Reddit authentication check failed: PRAW returned None for authenticated user. Reddit functionality disabled.')
                self.reddit = None
            else:
                logger.info("Reddit API client initialized and authenticated as '%s'.", authenticated_user.name)
        except prawcore.exceptions.OAuthException as e:
            logger.critical('Reddit OAuth failed: %s. Check credentials. Reddit functionality disabled.', e, exc_info=True)
            self.reddit = None
        except prawcore.exceptions.RequestException as e:
            logger.critical('Reddit API request error during initialization: %s. Reddit functionality may be limited.', e, exc_info=True)
            self.reddit = None
        except Exception as e:
            logger.critical('Failed to initialize PRAW client: %s. Reddit functionality disabled.', e, exc_info=True)
            self.reddit = None

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check local instance cache."""
        entry = self.cache.get(cache_key)
        if entry is None or not isinstance(entry, dict):
            return None
        expiry_time = entry.get('timestamp', 0) + self.cache_expiry
        if time.time() < expiry_time:
            logger.debug('Reddit cache HIT: %s', cache_key)
            return entry.get('data')
        else:
            logger.debug('Reddit cache MISS/EXPIRED: %s', cache_key)
            self.cache.pop(cache_key, None)
            return None

    def _update_cache(self, cache_key: str, data: Any) -> None:
        """Update local instance cache."""
        if data is None:
            return
        self.cache[cache_key] = {'timestamp': time.time(), 'data': data}
        if len(self.cache) > 50:
            try:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].get('timestamp', 0))
                self.cache.pop(oldest_key, None)
                logger.debug('Cleaned Reddit cache (removed oldest)')
            except ValueError:
                pass
            except Exception as e:
                logger.warning('Error during Reddit cache cleanup: %s', e)

    def test_connection(self) -> Tuple[bool, str]:
        """Test Reddit API connection by accessing the authenticated user."""
        if not self.reddit:
            return (False, 'PRAW client not initialized or authentication failed.')
        try:
            cache_key = 'reddit_test_connection'
            cached = self._check_cache(cache_key)
            if cached and isinstance(cached, str):
                return (True, cached)
            self.rate_limiter.wait_if_needed()
            user = self.reddit.user.me()
            if user is None:
                raise ConnectionError('Authentication check failed: user object is None.')
            result = f'Connection successful. Authenticated as user: {user.name}'
            logger.info(result)
            self._update_cache(cache_key, result)
            return (True, result)
        except prawcore.exceptions.Forbidden as e:
            logger.error('Reddit connection test failed - Forbidden: %s. Check API permissions/scopes.', e)
            return (False, f'Forbidden: {e}')
        except prawcore.exceptions.OAuthException as e:
            logger.error('Reddit connection test failed - OAuthException: %s. Check credentials.', e)
            return (False, f'OAuthException: {e}')
        except prawcore.exceptions.RequestException as e:
            logger.error('Reddit connection test failed - Request Error: %s', e)
            return (False, f'Request Error: {e}')
        except ConnectionError as e:
            logger.error('Reddit connection test failed: %s', e)
            return (False, str(e))
        except Exception as e:
            logger.error('Reddit connection test failed unexpectedly: %s', e, exc_info=True)
            return (False, f'Unexpected error: {e}')

    def get_sentiment_analysis(self, crypto_symbol: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Fetches Reddit posts for the symbol and analyzes their sentiment.

        Args:
            crypto_symbol (str): The crypto symbol (e.g., 'BTC').
            **kwargs: Optional arguments, e.g., limit (int) for number of posts.

        Returns:
            Dict: Dictionary with sentiment analysis results, including 'source_name'.
        """
        default_result = {'sentiment': 'neutral', 'strength': 0.0, 'confidence': 0.0, 'posts_analyzed': 0, 'source_name': 'Reddit', 'error': None}
        if not self.reddit:
            default_result['error'] = 'PRAW client not initialized or authentication failed.'
            return default_result
        limit = kwargs.get('limit', getattr(self.client_config, 'limit_per_subreddit', 10))
        try:
            limit = max(1, int(limit))
        except (ValueError, TypeError):
            limit = 10
        (success, posts_or_error) = self.get_crypto_posts(crypto_symbol, limit=limit)
        if not success or not isinstance(posts_or_error, list):
            error_msg = posts_or_error if isinstance(posts_or_error, str) else 'Failed to fetch posts'
            default_result['error'] = error_msg
            logger.error('Cannot perform sentiment analysis for %s: %s', crypto_symbol, error_msg)
            return default_result
        analysis_result = self._analyze_posts_sentiment(posts_or_error, crypto_symbol)
        analysis_result['source_name'] = 'Reddit'
        return analysis_result

    def search_subreddit_posts(self, subreddit_name: str, query: Optional[str]=None, limit: int=25, time_filter: Optional[str]=None) -> Tuple[bool, Union[List[Dict], str]]:
        """Search a subreddit for posts matching a query, using config defaults."""
        if not self.reddit:
            return (False, 'PRAW client not initialized or authentication failed.')
        default_tf = getattr(self.client_config, 'search_time_filter', 'week')
        final_time_filter = time_filter if time_filter else default_tf
        valid_time_filters = ['all', 'day', 'hour', 'month', 'week', 'year']
        if not isinstance(final_time_filter, str) or final_time_filter.lower() not in valid_time_filters:
            logger.warning("Invalid time_filter '%s'. Defaulting to '%s'.", final_time_filter, default_tf)
            final_time_filter = default_tf
        final_time_filter = final_time_filter.lower()
        query_str = query if query else 'hot'
        query_part = hashlib.md5(query_str.encode()).hexdigest()[:8] if len(query_str) > 20 else query_str.replace(' ', '_')
        cache_key = f'reddit_search_{subreddit_name}_{query_part}_{limit}_{final_time_filter}'
        cached_data = self._check_cache(cache_key)
        if cached_data is not None and isinstance(cached_data, list):
            return (True, cached_data)
        logger.info("Searching r/%s for '%s' (Limit=%s, Time=%s)", subreddit_name, query_str, limit, final_time_filter)
        processed_posts: list = []
        try:
            self.rate_limiter.wait_if_needed()
            subreddit = self.reddit.subreddit(subreddit_name)
            if query:
                results_gen = subreddit.search(query, limit=limit, time_filter=final_time_filter, sort='new')
            else:
                results_gen = subreddit.hot(limit=limit)
            posts = list(results_gen)
            for post in posts:
                author_obj = getattr(post, 'author', None)
                author_name = str(author_obj) if author_obj else '[deleted]'
                created_utc = getattr(post, 'created_utc', None)
                if created_utc is None:
                    continue
                processed_posts.append({'id': getattr(post, 'id', 'N/A'), 'title': getattr(post, 'title', ''), 'author': author_name, 'created_utc': float(created_utc), 'score': getattr(post, 'score', 0), 'upvote_ratio': getattr(post, 'upvote_ratio', None), 'num_comments': getattr(post, 'num_comments', 0), 'url': getattr(post, 'url', ''), 'selftext': getattr(post, 'selftext', ''), 'permalink': f"https://www.reddit.com{getattr(post, 'permalink', '')}"})
            logger.info('Retrieved %s posts from r/%s.', len(processed_posts), subreddit_name)
            self._update_cache(cache_key, processed_posts)
            return (True, processed_posts)
        except prawcore.exceptions.Redirect as e:
            error_msg = f'Subreddit r/{subreddit_name} not found or redirected: {e}'
            logger.error(error_msg)
            return (False, error_msg)
        except prawcore.exceptions.NotFound as e:
            error_msg = f'Subreddit r/{subreddit_name} not found: {e}'
            logger.error(error_msg)
            return (False, error_msg)
        except prawcore.exceptions.Forbidden as e:
            error_msg = f'Forbidden access to r/{subreddit_name}: {e}'
            logger.error(error_msg)
            return (False, error_msg)
        except prawcore.exceptions.RequestException as e:
            error_msg = f'PRAW request error searching r/{subreddit_name}: {e}'
            logger.error(error_msg, exc_info=True)
            return (False, error_msg)
        except prawcore.exceptions.PrawcoreException as e:
            error_msg = f'PRAW Core error searching r/{subreddit_name}: {e}'
            logger.error(error_msg, exc_info=True)
            return (False, error_msg)
        except Exception as e:
            error_msg = f'Unexpected error searching r/{subreddit_name}: {e}'
            logger.error(error_msg, exc_info=True)
            return (False, error_msg)

    def get_crypto_subreddits_posts(self, crypto_symbol: str, limit_per_sub: Optional[int]=None) -> Tuple[bool, Union[List[Dict], str]]:
        """Fetch posts about a crypto symbol from relevant subreddits, using config limit."""
        if not self.reddit:
            return (False, 'PRAW client not initialized or authentication failed.')
        default_limit = getattr(self.client_config, 'limit_per_subreddit', 10)
        final_limit = limit_per_sub if limit_per_sub is not None else default_limit
        try:
            final_limit = max(1, int(final_limit))
        except (ValueError, TypeError):
            final_limit = 10
        cache_key = f'reddit_crypto_{crypto_symbol}_{final_limit}'
        cached_data = self._check_cache(cache_key)
        if cached_data is not None and isinstance(cached_data, list):
            return (True, cached_data)
        logger.info('Fetching Reddit posts for %s (Limit/Sub: %s)', crypto_symbol, final_limit)
        subreddits = ['CryptoCurrency', 'CryptoMarkets', 'BitcoinMarkets', 'altcoin', 'CryptoTechnology', 'investing', 'wallstreetbets']
        symbol_upper = crypto_symbol.upper()
        if symbol_upper == 'BTC':
            subreddits.extend(['Bitcoin', 'BitcoinDiscussion'])
        elif symbol_upper == 'ETH':
            subreddits.extend(['ethereum', 'ethtrader', 'ethfinance'])
        elif symbol_upper == 'DOGE':
            subreddits.extend(['dogecoin'])
        full_name = self._get_full_name(crypto_symbol)
        query = f'("{crypto_symbol}" OR "{full_name}")'
        all_posts_dict = {}
        errors: list = []
        for sub_name in subreddits:
            (success, result) = self.search_subreddit_posts(sub_name, query=query, limit=final_limit, time_filter=getattr(self.client_config, 'search_time_filter', 'week'))
            if success and isinstance(result, list):
                logger.debug('Fetched %s posts from r/%s for %s', len(result), sub_name, crypto_symbol)
                for post in result:
                    if isinstance(post, dict) and 'id' in post:
                        all_posts_dict[post['id']] = post
            elif not success:
                errors.append(f"r/{sub_name}: {(result if isinstance(result, str) else 'Unknown error')}")
                logger.warning('Failed to fetch from r/%s for %s: %s', sub_name, crypto_symbol, result)
        if errors:
            logger.warning('Errors fetching some Reddit posts for %s: %s', crypto_symbol, '; '.join(errors))
        if not all_posts_dict:
            logger.warning('No Reddit posts found across relevant subreddits for %s.', crypto_symbol)
            self._update_cache(cache_key, [])
            return (True, [])
        all_posts_list = sorted(all_posts_dict.values(), key=lambda p: p.get('created_utc', 0), reverse=True)
        logger.info('Retrieved %s unique Reddit posts for %s.', len(all_posts_list), crypto_symbol)
        self._update_cache(cache_key, all_posts_list)
        return (True, all_posts_list)

    def get_crypto_posts(self, crypto_symbol: str, limit: Optional[int]=None) -> Tuple[bool, Union[List[Dict], str]]:
        """Public method to get crypto posts (calls internal fetcher)."""
        if not self.reddit:
            return (False, 'PRAW client not initialized or authentication failed.')
        return self.get_crypto_subreddits_posts(crypto_symbol, limit_per_sub=limit)

    def _analyze_posts_sentiment(self, posts_input: Union[List[Dict], Tuple[bool, Union[List[Dict], str]]], crypto_symbol: Optional[str]=None) -> Dict:
        """Internal: Analyze sentiment of Reddit posts using config thresholds and common utils."""
        default_result = {'sentiment': 'neutral', 'strength': 0.0, 'confidence': 0.0, 'posts_analyzed': 0, 'error': None}
        if not self.sentiment_analyzer:
            logger.error('SentimentAnalyzer not initialized in RedditAPI.')
            default_result['error'] = 'Analyzer not available'
            return default_result
        posts: list = []
        if isinstance(posts_input, tuple):
            (success, result) = posts_input
            posts = result if success and isinstance(result, list) else []
        elif isinstance(posts_input, list):
            posts = posts_input
        if not posts:
            logger.warning('No valid Reddit posts provided for sentiment analysis.')
            return default_result
        bullish_threshold = getattr(self.sentiment_config, 'bullish_threshold', 0.05)
        bearish_threshold = getattr(self.sentiment_config, 'bearish_threshold', -0.05)
        baseline_count = getattr(self.sentiment_config, 'volume_baseline_count', 20)
        trend_window = getattr(self.sentiment_config, 'trend_window', 5)
        items_with_sentiment: list = []
        analyzed_posts_for_log = []
        total_weighted_sentiment = 0.0
        total_confidence = 0.0
        analyzed_count = 0
        model_used = getattr(self.sentiment_analyzer, 'model_type', 'unknown')
        for post in posts:
            if not isinstance(post, dict):
                continue
            text = preprocess_post_text(post)
            if not text:
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
                post_copy = post.copy()
                post_copy['sentiment'] = sentiment
                post_copy['confidence'] = confidence
                items_with_sentiment.append(post_copy)
                analyzed_count += 1
                total_weighted_sentiment += sentiment * confidence
                total_confidence += confidence
                label = 'bullish' if sentiment > bullish_threshold else 'bearish' if sentiment < bearish_threshold else 'neutral'
                subreddit = extract_subreddit_from_permalink(post.get('permalink'))
                log_data = {'subreddit': subreddit, 'post_title': post.get('title'), 'post_url': post.get('url'), 'model': model_used, 'sentiment_label': label, 'confidence_score': confidence}
                analyzed_posts_for_log.append(log_data)
            except Exception as e:
                post_id = post.get('id', 'N/A')
                logger.error("Error analyzing sentiment for post '%s': %s", post_id, e, exc_info=True)
                continue
        if crypto_symbol and analyzed_posts_for_log:
            try:
                log_reddit_sentiment(crypto_symbol, analyzed_posts_for_log)
            except Exception as log_err:
                logger.error('Failed to log Reddit sentiment data for %s: %s', crypto_symbol, log_err, exc_info=True)
        num_items = analyzed_count
        avg_conf = total_confidence / num_items if num_items > 0 else 0.0
        overall_strength = total_weighted_sentiment / total_confidence if total_confidence > 1e-09 else 0.0
        time_weighted = calculate_time_weighted_sentiment(items_with_sentiment, timestamp_key='created_utc')
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
        logger.info('Reddit Sentiment (%s): Label=%s, Strength=%.3f, AvgConf=%.3f, Items=%s, Model=%s', crypto_symbol or 'N/A', sentiment_label, overall_strength, avg_conf, num_items, model_used)
        return {'sentiment': sentiment_label, 'strength': overall_strength, 'time_weighted': time_weighted, 'volume_adjusted': volume_adjusted, 'confidence': avg_conf, 'posts_analyzed': num_items, 'statistics': stats, 'trend_stats': trend_stats, 'error': None}

    def _get_full_name(self, crypto_symbol: str) -> str:
        """Simple helper to get full crypto name."""
        crypto_names_lower = {'btc': 'Bitcoin', 'eth': 'Ethereum', 'xrp': 'Ripple', 'ltc': 'Litecoin', 'doge': 'Dogecoin', 'dot': 'Polkadot', 'sol': 'Solana', 'ada': 'Cardano'}
        return crypto_names_lower.get(crypto_symbol.lower(), crypto_symbol)