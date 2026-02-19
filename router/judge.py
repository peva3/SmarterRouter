import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Any

import httpx

from router.config import settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating the quality of an AI model's response to a specific prompt.
Your goal is to provide a quality score between 0.0 and 1.0, where 1.0 is a perfect response.

Consider the following criteria:
1. Accuracy: Is the information correct?
2. Helpfulness: Does it directly answer the user's request?
3. Clarity: Is it easy to understand and well-structured?
4. Conciseness: Is it free of unnecessary filler?
5. Instruction Following: Did it follow all specific constraints in the prompt?

Prompt:
{prompt}

Model Response:
{response}

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text, markdown formatting, code blocks, or explanations outside the JSON.

Required JSON format (respond with EXACTLY this, no markdown):
{{"score": 0.85, "reasoning": "Your brief explanation here"}}

The score must be a number between 0.0 and 1.0.
"""


def _extract_json_from_content(content: str) -> str:
    """Extract JSON from content, handling markdown code blocks and extra text."""
    content = content.strip()
    
    # Handle markdown code blocks (```json ... ``` or ``` ... ```)
    if content.startswith("```"):
        # Find the end of the opening fence
        first_newline = content.find("\n")
        if first_newline != -1:
            # Skip the opening ```json or ``` line
            content = content[first_newline + 1:]
        
        # Find the closing ```
        closing_fence = content.rfind("```")
        if closing_fence != -1:
            content = content[:closing_fence]
        
        content = content.strip()
    
    # If there are multiple JSON objects or extra text, try to find the first valid JSON object
    # Look for {"score": pattern
    start_idx = content.find('{"score"')
    if start_idx != -1:
        content = content[start_idx:]
        # Find the matching closing brace
        brace_count = 0
        end_idx = 0
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        content = content[:end_idx]
    
    return content.strip()


class JudgeClient:
    # Patterns indicating obvious failures - no need to waste tokens judging these
    FAILURE_PATTERNS = [
        re.compile(r"(?i)^\s*($|i'?m\s+sorry)"),  # Empty or immediate apology
        re.compile(r"(?i)i\s+(cannot|can't|am\s+unable\s+to)\s+(help|assist)"),
        re.compile(r"(?i)as\s+an\s+ai\s+language\s+model"),
        re.compile(r"(?i)i\s+don'?t\s+have\s+(access|information)"),
        re.compile(r"(?i)unable\s+to\s+(process|generate|complete)"),
        re.compile(r"(?i)context\s+window\s+(exceeded|too\s+long)"),
        re.compile(r"(?i)rate\s+limit"),
    ]
    
    def __init__(self):
        self.enabled = settings.judge_enabled
        self.model = settings.judge_model
        self.base_url = settings.judge_base_url
        self.api_key = settings.judge_api_key
        self.http_referer = settings.judge_http_referer
        self.x_title = settings.judge_x_title
        self.max_retries = settings.judge_max_retries
        self.base_delay = settings.judge_retry_base_delay
        
        # Shared client for connection pooling
        self._client: httpx.AsyncClient | None = None
        
        # Rate limiting tracking
        self._request_times: deque[float] = deque(maxlen=100)
        self._rate_limit_per_minute = 50  # Conservative default
        self._max_concurrent = 3  # Limit concurrent API calls
    
    def _is_obvious_failure(self, response: str) -> bool:
        """Check if response matches known failure patterns to skip API call."""
        if not response or len(response.strip()) < 20:
            return True
        
        for pattern in self.FAILURE_PATTERNS:
            if pattern.search(response):
                return True
        
        return False
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self._client
    
    async def _respect_rate_limit(self) -> None:
        """Pause if approaching rate limit to avoid 429 errors."""
        now = time.time()
        one_minute_ago = now - 60
        
        # Clean old timestamps
        while self._request_times and self._request_times[0] < one_minute_ago:
            self._request_times.popleft()
        
        # If near limit, wait until we have capacity
        if len(self._request_times) >= self._rate_limit_per_minute * 0.8:
            delay = 60 - (now - self._request_times[0])
            if delay > 0:
                logger.debug(f"Rate limit approaching, pausing {delay:.1f}s")
                await asyncio.sleep(min(delay, 10))  # Cap at 10s
    
    async def score_responses_batch(
        self, 
        prompt_response_pairs: list[tuple[str, str]],
        max_concurrent: int = 3
    ) -> list[float]:
        """Score multiple responses in parallel with rate limiting.
        
        Args:
            prompt_response_pairs: List of (prompt, response) tuples
            max_concurrent: Maximum concurrent API calls (default 3)
            
        Returns:
            List of scores matching input order
        """
        if not prompt_response_pairs:
            return []
        
        # Always apply pattern pre-filtering (saves tokens even when judge is enabled)
        # and return fallback scores when judge is disabled
        if not self.enabled:
            return [
                0.1 if self._is_obvious_failure(r) else 0.5 
                for _, r in prompt_response_pairs
            ]
        
        # Pre-filter obvious failures
        filtered_pairs = []
        pre_assigned_scores: dict[int, float] = {}  # Index -> score mapping
        
        for i, (prompt, response) in enumerate(prompt_response_pairs):
            if self._is_obvious_failure(response):
                pre_assigned_scores[i] = 0.1  # Low score for failures
                logger.debug(f"Skipped judge API call for obvious failure at index {i}")
            else:
                filtered_pairs.append((i, prompt, response))
        
        # Score remaining items with semaphore-controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def score_single(index: int, prompt: str, response: str) -> tuple[int, float]:
            async with semaphore:
                await self._respect_rate_limit()
                
                try:
                    score = await self._score_single(prompt, response)
                    self._request_times.append(time.time())
                    return index, score
                except Exception as e:
                    logger.debug(f"Judge failed for index {index}: {e}")
                    return index, 0.5
        
        # Create tasks for all items that need API calls
        tasks = [
            score_single(idx, prompt, response)
            for idx, prompt, response in filtered_pairs
        ]
        
        # Execute all tasks
        scored_results = await asyncio.gather(*tasks)
        
        # Combine results
        final_scores: list[float] = [0.0] * len(prompt_response_pairs)
        
        # Add pre-assigned scores for failures
        for idx, score in pre_assigned_scores.items():
            final_scores[idx] = score
        
        # Add scored results
        for idx, score in scored_results:
            final_scores[idx] = score
        
        logger.debug(f"Batch scored {len(prompt_response_pairs)} responses, "
                    f"saved {len(pre_assigned_scores)} API calls via pre-filter")
        
        return final_scores
    
    async def _score_single(self, prompt: str, response: str) -> float:
        """Score a single response using the judge API."""
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)
        
        client = await self._get_client()
        
        for attempt in range(self.max_retries):
            try:
                return await self._try_score(client, judge_prompt, prompt, response)
            except (httpx.HTTPStatusError, httpx.NetworkError, httpx.TimeoutException) as e:
                if attempt == self.max_retries - 1:
                    break
                
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400:
                    break  # Don't retry client errors
                
                delay = self._calculate_backoff(attempt)
                await asyncio.sleep(delay)
        
        # Fallback
        return 0.5 if response and len(response.strip()) > 0 else 0.0
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.base_delay * (2 ** attempt)
    
    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def score_response(self, prompt: str, response: str) -> float:
        """Score a model response using the configured judge model."""
        if not self.enabled:
            # Fallback: neutral score for non-empty responses
            if response and len(response.strip()) > 0:
                return 0.5
            return 0.0

        if not response or len(response.strip()) < 5:
            return 0.0

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._try_score(client=None, judge_prompt=judge_prompt, prompt=prompt, response=response)
            except (httpx.HTTPStatusError, httpx.NetworkError, httpx.TimeoutException) as e:
                last_exception = e
                
                # Check if it's a retryable error
                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    
                    # Log the actual error response for debugging
                    if status_code == 400:
                        try:
                            error_body = e.response.text
                            logger.warning(f"Judge 400 error (attempt {attempt + 1}/{self.max_retries}): {error_body[:500]}")
                        except Exception:
                            logger.warning(f"Judge 400 error (attempt {attempt + 1}/{self.max_retries}): {e}")
                        # Don't retry 400 errors (client error)
                        break
                    elif status_code == 429:
                        # Rate limited - definitely retry with backoff
                        logger.warning(f"Judge rate limited (429) on attempt {attempt + 1}/{self.max_retries}")
                    elif status_code >= 500:
                        # Server errors - retry
                        logger.warning(f"Judge server error {status_code} on attempt {attempt + 1}/{self.max_retries}")
                    else:
                        # Other 4xx errors - don't retry
                        logger.warning(f"Judge client error {status_code} on attempt {attempt + 1}/{self.max_retries}: {e}")
                        break
                else:
                    # Network or timeout errors - retry
                    logger.warning(f"Judge network/timeout error on attempt {attempt + 1}/{self.max_retries}: {e}")
                
                # Calculate exponential backoff delay
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Retrying judge request in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                
            except Exception as e:
                # Unexpected error - log and break
                logger.error(f"Unexpected judge error: {e}")
                last_exception = e
                break
        
        # All retries exhausted or non-retryable error
        if last_exception:
            error_msg = str(last_exception) if str(last_exception) else type(last_exception).__name__
            logger.warning(f"Judge scoring failed after {self.max_retries} attempts: {error_msg}. Falling back to basic check.")
        
        # Fallback: neutral score for non-empty responses
        if response and len(response.strip()) > 0:
            return 0.5
        return 0.0

    async def _try_score(self, client: httpx.AsyncClient | None, judge_prompt: str, prompt: str, response: str) -> float:
        """Make a single scoring attempt."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Add optional OpenRouter headers if configured
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that evaluates AI responses.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            # Note: response_format is not used for universal compatibility
            # We rely on prompt engineering to get JSON output
        }

        if client is None:
            async with httpx.AsyncClient(timeout=30.0) as client:
                return await self._make_request(client, headers, payload)
        else:
            return await self._make_request(client, headers, payload)

    async def _make_request(self, client: httpx.AsyncClient, headers: dict, payload: dict) -> float:
        """Make the HTTP request and parse the response."""
        res = await client.post(
            f"{self.base_url.rstrip('/')}/chat/completions", 
            json=payload, 
            headers=headers
        )
        res.raise_for_status()
        data = res.json()

        # Validate response structure
        if not data.get("choices") or len(data["choices"]) == 0:
            logger.warning("Judge returned empty choices array")
            raise ValueError("Empty choices in judge response")
        
        message = data["choices"][0].get("message", {})
        content = message.get("content", "")
        
        if not content or not content.strip():
            logger.warning("Judge returned empty content")
            raise ValueError("Empty content in judge response")
        
        # Extract and parse JSON response
        json_content = _extract_json_from_content(content)
        try:
            result = json.loads(json_content)
        except json.JSONDecodeError as json_err:
            logger.warning(f"Judge returned invalid JSON: {content[:200]}... Error: {json_err}")
            raise ValueError(f"Invalid JSON from judge: {json_err}")
        
        score = float(result.get("score", 0.0))

        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))

        logger.debug(
            f"Judge score for response: {score} (Reasoning: {result.get('reasoning')})"
        )
        return score
