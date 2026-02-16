import json
import logging
from typing import Any

import httpx

from router.config import settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating the quality of an AI model's response to a specific prompt.
Your goal is to provide a quality score between 0.0 and 1.0, where 1.0 is a perfect response.

Consider the following criteria:
1. Accuracy: Is the information correct?
2. Helpfuless: Does it directly answer the user's request?
3. Clarity: Is it easy to understand and well-structured?
4. Conciseness: Is it free of unnecessary filler?
5. Instruction Following: Did it follow all specific constraints in the prompt?

Prompt:
{prompt}

Model Response:
{response}

Please provide your evaluation in JSON format with a "score" (0.0 to 1.0) and a brief "reasoning" field.
Example: {{"score": 0.85, "reasoning": "The response was accurate and helpful, but slightly verbose."}}
"""

class JudgeClient:
    def __init__(self):
        self.enabled = settings.judge_enabled
        self.model = settings.judge_model
        self.base_url = settings.judge_base_url
        self.api_key = settings.judge_api_key

    async def score_response(self, prompt: str, response: str) -> float:
        """Score a model response using the configured judge model."""
        if not self.enabled:
            return 1.0 if len(response.strip()) > 10 else 0.0

        if not response or len(response.strip()) < 5:
            return 0.0

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that evaluates AI responses."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }

                res = await client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers
                )
                res.raise_for_status()
                data = res.json()
                
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                score = float(result.get("score", 0.0))
                
                # Ensure score is within bounds
                score = max(0.0, min(1.0, score))
                
                logger.debug(f"Judge score for response: {score} (Reasoning: {result.get('reasoning')})")
                return score

        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}. Falling back to basic check.")
            return 1.0 if len(response.strip()) > 10 else 0.0
