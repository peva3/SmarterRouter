"""Pydantic models for API request/response validation."""

import re

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """A single chat message with validation."""
    
    role: str = Field(
        ...,
        pattern="^(user|assistant|system)$",
        description="Role of the message sender"
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Message content"
    )
    
    @field_validator('content')
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        """Sanitize content by removing null bytes and control characters."""
        # Remove null bytes
        v = v.replace('\x00', '')
        # Remove control characters except newlines, tabs, and carriage returns
        v = ''.join(c for c in v if c in '\n\r\t' or ord(c) >= 32)
        return v.strip()


class ChatCompletionRequest(BaseModel):
    """Chat completion request with validation."""
    
    model: str | None = Field(
        default=None,
        max_length=100,
        description="Optional model override"
    )
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of chat messages"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    @field_validator('model')
    @classmethod
    def validate_model_name(cls, v: str | None) -> str | None:
        """Validate model name doesn't contain dangerous characters."""
        if v is None:
            return v
        # Allow alphanumeric, hyphens, underscores, colons, dots, and slashes
        if not re.match(r'^[\w\-:.\/]+$', v):
            raise ValueError("Model name contains invalid characters")
        return v


class FeedbackRequest(BaseModel):
    """User feedback for a routing decision."""
    
    response_id: str | None = Field(default=None, description="The ID of the chat completion response")
    model_name: str | None = Field(default=None, description="The name of the model (optional if response_id provided)")
    score: float = Field(..., ge=-1.0, le=1.0, description="Feedback score: 1.0 (good), -1.0 (bad), or 0.0-1.0 scale")
    comment: str | None = Field(default=None, max_length=500, description="Optional comment")
    category: str | None = Field(default=None, description="Optional task category")


def sanitize_prompt(prompt: str, max_length: int = 10000) -> str:
    """
    Sanitize a prompt for safe processing.
    
    Args:
        prompt: The input prompt
        max_length: Maximum allowed length
        
    Returns:
        Sanitized prompt string
    """
    if not prompt:
        return ""
    
    # Truncate to max length
    prompt = prompt[:max_length]
    
    # Remove null bytes
    prompt = prompt.replace('\x00', '')
    
    # Remove control characters except common whitespace
    prompt = ''.join(
        c for c in prompt 
        if c in '\n\r\t' or ord(c) >= 32
    )
    
    return prompt.strip()


def sanitize_for_logging(text: str, max_length: int = 200) -> str:
    """
    Sanitize text for safe logging.
    
    Args:
        text: The text to sanitize
        max_length: Maximum length before truncation
        
    Returns:
        Sanitized text safe for logging
    """
    if not text:
        return ""
    
    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Redact potential API keys (OpenAI format: sk-...)
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[API_KEY_REDACTED]', text)
    
    # Redact other common secret patterns
    text = re.sub(r'[a-zA-Z0-9]{32,}', '[POTENTIAL_SECRET]', text)
    
    # Remove newlines for single-line logging
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text
