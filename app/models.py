"""
Pydantic v2 Request & Response Models for Decision Gateway.
Provides strict validation, normalization, and polymorphic question handling.
"""

import re
from typing import Dict, List, Literal, Optional, Union, Annotated, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Characters and patterns for model identifier validation
INVALID_UNICODE_CHARS = (
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space / byte order mark
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
)
MODEL_BASE_ID_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9._-]+)*$")
MODEL_TAG_REGEX = re.compile(r"^[a-zA-Z0-9._-]+$")

# ---------------------------------------------------------------------------
# Polymorphic Question Models
# ---------------------------------------------------------------------------


class ChoiceQuestion(BaseModel):
    """
    Categorical / classification question with discrete candidate options.
    Requires at least 2 distinct criteria choices mapped to label descriptions.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"]
    criteria: Dict[str, str] = Field(
        ...,
        min_length=2,
        max_length=255,
        description=(
            "Candidate labels mapped to their classification criteria (between 2 and 255 required)"
        ),
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: Dict[str, str]) -> Dict[str, str]:
        for key, desc in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Criteria label keys cannot be empty or whitespace")
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"Criteria description for '{key}' cannot be empty or whitespace")
        return v


class ScoreQuestion(BaseModel):
    """
    Continuous or ordinal score question evaluated along anchored boundaries.
    Requires at least 2 anchor string definitions in ascending order.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["score"]
    criteria: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Rating boundary definitions in ascending order (between 2 and 10 required)",
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: List[str]) -> List[str]:
        for idx, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Score anchor at index {idx} cannot be empty or whitespace")
        return v


class NoulQuestion(BaseModel):
    """
    Binary decision or probability question (yes/no, true/false).
    Criteria mapping is optional.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"]
    criteria: Optional[Dict[str, str]] = Field(
        default=None, description="Optional yes/no criteria definition"
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Criteria must be a dictionary")
            for key, desc in v.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Criteria label keys cannot be empty or whitespace")
                if not isinstance(desc, str) or not desc.strip():
                    raise ValueError(
                        f"Criteria description for '{key}' cannot be empty or whitespace"
                    )
        return v


# Discriminated union on field 'type'
QuestionItem = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")
]


# ---------------------------------------------------------------------------
# Decision Request Model
# ---------------------------------------------------------------------------

# Allowed SLA profile identifiers
SlaProfile = Literal["fast", "accurate", "cost", "balanced", "smart", "pointer"]


class DecisionRequest(BaseModel):
    """
    Decision evaluation request payload for /v1/systemone and /v1/auto.
    """

    model_config = ConfigDict(extra="ignore")

    state: str = Field(
        ...,
        min_length=1,
        max_length=1048576,  # 1MB bound to prevent host memory exhaustion
        description="Text or document state representation to evaluate (must be non-empty)",
    )
    questions: Dict[str, QuestionItem] = Field(
        ...,
        min_length=1,
        max_length=256,
        description=(
            "Dictionary of typed questions keyed by question ID (between 1 and 256 required)"
        ),
    )
    sla: Optional[SlaProfile] = Field(
        default=None,
        description=(
            "Performance SLA profile ('fast', 'accurate', 'cost', 'balanced', 'smart', 'pointer')"
        ),
    )
    max_latency_ms: Optional[int] = Field(
        default=None, gt=0, le=600000, description="Latency budget in milliseconds (must be > 0)"
    )
    model: Optional[str] = Field(
        default=None, max_length=128, description="Explicit model name override or 'auto'"
    )

    @field_validator("state")
    @classmethod
    def state_must_not_be_blank(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("State must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("State cannot be empty or contain only whitespace")
        return stripped

    @field_validator("questions")
    @classmethod
    def validate_question_ids(cls, v: Dict[str, QuestionItem]) -> Dict[str, QuestionItem]:
        for qid in v.keys():
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("Question IDs cannot be empty or contain only whitespace")
        return v

    @field_validator("sla", mode="before")
    @classmethod
    def normalize_sla(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("SLA profile must be a string, not a boolean")
        if isinstance(v, str):
            clean = v.strip().lower()
            return clean
        return v

    @field_validator("max_latency_ms", mode="before")
    @classmethod
    def validate_max_latency_ms(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("max_latency_ms must be an integer, not a boolean")
        if isinstance(v, str):
            raise ValueError("max_latency_ms must be an integer, not a string")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_override(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Model name must be a string")
        clean = v.strip()
        if not clean:
            raise ValueError("Model name cannot be empty or contain only whitespace")
        if clean.startswith(":") or clean.endswith(":") or "::" in clean:
            raise ValueError("Model name contains invalid colon placement")
        if any(ord(c) < 32 or ord(c) == 127 for c in clean):
            raise ValueError("Model name contains non-printable characters")
        if any(c in clean for c in INVALID_UNICODE_CHARS):
            raise ValueError("Model name contains invalid Unicode control characters")
        parts = clean.split(":")
        if len(parts) == 1:
            if not MODEL_BASE_ID_REGEX.match(parts[0]):
                raise ValueError("Invalid model identifier format")
        elif len(parts) == 2:
            if not MODEL_BASE_ID_REGEX.match(parts[0]):
                raise ValueError("Invalid model identifier format")
            if not MODEL_TAG_REGEX.match(parts[1]):
                raise ValueError("Invalid model tag format")
        elif len(parts) == 3:
            if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", parts[0]):
                raise ValueError("Invalid registry host format")
            if "/" not in parts[1]:
                raise ValueError("Invalid model identifier format")
            port, path = parts[1].split("/", 1)
            if not port.isdigit() or not MODEL_BASE_ID_REGEX.match(path):
                raise ValueError("Invalid registry port or model path format")
            if not MODEL_TAG_REGEX.match(parts[2]):
                raise ValueError("Invalid model tag format")
        else:
            raise ValueError("Invalid model identifier format")
        return clean


# ---------------------------------------------------------------------------
# Model Management Models
# ---------------------------------------------------------------------------


class PullModelRequest(BaseModel):
    """
    Model pull request payload for POST /v1/models/pull.
    """

    model_config = ConfigDict(extra="ignore")

    model: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Name of the model to download (e.g. 'gliclass', 'qwen3guard', 'decider')",
    )
    stream: bool = Field(
        default=False, description="Whether to stream the NDJSON download progress"
    )

    @field_validator("model")
    @classmethod
    def model_name_must_not_be_blank(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Model name must be a string")
        clean = v.strip()
        if not clean:
            raise ValueError("Model name cannot be empty or contain only whitespace")
        return clean
