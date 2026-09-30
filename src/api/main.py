"""FastAPI service for youth mental health risk scores and SHAP values."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.predict import predict_and_explain

app = FastAPI(title="Youth mental health risk", version="1.0.0")

FamilyStructure = Literal[
    "two_parent_married",
    "two_parent_unmarried",
    "single_parent",
    "grandparent_household",
    "other",
    "missing",
]


class FeatureInput(BaseModel):
    """Seven predictors from the youth feature matrix."""

    model_config = ConfigDict(extra="forbid")

    SC_AGE_YEARS: int = Field(ge=12, le=17)
    FPL: int = Field(ge=50, le=400)
    family_structure: FamilyStructure
    ace_parent_divorced: int | None = None
    ace_victim_of_violence: int | None = None
    ace_treated_unfairly_race: int | None = None
    ace_items_missing: int = Field(ge=0, le=10)

    @field_validator(
        "ace_parent_divorced",
        "ace_victim_of_violence",
        "ace_treated_unfairly_race",
    )
    @classmethod
    def ace_answer(cls, value: int | None) -> int | None:
        """Accept only yes, no, or an unanswered item."""
        if value is not None and value not in (0, 1):
            raise ValueError("ACE fields must be 0, 1, or null")
        return value


class PredictResponse(BaseModel):
    """Probability that the youth has a diagnosed mental health condition."""

    probability: float


class ShapValue(BaseModel):
    """One feature's value and its SHAP contribution."""

    feature: str
    value: int | float | str | None
    shap_value: float


class ExplainResponse(BaseModel):
    """Probability plus per-feature SHAP values."""

    probability: float
    shap_values: list[ShapValue]


def _score(payload: FeatureInput) -> ExplainResponse:
    try:
        result = predict_and_explain(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    body = result.as_dict()
    return ExplainResponse.model_validate(body)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: FeatureInput) -> PredictResponse:
    """Return the probability of a diagnosed mental health condition."""
    scored = _score(payload)
    return PredictResponse(probability=scored.probability)


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: FeatureInput) -> ExplainResponse:
    """Return the probability and SHAP values for the submitted youth."""
    return _score(payload)


def main() -> None:
    """Run the API on localhost:8000."""
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
