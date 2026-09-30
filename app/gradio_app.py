"""Two-column Gradio dashboard for a youth mental health risk score."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from src.models.predict import PredictionExplanation, ShapContribution, predict_and_explain
from src.models.schema import FAMILY_LEVELS

ACE_CHOICES = ["Yes", "No", "Missing"]
ACE_CODES = {"Yes": 1, "No": 0, "Missing": None}

FEATURE_LABELS = {
    "SC_AGE_YEARS": "Age",
    "FPL": "Poverty level",
    "family_structure": "Family",
    "ace_parent_divorced": "Divorced parent",
    "ace_victim_of_violence": "Neighborhood violence",
    "ace_treated_unfairly_race": "Discrimination",
    "ace_items_missing": "Unanswered ACEs",
}

FAMILY_LABELS = {
    "two_parent_married": "Two parents, married",
    "two_parent_unmarried": "Two parents, unmarried",
    "single_parent": "Single parent",
    "grandparent_household": "Grandparent household",
    "other": "Other household",
    "missing": "Missing",
}

def _risk_tier(probability: float) -> tuple[str, str, str, str]:
    """Map a probability to a tier name, text color, fill color, and background.

    The cut points use the same one-decimal percent shown on the card, so a
    value displayed as 20.0% is moderate and 50.0% is still moderate.
    """
    shown = round(100 * probability, 1)
    if shown < 20:
        return "Low risk", "#166534", "#16a34a", "#dcfce7"
    if shown <= 50:
        return "Moderate risk", "#92400e", "#d97706", "#fef3c7"
    return "High risk", "#9f1239", "#e11d48", "#ffe4e6"


def risk_card(probability: float) -> str:
    """Build the color-coded risk tier card.

    Args:
        probability: Predicted probability of a diagnosed condition.

    Returns:
        HTML for a low, moderate, or high risk badge with a meter.
    """
    tier, text, fill, background = _risk_tier(probability)
    percent = 100 * probability
    return (
        f"<div style=\"border:1px solid {fill};background:{background};color:{text};"
        "border-radius:16px;padding:22px 24px;\">"
        f"<div style=\"font-size:13px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;\">{tier}</div>"
        f"<div style=\"font-size:52px;font-weight:700;line-height:1.05;margin:4px 0 12px;\">{percent:.1f}%</div>"
        "<div style=\"height:12px;border-radius:999px;background:rgba(0,0,0,0.08);overflow:hidden;\">"
        f"<div style=\"width:{percent:.1f}%;height:12px;background:{fill};\"></div></div>"
        "<div style=\"margin-top:12px;font-size:14px;line-height:1.4;\">"
        "Probability of a depression, anxiety, or behavior-problem diagnosis.</div></div>"
    )


def _empty_card() -> str:
    return (
        "<div style=\"border:1px solid #d1d5db;background:#f3f4f6;color:#374151;"
        "border-radius:16px;padding:22px 24px;\">"
        "<div style=\"font-size:13px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;\">Risk tier</div>"
        "<div style=\"font-size:52px;font-weight:700;line-height:1.05;margin:4px 0 12px;\">—</div>"
        "<div style=\"font-size:14px;line-height:1.4;\">Enter this youth's circumstances, then choose Score.</div>"
        "</div>"
    )


def _caption(item: ShapContribution) -> str:
    if item.feature == "family_structure":
        shown = FAMILY_LABELS.get(str(item.value), str(item.value))
    elif item.feature in {"ace_parent_divorced", "ace_victim_of_violence", "ace_treated_unfairly_race"}:
        shown = {1: "Yes", 0: "No"}.get(item.value, "Missing")
    elif item.feature == "FPL":
        shown = f"{item.value}%"
    elif item.value is None:
        shown = "Missing"
    else:
        shown = str(item.value)
    return f"{FEATURE_LABELS[item.feature]} = {shown}"


def _chance(log_odds: float) -> float:
    """Convert a log-odds score to a probability."""
    return 1.0 / (1.0 + math.exp(-log_odds))


def _percentage_steps(result: PredictionExplanation) -> list[tuple[ShapContribution, float]]:
    """Turn SHAP values into percentage-point changes, smallest effect first."""
    ordered = sorted(result.shap_values, key=lambda item: abs(item.shap_value))
    log_odds = result.base_value
    chance = _chance(log_odds)
    steps: list[tuple[ShapContribution, float]] = []
    for item in ordered:
        updated = _chance(log_odds + item.shap_value)
        steps.append((item, 100 * (updated - chance)))
        log_odds += item.shap_value
        chance = updated
    return steps


def shap_waterfall(result: PredictionExplanation) -> plt.Figure:
    """Draw a SHAP waterfall in percentage points.

    Each bar is the change in the chance of a diagnosis after that answer.
    The dashed line is the average youth. Blue lowers the chance. Red raises
    it. The last step matches the risk card.

    Args:
        result: Probability, base log-odds, and per-feature SHAP values.

    Returns:
        A matplotlib figure for ``gr.Plot``.
    """
    steps = _percentage_steps(result)
    labels = [_caption(item) for item, _delta in steps]
    starts: list[float] = []
    deltas: list[float] = []
    chance = 100 * _chance(result.base_value)
    for _item, delta in steps:
        starts.append(chance)
        deltas.append(delta)
        chance += delta

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for index, delta in enumerate(deltas):
        color = "#e11d48" if delta > 0 else "#2563eb"
        axis.barh(index, delta, left=starts[index], color=color, height=0.62, zorder=2)
        if index < len(deltas) - 1:
            edge = starts[index] + delta
            axis.plot([edge, edge], [index + 0.31, index + 0.69], color="#9ca3af", linewidth=0.8, zorder=1)
    axis.axvline(100 * _chance(result.base_value), color="#6b7280", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.0f}%"))
    axis.set_xlabel("Chance of a diagnosis")
    axis.set_title("Blue lowers the chance. Red raises it.")
    figure.subplots_adjust(left=0.42, right=0.98, top=0.88, bottom=0.16)
    return figure


MODEL_PATTERNS = """### Patterns in this model
- Neighborhood violence raises the chance, often by a large amount.
- A parent or guardian who divorced or separated raises the chance.
- Unfair treatment because of race or ethnicity raises the chance.
- A grandparent household has a higher chance than two married parents.
- Age and income do not move in one direction. Lower income does not always mean a higher chance, including for older youth.
"""


def score_youth(
    age: float,
    poverty_ratio: float,
    family_structure: str,
    parent_divorced: str,
    victim_of_violence: str,
    treated_unfairly_race: str,
    ace_items_missing: float,
) -> tuple[str, plt.Figure]:
    """Score one form submission and draw its SHAP waterfall.

    Args:
        age: Selected child's age in years.
        poverty_ratio: Family income as a percent of the federal poverty level.
        family_structure: NSCH household type code.
        parent_divorced: Yes, no, or missing for parent or guardian divorce.
        victim_of_violence: Yes, no, or missing for violence exposure.
        treated_unfairly_race: Yes, no, or missing for unfair treatment because of race.
        ace_items_missing: How many of the 10 ACE indicators were unanswered.

    Returns:
        A risk-tier card and a matplotlib SHAP waterfall.
    """
    result = predict_and_explain(
        {
            "SC_AGE_YEARS": int(age),
            "FPL": int(poverty_ratio),
            "family_structure": family_structure,
            "ace_parent_divorced": ACE_CODES[parent_divorced],
            "ace_victim_of_violence": ACE_CODES[victim_of_violence],
            "ace_treated_unfairly_race": ACE_CODES[treated_unfairly_race],
            "ace_items_missing": int(ace_items_missing),
        }
    )
    return risk_card(result.probability), shap_waterfall(result)


def build_demo() -> gr.Blocks:
    """Build the two-column scoring dashboard."""
    family_choices = [(FAMILY_LABELS[level], level) for level in FAMILY_LEVELS]
    with gr.Blocks(title="Youth mental health risk") as demo:
        gr.Markdown(
            "## Youth mental health risk\n"
            "Estimate the probability of a depression, anxiety, or behavior-problem "
            "diagnosis for a youth age 12–17. The waterfall starts at the average "
            "youth's chance and shows how each answer moves that chance."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Demographics and poverty", open=True):
                    age = gr.Slider(12, 17, value=15, step=1, label="Age (years)")
                    poverty = gr.Slider(
                        50,
                        400,
                        value=200,
                        step=1,
                        label="Family poverty level (percent of the federal poverty level)",
                    )
                with gr.Accordion("Family structure", open=True):
                    family = gr.Dropdown(
                        choices=family_choices,
                        value="two_parent_married",
                        label="Household type",
                    )
                with gr.Accordion("Adverse childhood experiences", open=True):
                    divorced = gr.Radio(
                        ACE_CHOICES,
                        value="No",
                        label="Parent or guardian divorced or separated",
                    )
                    violence = gr.Radio(
                        ACE_CHOICES,
                        value="No",
                        label="Victim of violence or witnessed neighborhood violence",
                    )
                    race = gr.Radio(
                        ACE_CHOICES,
                        value="No",
                        label="Treated unfairly because of race or ethnicity",
                    )
                    unanswered = gr.Slider(0, 10, value=0, step=1, label="Unanswered ACE items")
                score = gr.Button("Score", variant="primary")
            with gr.Column(scale=1):
                card = gr.HTML(value=_empty_card())
                plot = gr.Plot(label="SHAP waterfall")
                gr.Markdown(MODEL_PATTERNS)
        score.click(
            score_youth,
            inputs=[age, poverty, family, divorced, violence, race, unanswered],
            outputs=[card, plot],
        )
        demo.load(
            score_youth,
            inputs=[age, poverty, family, divorced, violence, race, unanswered],
            outputs=[card, plot],
        )
    return demo


def main() -> None:
    """Launch the Gradio app."""
    build_demo().launch()


if __name__ == "__main__":
    main()
