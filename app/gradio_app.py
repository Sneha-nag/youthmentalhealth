"""Two-column Gradio dashboard for a youth mental health risk score."""

from __future__ import annotations

import gradio as gr

from app.scoring_view import (
    ACE_CHOICES,
    FAMILY_LABELS,
    MODEL_PATTERNS,
    score_youth,
)
from src.models.schema import FAMILY_LEVELS


def build_demo() -> gr.Blocks:
    """Build the two-column scoring dashboard."""
    family_choices = [(FAMILY_LABELS[level], level) for level in FAMILY_LEVELS]
    with gr.Blocks(title="Youth mental health risk") as demo:
        gr.Markdown(
            "## Youth mental health risk\n"
            "Estimate the probability of a depression, anxiety, or behavior-problem "
            "diagnosis for a youth age 12–17, using the 2024 National Survey of "
            "Children's Health. This is a survey model, not a clinical assessment. "
            "The waterfall starts at the average youth's chance and shows how each "
            "answer moves that chance."
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
