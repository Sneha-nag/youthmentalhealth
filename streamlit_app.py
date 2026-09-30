"""Streamlit scoring page for Streamlit Community Cloud."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app.scoring_view import (
    ACE_CHOICES,
    FAMILY_LABELS,
    MODEL_PATTERNS,
    score_youth,
)
from src.models.schema import FAMILY_LEVELS

st.set_page_config(page_title="Youth mental health risk", layout="wide")
st.markdown(
    """
## Youth mental health risk
Estimate the probability of a depression, anxiety, or behavior-problem
diagnosis for a youth age 12–17, using the 2024 National Survey of
Children's Health. This is a survey model, not a clinical assessment.
The waterfall starts at the average youth's chance and shows how each
answer moves that chance.
"""
)

left, right = st.columns(2, gap="large")

with left:
    with st.expander("Demographics and poverty", expanded=True):
        age = st.slider("Age (years)", min_value=12, max_value=17, value=15, step=1)
        poverty = st.slider(
            "Family poverty level (percent of the federal poverty level)",
            min_value=50,
            max_value=400,
            value=200,
            step=1,
        )
    with st.expander("Family structure", expanded=True):
        family = st.selectbox(
            "Household type",
            options=FAMILY_LEVELS,
            index=FAMILY_LEVELS.index("two_parent_married"),
            format_func=lambda level: FAMILY_LABELS[level],
        )
    with st.expander("Adverse childhood experiences", expanded=True):
        divorced = st.radio(
            "Parent or guardian divorced or separated",
            ACE_CHOICES,
            index=1,
            horizontal=True,
        )
        violence = st.radio(
            "Victim of violence or witnessed neighborhood violence",
            ACE_CHOICES,
            index=1,
            horizontal=True,
        )
        race = st.radio(
            "Treated unfairly because of race or ethnicity",
            ACE_CHOICES,
            index=1,
            horizontal=True,
        )
        unanswered = st.slider("Unanswered ACE items", min_value=0, max_value=10, value=0, step=1)

card_html, figure = score_youth(age, poverty, family, divorced, violence, race, unanswered)

with right:
    st.markdown(card_html, unsafe_allow_html=True)
    st.pyplot(figure, clear_figure=True)
    st.markdown(MODEL_PATTERNS)

plt.close("all")
