"""Build the NSCH 2024 youth mental health feature matrix.

The public-use topical file does not contain columns named ``FPL`` or
``FAMILY``. Poverty is ``FPL_I1`` (first implicate of the family poverty
ratio, 50–400 percent of the federal poverty level). Family structure is
``FAMILY_R``. ``FWC`` is the final selected-child survey weight.

2024 codebook labels for the ACE items named in the project brief:

* ``ACE3`` — parent or guardian divorced or separated.
* ``ACE7`` — child was a victim of violence or witnessed neighborhood
  violence. This file has no foster-care item.
* ``ACE10`` — child was treated or judged unfairly because of race or
  ethnicity. It is not an ACE total. ``ace_count`` is derived here from
  the hardship item and ``ACE3``–``ACE11``.

Diagnosis items ``K2Q32A``, ``K2Q33A``, and ``K2Q34A`` are yes/no ever-told
flags (1 = yes, 2 = no). The follow-up "currently" items are the ``B``
variables and are not used for the outcome.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyreadstat

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "nsch_2024e_topical.sas7bdat"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "nsch_2024_youth.parquet"

SOURCE_COLUMNS: list[str] = [
    "SC_AGE_YEARS",
    "K2Q32A",
    "K2Q33A",
    "K2Q34A",
    "FPL_I1",
    "FAMILY_R",
    "ACE1",
    "ACE3",
    "ACE4",
    "ACE5",
    "ACE6",
    "ACE7",
    "ACE8",
    "ACE9",
    "ACE10",
    "ACE11",
    "FWC",
]

DIAGNOSIS_COLUMNS: list[str] = ["K2Q32A", "K2Q33A", "K2Q34A"]
BINARY_ACE_COLUMNS: list[str] = [
    "ACE3",
    "ACE4",
    "ACE5",
    "ACE6",
    "ACE7",
    "ACE8",
    "ACE9",
    "ACE10",
    "ACE11",
]

# ACE1: 1 = never, 2 = rarely, 3 = somewhat often, 4 = very often.
HARDSHIP_ADVERSE_CODES: frozenset[int] = frozenset({3, 4})

# FAMILY_R recode. Married two-parent homes are codes 1 and 3.
FAMILY_LABELS: dict[int, str] = {
    1: "two_parent_married",
    2: "two_parent_unmarried",
    3: "two_parent_married",
    4: "two_parent_unmarried",
    5: "single_parent",
    6: "single_parent",
    7: "grandparent_household",
    8: "other",
}
FAMILY_LEVELS: list[str] = [
    "two_parent_married",
    "two_parent_unmarried",
    "single_parent",
    "grandparent_household",
    "other",
    "missing",
]

OUTPUT_COLUMNS: list[str] = [
    "SC_AGE_YEARS",
    "has_mental_health_condition",
    "FPL",
    "family_structure",
    "ace_parent_divorced",
    "ace_victim_of_violence",
    "ace_treated_unfairly_race",
    "ace_count",
    "ace_items_missing",
    "FWC",
]

AGE_MIN = 12
AGE_MAX = 17
FPL_MIN = 50
FPL_MAX = 400


def load_nsch_columns(path: Path) -> pd.DataFrame:
    """Read the source columns from the NSCH topical SAS file.

    User-defined SAS missing values are returned as null. The file does not
    keep a flag that separates a legitimate skip from item nonresponse.

    Args:
        path: Path to ``nsch_2024e_topical.sas7bdat``.

    Returns:
        Data frame containing only the source columns used by this extract.
    """
    frame, _metadata = pyreadstat.read_sas7bdat(
        path,
        usecols=SOURCE_COLUMNS,
        user_missing=False,
    )
    missing = [name for name in SOURCE_COLUMNS if name not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"NSCH file is missing expected columns: {joined}")
    return frame.loc[:, SOURCE_COLUMNS]


def _integer_codes(series: pd.Series) -> pd.Series:
    """Coerce a survey item to a nullable integer code."""
    numeric = pd.to_numeric(series, errors="coerce")
    observed = numeric.dropna()
    if not observed.empty:
        fractional = (observed - observed.round()).abs()
        if bool((fractional > 1e-6).any()):
            raise ValueError(f"Column {series.name} has non-integer codes")
    return numeric.round().astype("Int8")


def _recode_yes_no(series: pd.Series) -> pd.Series:
    """Map NSCH 1 = yes and 2 = no onto 1 and 0. Other codes become null."""
    codes = _integer_codes(series)
    return codes.map({1: 1, 2: 0}).astype("Int8")


def _diagnosis_flag(frame: pd.DataFrame) -> pd.Series:
    """Flag a youth with depression, anxiety, or behavior problems.

    The flag is 1 when any diagnosis item is yes, and 0 when every item is
    no. It stays null when no item is yes and at least one item is missing.
    """
    diagnoses = pd.concat(
        [_recode_yes_no(frame[column]) for column in DIAGNOSIS_COLUMNS],
        axis=1,
    )
    any_yes = diagnoses.eq(1).fillna(False).any(axis=1)
    all_no = diagnoses.notna().all(axis=1) & diagnoses.eq(0).fillna(False).all(axis=1)
    flag = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    flag = flag.mask(any_yes, 1)
    return flag.mask(all_no, 0)


def _family_structure(series: pd.Series) -> pd.Series:
    """Collapse ``FAMILY_R`` and keep an explicit missing category."""
    codes = _integer_codes(series)
    labels = codes.map(FAMILY_LABELS).astype("string").fillna("missing")
    return pd.Series(
        pd.Categorical(labels, categories=FAMILY_LEVELS),
        index=series.index,
        name="family_structure",
    )


def _ace_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the 10 binary ACE indicators used for ``ace_count``.

    Household hardship counts as an adversity when basics were hard to cover
    somewhat often or very often. Each other ACE item counts when the answer
    is yes.
    """
    hardship_codes = _integer_codes(frame["ACE1"])
    hardship = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    hardship = hardship.mask(hardship_codes.isin(HARDSHIP_ADVERSE_CODES), 1)
    hardship = hardship.mask(hardship_codes.isin({1, 2}), 0)

    binary = pd.DataFrame(
        {column: _recode_yes_no(frame[column]) for column in BINARY_ACE_COLUMNS},
        index=frame.index,
    )
    binary.insert(0, "ACE1_HARDSHIP", hardship)
    return binary


def build_feature_matrix(source: pd.DataFrame) -> pd.DataFrame:
    """Filter to ages 12–17 and convert survey codes into model features.

    Rows whose diagnosis outcome cannot be determined are dropped. Predictor
    gaps stay null, except family structure, which uses a ``missing`` level.
    ``ace_items_missing`` records how many of the 10 ACE indicators were
    unanswered. ``ace_count`` sums the answered adversities and is null only
    when every ACE indicator is missing.

    Args:
        source: Columns returned by :func:`load_nsch_columns`.

    Returns:
        Youth feature matrix with one row per retained respondent.
    """
    age = _integer_codes(source["SC_AGE_YEARS"])
    youth = source.loc[age.between(AGE_MIN, AGE_MAX)].copy()
    youth_age = age.loc[youth.index]

    outcome = _diagnosis_flag(youth)
    known_outcome = outcome.notna()
    dropped = int((~known_outcome).sum())
    LOGGER.info("Dropped %s ages 12-17 rows with an unknown diagnosis outcome", dropped)
    youth = youth.loc[known_outcome]
    youth_age = youth_age.loc[known_outcome]
    outcome = outcome.loc[known_outcome]

    fpl = pd.to_numeric(youth["FPL_I1"], errors="coerce")
    fpl = fpl.where(fpl.between(FPL_MIN, FPL_MAX)).round().astype("Int16")

    weight = pd.to_numeric(youth["FWC"], errors="coerce")
    weight = weight.where(weight.gt(0))

    ace = _ace_components(youth)
    features = pd.DataFrame(
        {
            "SC_AGE_YEARS": youth_age.astype("Int8"),
            "has_mental_health_condition": outcome.astype("Int8"),
            "FPL": fpl,
            "family_structure": _family_structure(youth["FAMILY_R"]),
            "ace_parent_divorced": ace["ACE3"],
            "ace_victim_of_violence": ace["ACE7"],
            "ace_treated_unfairly_race": ace["ACE10"],
            "ace_count": ace.sum(axis=1, min_count=1).astype("Int8"),
            "ace_items_missing": ace.isna().sum(axis=1).astype("Int8"),
            "FWC": weight,
        },
        index=youth.index,
    )
    return features.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def save_extract(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Write a snappy-compressed parquet file and a CSV with the same stem.

    Args:
        frame: Feature matrix to save.
        parquet_path: Parquet destination. The CSV is written beside it.

    Returns:
        Path of the CSV file.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")
    frame.to_parquet(parquet_path, index=False, compression="snappy")
    frame.to_csv(csv_path, index=False)
    return csv_path


def ingest(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Load the NSCH topical file and save the youth feature matrix.

    Args:
        input_path: SAS topical file. Defaults to ``data/raw/nsch_2024e_topical.sas7bdat``.
        output_path: Parquet destination. Defaults to ``data/processed/nsch_2024_youth.parquet``.

    Returns:
        The feature matrix that was written.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"NSCH SAS file not found: {input_path}")

    LOGGER.info("Reading %s", input_path)
    source = load_nsch_columns(input_path)
    features = build_feature_matrix(source)
    csv_path = save_extract(features, output_path)

    outcome_rate = float(features["has_mental_health_condition"].mean())
    LOGGER.info("Wrote %s rows to %s", len(features), output_path)
    LOGGER.info("Wrote %s rows to %s", len(features), csv_path)
    LOGGER.info("Outcome prevalence: %.3f", outcome_rate)
    missing_counts = features.isna().sum()
    for column in OUTPUT_COLUMNS:
        LOGGER.info("Missing %s: %s", column, int(missing_counts[column]))
    return features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the NSCH 2024 youth mental health feature matrix.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the sas7bdat file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Parquet path. A CSV with the same name is written beside it.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    ingest(args.input, args.output)


if __name__ == "__main__":
    main()
