"""
Baseline tests for src.train_model.

These tests focus on the critical behaviour of the training pipeline without
running expensive Optuna searches or repeatedly training large XGBoost models.

The suite verifies:

    - dataset validation and cleaning
    - chronological train/holdout splitting
    - route-median target construction without leakage
    - feature schema validation
    - preprocessing behaviour
    - evaluation metrics
    - carrier eligibility
    - chronological CV policy
    - one lightweight XGBoost training/save smoke test

Run with:

    pytest -q

For more verbose output:

    pytest -v
"""

import os

import joblib
import numpy as np
import pandas as pd
import pytest

import src.train_model as train_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_df():

    """
    Creates a small deterministic feature dataframe containing all columns
    required by the training pipeline.

    The fixture is intentionally small so that tests remain fast and do not
    depend on the real production feature dataset.
    """
    n = 20

    dates = pd.date_range(
        "2022-01-01",
        periods=n,
        freq="90D",
        tz="UTC",
    )

    df = pd.DataFrame({
        "icao24": [f"abc{i:04d}" for i in range(n)],
        "firstSeen": dates,
        "lastSeen": dates,
        "route": ["AAA->BBB"] * n,
        "airline": ["TEST"] * n,
        "flight_duration": np.linspace(90, 120, n),
        "hour_of_day": [8, 10, 12, 14, 16] * 4,
        "month": dates.month,
        "flight_date_parsed": dates,
    })

    # Populate every expected weather feature with deterministic values.
    # The actual values are not important; the tests only need valid columns.
    for column in (
        train_model.DEPARTURE_WEATHER_COLUMNS
        + train_model.ARRIVAL_WEATHER_COLUMNS
    ):
        df[column] = np.linspace(1, 20, n)

    return df


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def test_prepare_dataset_sorts_by_first_seen(feature_df):

    """
    Ensure the feature dataframe is returned in chronological order.

    Chronological ordering is critical because the final holdout and all
    time-series validation depend on row order representing time.
    """
    # Shuffle the input to ensure prepare_dataset() performs the ordering.
    shuffled = feature_df.sample(frac=1, random_state=42)

    result = train_model.prepare_dataset(shuffled)

    assert result["firstSeen"].is_monotonic_increasing


def test_prepare_dataset_removes_duration_outliers(feature_df):

    """
    Ensure flights exceeding the configured duration threshold are removed.

    Extremely long durations are treated as invalid/outlier observations by
    the production training pipeline.
    """
    df = feature_df.copy()

    # Create one duration that is just above the configured threshold.
    df.loc[0, "flight_duration"] = (
        train_model.OUTLIER_DURATION_MINUTES + 1
    )

    result = train_model.prepare_dataset(df)

    assert len(result) == len(df) - 1
    assert (
        result["flight_duration"]
        <= train_model.OUTLIER_DURATION_MINUTES
    ).all()


def test_prepare_dataset_keeps_duration_at_threshold(feature_df):

    """
    Ensure a flight exactly at the configured duration threshold is retained.

    The production rule is strictly greater than the threshold, not greater
    than or equal to it.
    """
    df = feature_df.copy()

    # Set one observation exactly at the allowed boundary.
    df.loc[0, "flight_duration"] = (
        train_model.OUTLIER_DURATION_MINUTES
    )

    result = train_model.prepare_dataset(df)

    assert len(result) == len(df)


def test_prepare_dataset_rejects_non_positive_duration(feature_df):

    """
    Ensure zero or negative flight durations are rejected.

    A non-positive duration is invalid data and should never reach model
    training.
    """
    df = feature_df.copy()

    # Replace one valid duration with an impossible value.
    df.loc[0, "flight_duration"] = 0

    with pytest.raises(
        ValueError,
        match="zero/negative",
    ):
        train_model.prepare_dataset(df)


def test_prepare_dataset_rejects_non_finite_duration(feature_df):

    """
    Ensure NaN or other non-finite flight durations are rejected.

    Non-finite target values would make model training and evaluation invalid.
    """
    df = feature_df.copy()

    # Introduce an invalid target value.
    df.loc[0, "flight_duration"] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        train_model.prepare_dataset(df)


def test_prepare_dataset_rejects_missing_required_column(feature_df):

    """
    Ensure dataset preparation fails clearly when a required feature column
    is missing.
    """
    # Remove a required column to simulate a malformed feature dataset.
    df = feature_df.drop(columns=["airline"])

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        train_model.prepare_dataset(df)


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def test_chronological_split_is_80_20(feature_df):

    """
    Ensure the final holdout contains the configured fraction of observations.

    The production pipeline uses the final 20% of chronologically ordered
    observations as its untouched holdout set.
    """
    train, test = train_model.chronological_split(feature_df)

    expected_train_size = int(
        len(feature_df) * (1 - train_model.TEST_FRACTION)
    )

    assert len(train) == expected_train_size
    assert len(train) + len(test) == len(feature_df)


def test_chronological_split_has_no_temporal_overlap(feature_df):

    """
    Ensure the training period ends before the holdout period begins.

    This protects against accidentally creating a random or overlapping
    train/test split.
    """
    train, test = train_model.chronological_split(feature_df)

    assert train["firstSeen"].max() < test["firstSeen"].min()


def test_chronological_split_preserves_order(feature_df):

    """
    Ensure both resulting datasets remain chronologically ordered.
    """
    train, test = train_model.chronological_split(feature_df)

    assert train["firstSeen"].is_monotonic_increasing
    assert test["firstSeen"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Target construction / leakage
# ---------------------------------------------------------------------------

def test_target_uses_training_median_only():

    """
    Ensure the residual target uses the median calculated from the training
    data and applies that same median to the validation data.

    This is one of the most important leakage protections in the pipeline.
    """
    train = pd.DataFrame({
        "flight_duration": [100, 110, 120],
    })

    validation = pd.DataFrame({
        "flight_duration": [1000, 2000, 3000],
    })

    y_train, y_validation, route_median = (
        train_model.target_from_training(
            train,
            validation,
        )
    )

    # The training median is 110 and must be used for both datasets.
    assert route_median == 110

    assert y_train.tolist() == [-10, 0, 10]

    assert y_validation.tolist() == [
        890,
        1890,
        2890,
    ]


def test_target_does_not_use_validation_median():

    """
    Ensure extreme validation durations cannot influence the route median.

    If the validation period were used to calculate the median, this test
    would produce a different residual target.
    """
    train = pd.DataFrame({
        "flight_duration": [100, 100, 100],
    })

    validation = pd.DataFrame({
        "flight_duration": [1000, 1000, 1000],
    })

    _, y_validation, route_median = (
        train_model.target_from_training(
            train,
            validation,
        )
    )

    # The validation median must have no effect on the learned median.
    assert route_median == 100
    assert (y_validation == 900).all()


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def test_candidate_feature_schema_is_valid(feature_df):

    """
    Ensure a correctly constructed feature dataframe passes candidate
    feature schema validation.
    """
    # No exception should be raised for a valid feature dataframe.
    train_model.validate_candidate_features(feature_df)


def test_candidate_feature_schema_rejects_missing_feature(feature_df):

    """
    Ensure feature validation fails when an expected numerical model feature
    is missing from the dataframe.
    """
    # Remove one expected model feature to simulate a broken feature pipeline.
    df = feature_df.drop(
        columns=[train_model.EXPECTED_NUMERICAL_MODEL_FEATURES[0]]
    )

    with pytest.raises(
        ValueError,
        match="Candidate feature schema validation failed",
    ):
        train_model.validate_candidate_features(df)


def test_model_schema_accepts_pooled_features():

    """
    Ensure the pooled model accepts exactly the expected numerical features
    plus the encoded airline feature.
    """
    X = pd.DataFrame(
        np.zeros(
            (2, len(train_model.EXPECTED_POOLED_MODEL_FEATURES))
        ),
        columns=train_model.EXPECTED_POOLED_MODEL_FEATURES,
    )

    # A correctly ordered pooled matrix should pass validation.
    train_model.validate_model_features(
        X,
        pooled=True,
    )


def test_model_schema_accepts_carrier_features():

    """
    Ensure a dedicated carrier model accepts exactly the expected numerical
    feature schema.
    """
    X = pd.DataFrame(
        np.zeros(
            (2, len(train_model.EXPECTED_NUMERICAL_MODEL_FEATURES))
        ),
        columns=train_model.EXPECTED_NUMERICAL_MODEL_FEATURES,
    )

    # A correctly ordered dedicated-carrier matrix should pass validation.
    train_model.validate_model_features(
        X,
        pooled=False,
    )


def test_model_schema_rejects_wrong_order():

    """
    Ensure model feature validation rejects the correct columns when they
    appear in the wrong order.

    Column order matters because the trained model expects a fixed schema.
    """
    columns = list(
        train_model.EXPECTED_NUMERICAL_MODEL_FEATURES
    )

    # Reverse the feature order to simulate an incorrectly constructed matrix.
    columns.reverse()

    X = pd.DataFrame(
        np.zeros((2, len(columns))),
        columns=columns,
    )

    with pytest.raises(
        ValueError,
        match="feature schema validation failed",
    ):
        train_model.validate_model_features(
            X,
            pooled=False,
        )


# ---------------------------------------------------------------------------
# Leakage columns
# ---------------------------------------------------------------------------

def test_remove_fold_leakage_columns():

    """
    Ensure target-construction and operational columns are removed before
    feature preprocessing/model training.
    """
    df = pd.DataFrame({
        "flight_duration": [100],
        "route": ["AAA->BBB"],
        "flight_date_parsed": ["2022-01-01"],
        "median_flight_time": [100],
        "hour_of_day": [12],
    })

    result = train_model.remove_fold_leakage_columns(df)

    # These columns are needed for target construction but must not reach XGBoost.
    assert "flight_duration" not in result
    assert "route" not in result
    assert "flight_date_parsed" not in result
    assert "median_flight_time" not in result

    # Genuine model features must remain available.
    assert "hour_of_day" in result


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocessor_handles_missing_values(feature_df):

    """
    Ensure numerical missing values are handled by the training-fitted
    median imputer.
    """
    df = feature_df.copy()

    # Introduce missing values into model features.
    df.loc[0, "dep_temperature_2m"] = np.nan
    df.loc[1, "arr_temperature_2m"] = np.nan

    y = (
        df["flight_duration"]
        - df["flight_duration"].median()
    )

    clean = train_model.remove_fold_leakage_columns(df)

    preprocessing, imputer = train_model.fit_preprocessor(
        clean,
        y,
        pooled=False,
        model_feature_columns=(
            train_model.EXPECTED_NUMERICAL_MODEL_FEATURES
        ),
        quantile=0.5,
    )

    X = train_model.transform_features(
        clean,
        preprocessing,
        imputer,
    )

    # After preprocessing there should be no missing model values.
    assert not X.isna().any().any()


def test_pooled_preprocessor_creates_airline_encoding(feature_df):

    """
    Ensure the pooled model creates a training-data airline target encoding.

    The encoding is fitted from the training population and persisted as part
    of the preprocessing metadata.
    """
    df = feature_df.copy()

    df["airline"] = [
        "AAA" if i % 2 == 0 else "BBB"
        for i in range(len(df))
    ]

    y = (
        df["flight_duration"]
        - df["flight_duration"].median()
    )

    clean = train_model.remove_fold_leakage_columns(df)

    preprocessing, _ = train_model.fit_preprocessor(
        clean,
        y,
        pooled=True,
        model_feature_columns=(
            train_model.EXPECTED_NUMERICAL_MODEL_FEATURES
        ),
        quantile=0.9,
    )

    assert preprocessing["pooled"] is True
    assert "airline_encoding" in preprocessing
    assert "mapping" in preprocessing["airline_encoding"]
    assert "fallback" in preprocessing["airline_encoding"]


def test_unseen_airline_uses_training_fallback(feature_df):

    """
    Ensure an airline not seen during training receives the configured
    fallback encoding rather than causing a missing value.
    """
    train = feature_df.iloc[:10].copy()
    validation = feature_df.iloc[10:].copy()

    train["airline"] = "AAA"
    validation["airline"] = "UNSEEN"

    y_train = (
        train["flight_duration"]
        - train["flight_duration"].median()
    )

    train_clean = train_model.remove_fold_leakage_columns(train)
    valid_clean = train_model.remove_fold_leakage_columns(validation)

    preprocessing, imputer = train_model.fit_preprocessor(
        train_clean,
        y_train,
        pooled=True,
        model_feature_columns=(
            train_model.EXPECTED_NUMERICAL_MODEL_FEATURES
        ),
        quantile=0.9,
    )

    X_valid = train_model.transform_features(
        valid_clean,
        preprocessing,
        imputer,
    )

    airline_index = preprocessing["feature_columns"].index(
        "airline"
    )

    fallback = preprocessing["airline_encoding"]["fallback"]

    # Unknown airlines should use the training-derived fallback value.
    assert (
        X_valid.iloc[:, airline_index] == fallback
    ).all()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_pinball_loss_is_zero_for_perfect_prediction():

    """
    Ensure pinball loss is zero when predictions exactly match observations.
    """
    y = np.array([1, 2, 3])

    assert train_model.pinball_loss(
        y,
        y,
        0.5,
    ) == pytest.approx(0)


def test_coverage_error_is_zero_when_coverage_matches_target():

    """
    Ensure coverage error is zero when empirical coverage equals the target
    quantile coverage.
    """
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])

    # Every observation is below its prediction, giving 100% coverage.
    assert train_model.coverage_error(
        y_true,
        y_pred,
        1.0,
    ) == pytest.approx(0)


def test_calculate_metrics_returns_expected_fields():

    """
    Ensure the evaluation function returns all expected diagnostic metrics.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    metrics = train_model.calculate_metrics(
        y_true,
        y_pred,
        0.9,
    )

    expected = {
        "pinball_loss",
        "coverage_error",
        "coverage_penalty",
        "tuning_score",
        "mae",
        "medae",
        "rmse",
        "mean_error",
        "empirical_coverage_pct",
        "target_coverage_pct",
        "coverage_gap_pct",
    }

    assert expected.issubset(metrics.keys())


def test_calculate_metrics_reports_target_coverage():

    """
    Ensure the requested quantile is correctly reported as the target
    empirical coverage percentage.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    metrics = train_model.calculate_metrics(
        y_true,
        y_pred,
        0.9,
    )

    assert metrics["target_coverage_pct"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Carrier eligibility
# ---------------------------------------------------------------------------

def test_eligible_carriers_respects_minimum_rows(feature_df, monkeypatch,):

    """
    Ensure only carriers meeting the configured historical-row threshold
    are considered eligible for dedicated models.
    """
    df = feature_df.copy()

    df["airline"] = (
        ["AAA"] * 10
        + ["BBB"] * 6
        + ["CCC"] * 4
    )

    # Use a small threshold for this synthetic test.
    monkeypatch.setattr(
        train_model,
        "MIN_CARRIER_ROWS",
        6,
    )

    result = train_model.eligible_carriers(df)

    assert result == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# CV policy
# ---------------------------------------------------------------------------

def test_determine_cv_folds_requires_two_complete_years():

    """
    Ensure chronological tuning is rejected when the training population
    does not contain the minimum number of complete calendar years.
    """
    dates = pd.date_range(
        "2024-01-01",
        periods=12,
        freq="MS",
        tz="UTC",
    )

    df = pd.DataFrame({
        "firstSeen": dates,
    })

    with pytest.raises(
        ValueError,
        match="At least",
    ):
        train_model.determine_cv_folds(df)


def test_determine_cv_folds_counts_complete_years():

    """
    Ensure the CV policy correctly counts complete calendar years.
    """
    dates = []

    for year in [2021, 2022]:
        dates.extend(
            pd.date_range(
                f"{year}-01-01",
                f"{year}-12-01",
                freq="MS",
                tz="UTC",
            )
        )

    df = pd.DataFrame({
        "firstSeen": dates,
    })

    n_folds, years = train_model.determine_cv_folds(df)

    assert n_folds == 2
    assert years == [2021, 2022]


# ---------------------------------------------------------------------------
# XGBoost smoke test
# ---------------------------------------------------------------------------

def test_xgboost_model_can_train_and_save(feature_df, tmp_path,):
    
    """
    Ensure the complete basic model path works with a tiny XGBoost model.

    This is intentionally the only test that performs real XGBoost training.
    It verifies that preprocessing, feature validation, model fitting,
    prediction, and serialization all work together.
    """
    df = feature_df.copy()

    # Build the same residual target used by production training.
    y = (
        df["flight_duration"]
        - df["flight_duration"].median()
    )

    clean = train_model.remove_fold_leakage_columns(df)

    preprocessing, imputer = train_model.fit_preprocessor(
        clean,
        y,
        pooled=False,
        model_feature_columns=(
            train_model.EXPECTED_NUMERICAL_MODEL_FEATURES
        ),
        quantile=0.5,
    )

    X = train_model.transform_features(
        clean,
        preprocessing,
        imputer,
    )

    # Confirm that preprocessing produced exactly the schema expected by XGBoost.
    train_model.validate_model_features(
        X,
        pooled=False,
    )

    # Use only two trees so the smoke test remains very fast in CI.
    params = {
        **train_model.default_params(),
        "n_estimators": 2,
        "max_depth": 2,
    }

    model = train_model.make_model(
        quantile=0.5,
        params=params,
    )

    # Confirm that the actual XGBoost model can train on the resulting matrix.
    model.fit(X, y)

    predictions = model.predict(X)

    # The model should produce one finite prediction for every input row.
    assert len(predictions) == len(df)
    assert np.isfinite(predictions).all()

    model_path = tmp_path / "model.json"
    preprocessing_path = tmp_path / "preprocessing.pkl"

    # Verify that the artifacts used by deployment can actually be serialized.
    model.save_model(model_path)
    joblib.dump(preprocessing, preprocessing_path)

    assert os.path.exists(model_path)
    assert os.path.exists(preprocessing_path)

    # Confirm that the saved preprocessing artifact can be loaded again.
    loaded_preprocessing = joblib.load(
        preprocessing_path
    )

    assert loaded_preprocessing["pooled"] is False

