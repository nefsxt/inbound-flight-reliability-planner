"""
Production training pipeline for route flight-duration anomaly models.

Architecture:

    features.parquet
          |
          +--> chronological ordering / training-data cleaning
          |
          +--> 80/20 chronological final holdout split
          |       |
          |       +--> Optuna + TimeSeriesSplit on TRAIN ONLY
          |       |
          |       +--> final TRAIN fit
          |       +--> ONE final holdout evaluation
          |
          +--> expanding walk-forward backtest
          |
          +--> production refit on ALL historical rows
                  |
                  +--> pooled all-carriers model
                  +--> dedicated carrier models where feasible

The deployed target is:

    flight_duration - route median duration

The route median is always calculated from the data available to that
training fit. It is never calculated from a validation/test period.

Production models are written to:

    models/<route>/
        all_carriers/
            quantile_0.5.json
            quantile_0.9.json
            quantile_0.95.json
            preprocessing_*.pkl
        by_carrier/
            <AIRLINE>/
                quantile_*.json
                preprocessing_*.pkl
        tier_manifest.json
        backtest.json
        tuning_history.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
            )
        )
    )

import config
from src.features import DEPARTURE_WEATHER_COLUMNS, ARRIVAL_WEATHER_COLUMNS
from src.utils import record_stage_stats


FEATURE_COLUMNS = ['hour_of_day', 'month'] + DEPARTURE_WEATHER_COLUMNS + ARRIVAL_WEATHER_COLUMNS

EXCLUDED_FEATURE_COLUMNS = {
    'icao24',
    'firstSeen',
    'lastSeen',
    'estDepartureAirport',
    'estArrivalAirport',
    'callsign',
    'flight_date',
    'firstSeen_hourly_utc',
    'lastSeen_hourly_utc',
    'dep_merge_key',
    'arr_merge_key',
    'departureAirportCandidatesCount',
    'arrivalAirportCandidatesCount',
    'estDepartureAirportHorizDistance',
    'estDepartureAirportVertDistance',
    'estArrivalAirportHorizDistance',
    'estArrivalAirportVertDistance',
    'route',
    'flight_duration',
    'flight_date_parsed',
    'median_flight_time'
}

CATEGORICAL_COLUMNS = ['airline']

EXPECTED_NUMERICAL_MODEL_FEATURES = [
    'dep_temperature_2m',
    'dep_relative_humidity_2m',
    'dep_dew_point_2m',
    'dep_precipitation',
    'dep_weather_code',
    'dep_pressure_msl',
    'dep_cloud_cover_low',
    'dep_cloud_cover_high',
    'dep_visibility',
    'dep_wind_speed_10m',
    'dep_wind_speed_180m',
    'dep_wind_direction_10m',
    'dep_wind_direction_180m',
    'dep_cape',
    'dep_geopotential_height_850hPa',
    'arr_temperature_2m',
    'arr_relative_humidity_2m',
    'arr_dew_point_2m',
    'arr_precipitation',
    'arr_weather_code',
    'arr_pressure_msl',
    'arr_cloud_cover_low',
    'arr_cloud_cover_high',
    'arr_visibility',
    'arr_wind_speed_10m',
    'arr_wind_speed_180m',
    'arr_wind_direction_10m',
    'arr_wind_direction_180m',
    'arr_cape',
    'arr_geopotential_height_850hPa',
    'hour_of_day',
    'month',
]

EXPECTED_POOLED_MODEL_FEATURES = EXPECTED_NUMERICAL_MODEL_FEATURES + ['airline']

TEST_FRACTION = 0.2
MIN_COMPLETE_YEARS_FOR_CV = 2
BACKTEST_FOLDS = None
OUTLIER_DURATION_MINUTES = 240
FLIGHT_DURATION_COLUMN = 'flight_duration'
GROUP_COLUMN = 'route'
COVERAGE_ERROR_WEIGHT = float(config.COVERAGE_ERROR_WEIGHT)
FOLD_BALANCE_WEIGHT =float(config.FOLD_BALANCE_WEIGHT)
DEFAULT_TRIALS = int(config.OPTUNA_TRIALS)
MIN_CARRIER_ROWS = int(config.MIN_ROWS_FOR_MODEL_FEASIBILITY)
MODEL_ROOT = config.MODELS_DIR


def route_dir(route):

    """Return the root model directory for a specific route."""

    return os.path.join(MODEL_ROOT, route.replace('->', '_'))


def all_carriers_dir(route):

    """Return the directory used for the pooled all-carriers model."""

    return os.path.join(route_dir(route), 'all_carriers')


def by_carrier_dir(route):

    """Return the parent directory used for dedicated carrier models."""

    return os.path.join(route_dir(route), 'by_carrier')


def tier_manifest_path(route):

    """Return the file path where the route model manifest is stored."""

    return os.path.join(route_dir(route), 'tier_manifest.json')


def backtest_path(route):

    """Return the file path where walk-forward backtest results are stored."""

    return os.path.join(route_dir(route), 'backtest.json')


def tuning_history_path(route):

    """Return the file path where Optuna tuning history is stored."""

    return os.path.join(route_dir(route), 'tuning_history.json')


def model_path(out_dir, quantile):

    """Build the model filename for a given output directory and quantile."""

    return os.path.join(out_dir, f'quantile_{quantile}.json')


def preprocessing_path(out_dir, quantile):

    """Build the preprocessing filename for a given output directory and quantile."""

    return os.path.join(out_dir, f'preprocessing_{quantile}.pkl')


def pinball_loss(y_true, y_pred, quantile):

    """Calculate quantile regression pinball loss for the predictions."""

    error = np.asarray(y_true) - np.asarray(y_pred)

    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def coverage_error(y_true, y_pred, quantile):

    """Measure the absolute difference between actual and target coverage."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    empirical_coverage = np.mean(y_true <= y_pred)

    return float(abs(empirical_coverage - quantile))


def calculate_fold_balanced_score(fold_results):

    """
    Combine fold metrics into the Optuna objective score.

    This balances average pinball loss, average coverage error, and
    fold-to-fold coverage consistency so that one unusually good or bad
    validation fold does not dominate model selection.
    """
    pinball_values = [result['pinball_loss'] for result in fold_results]
    coverage_errors = [result['coverage_error'] for result in fold_results]

    mean_pinball = float(np.mean(pinball_values))
    mean_coverage_error = float(np.mean(coverage_errors))
    fold_coverage_error_std = float(np.std(coverage_errors))

    score = (
        mean_pinball
        + COVERAGE_ERROR_WEIGHT * mean_coverage_error
        + FOLD_BALANCE_WEIGHT * fold_coverage_error_std
    )

    return {
        'score': float(score),
        'mean_pinball_loss': mean_pinball,
        'mean_coverage_error': mean_coverage_error,
        'fold_coverage_error_std': fold_coverage_error_std,
        'coverage_error_weight': COVERAGE_ERROR_WEIGHT,
        'fold_balance_weight': FOLD_BALANCE_WEIGHT
    }


def tuning_score(y_true, y_pred, quantile):

    """
    Calculate the single-period reporting score.

    This combines pinball loss with scaled coverage error. Optuna itself
    uses the fold-balanced score from calculate_fold_balanced_score().
    """
    loss = pinball_loss(y_true, y_pred, quantile)
    coverage = coverage_error(y_true, y_pred, quantile)

    return float(loss + COVERAGE_ERROR_WEIGHT * coverage)


def calculate_metrics(y_true, y_pred, quantile):

    """
    Calculate the main evaluation metrics for a quantile model.

    This includes quantile accuracy, coverage accuracy, standard error
    diagnostics, prediction bias, and empirical versus target coverage.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    empirical_coverage = np.mean(y_true <= y_pred)
    coverage = coverage_error(y_true, y_pred, quantile)

    return {
        'pinball_loss': pinball_loss(y_true, y_pred, quantile),
        'coverage_error': coverage,
        'coverage_penalty': coverage,
        'tuning_score': tuning_score(y_true, y_pred, quantile),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'medae': float(median_absolute_error(y_true, y_pred)),
        'rmse': float(root_mean_squared_error(y_true, y_pred)),
        'mean_error': float(np.mean(error)),
        'empirical_coverage_pct': float(empirical_coverage * 100),
        'target_coverage_pct': float(quantile * 100),
        'coverage_gap_pct': float((empirical_coverage - quantile) * 100)
    }


def prepare_dataset(df):

    """
    Validate the feature dataframe, preserve chronological order, and remove
    only flights whose duration exceeds the fixed 240-minute threshold.
    """
    df = df.copy()

    required = [
        'icao24',
        'firstSeen',
        'lastSeen',
        'route',
        'airline',
        'flight_duration',
        'hour_of_day',
        'month',
        'flight_date_parsed'
    ] + DEPARTURE_WEATHER_COLUMNS + ARRIVAL_WEATHER_COLUMNS

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f'Feature dataframe is missing columns: {missing}')

    df = df.sort_values(['firstSeen', 'icao24'], kind='stable').reset_index(drop=True)

    outlier_mask = df[FLIGHT_DURATION_COLUMN] > OUTLIER_DURATION_MINUTES

    print(
        f'Removing {int(outlier_mask.sum())} duration outlier(s) '
        f'> {OUTLIER_DURATION_MINUTES} minutes.'
    )

    df = df.loc[~outlier_mask].reset_index(drop=True)

    if (df[FLIGHT_DURATION_COLUMN] <= 0).any():
        raise ValueError('Found zero/negative flight durations after cleaning.')

    if (~np.isfinite(df[FLIGHT_DURATION_COLUMN])).any():
        raise ValueError('flight_duration contains non-finite values after cleaning.')

    return df


def select_model_features(df):

    """
    Select prototype-approved model features and remove globally constant
    numerical features. Columns in EXCLUDED_FEATURE_COLUMNS are never
    allowed into the model feature matrix.
    """
    validate_candidate_features(df)

    available_features = [
        col for col in df.columns
        if col not in EXCLUDED_FEATURE_COLUMNS
    ]

    numeric_cols = [
        col for col in available_features
        if col in FEATURE_COLUMNS
    ]

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f'Model feature dataframe is missing columns: {missing}')



    selected_features = [
        c for c in numeric_cols
        
    ]



    if not selected_features:
        raise ValueError(
            'No usable numerical model features remain after constant-feature removal.'
        )

    return selected_features

def validate_model_features(X, pooled):
    """
    Validate that the model feature matrix has exactly the expected
    columns in the expected order before it is passed to XGBoost.

    Pooled models additionally require the airline feature. Dedicated
    carrier models do not.
    """
    expected = EXPECTED_POOLED_MODEL_FEATURES if pooled else EXPECTED_NUMERICAL_MODEL_FEATURES
    actual = list(X.columns)

    if actual != expected:
        missing = [column for column in expected if column not in actual]
        unexpected = [column for column in actual if column not in expected]

        raise ValueError(
            'Model feature schema validation failed.\n'
            f'Expected columns: {expected}\n'
            f'Actual columns: {actual}\n'
            f'Missing columns: {missing}\n'
            f'Unexpected columns: {unexpected}'
        )


def validate_candidate_features(df):
    """
    Validate that the feature dataframe contains exactly the expected
    numerical model features before feature selection and preprocessing.
    """
    actual = [
        column for column in df.columns
        if column in FEATURE_COLUMNS
    ]

    expected = EXPECTED_NUMERICAL_MODEL_FEATURES

    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]

    if missing or unexpected:
        raise ValueError(
            'Candidate feature schema validation failed.\n'
            f'Missing columns: {missing}\n'
            f'Unexpected model candidates: {unexpected}'
        )

def chronological_split(df):

    """
    Split the chronologically ordered dataframe into an initial training
    period and a final holdout period while preserving row order.
    """
    split = int(len(df) * (1 - TEST_FRACTION))

    if split <= 0 or split >= len(df):
        raise ValueError(
            'Dataset is too small for chronological train/test split.'
        )

    return df.iloc[:split].copy(), df.iloc[split:].copy()


def determine_cv_folds(df):

    """
    Count complete calendar years and use that count as the number of
    chronological validation folds.

    This function is population-specific. It is called independently for
    pooled route data and for each dedicated carrier dataset so that each
    model tier gets a validation scheme appropriate to its own history.
    """
    dates = pd.to_datetime(df['firstSeen'], utc=True)

    years = sorted(dates.dt.year.dropna().unique())

    complete_years = []

    for year in years:
        year_dates = dates[dates.dt.year == year]

        if (
            year_dates.dt.month.min() == 1
            and year_dates.dt.month.max() == 12
        ):
            complete_years.append(int(year))

    n_folds = len(complete_years)

    if n_folds < MIN_COMPLETE_YEARS_FOR_CV:
        raise ValueError(
            f'Only {n_folds} complete calendar year(s) found in this '
            f'training population. At least {MIN_COMPLETE_YEARS_FOR_CV} '
            f'complete years are required for chronological tuning/backtesting.'
        )

    return n_folds, complete_years


def remove_fold_leakage_columns(df):

    """
    Remove columns that are required for fold-level target construction or
    operational logging but must never be exposed to the ML model.

    These columns are intentionally retained until after the target has
    been calculated because flight_duration and route are required to
    construct the residual target.
    """
    columns_to_drop = [
        FLIGHT_DURATION_COLUMN,
        GROUP_COLUMN,
        'flight_date_parsed',
        'median_flight_time'
    ]

    return df.drop(
        columns=[c for c in columns_to_drop if c in df.columns]
    ).copy()

def target_from_training(train_df, other_df):

    """
    Calculate residual targets using the route median learned only from
    the training dataframe, then apply that same median to the other dataframe.
    """
    route_median = float(train_df[FLIGHT_DURATION_COLUMN].median())

    y_train = train_df[FLIGHT_DURATION_COLUMN] - route_median
    y_other = other_df[FLIGHT_DURATION_COLUMN] - route_median

    return y_train, y_other, route_median

def fit_preprocessor(train_df, y_train, pooled, model_feature_columns, quantile):

    """
    Fit feature preprocessing using training data only, including the pooled
    airline target encoding and numerical median imputation.

    train_df must already have fold-level leakage columns removed.
    """
    columns = (
        model_feature_columns + CATEGORICAL_COLUMNS
        if pooled
        else model_feature_columns
    )

    X = train_df[columns].copy()

    preprocessing = {
        'feature_columns': columns,
        'model_feature_columns': model_feature_columns,
        'pooled': pooled,
        'categorical_columns': CATEGORICAL_COLUMNS if pooled else [],
        'target_encoding_quantile': quantile if pooled else None
    }

    if pooled:
        encoding = pd.DataFrame({
            'airline': train_df['airline'].values,
            'target': np.asarray(y_train)
        })

        airline_map = encoding.groupby('airline')['target'].quantile(quantile)
        global_value = float(np.quantile(y_train, quantile))

        X['airline'] = (
            X['airline']
            .map(airline_map)
            .fillna(global_value)
        )

        preprocessing['airline_encoding'] = {
            'method': 'training_target_quantile',
            'quantile': quantile,
            'mapping': {
                str(k): float(v)
                for k, v in airline_map.items()
            },
            'fallback': global_value
        }

    imputer = SimpleImputer(strategy='median')
    imputer.fit(X)

    preprocessing['imputer_statistics'] = {
        col: float(value) if np.isfinite(value) else None
        for col, value in zip(X.columns, imputer.statistics_)
    }

    return preprocessing, imputer


def transform_features(df, preprocessing, imputer):
    """
    Apply previously fitted encoding and imputation to new data without
    recalculating preprocessing statistics from that data.
    """
    columns = preprocessing['feature_columns']

    X = df[columns].copy()

    if preprocessing['pooled']:
        encoding = preprocessing['airline_encoding']

        X['airline'] = (
            X['airline']
            .map(encoding['mapping'])
            .fillna(encoding['fallback'])
        )

    return pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index
    )


def make_fold_features(train_df, valid_df, y_train, pooled, model_feature_columns, quantile):

    """
    Fit preprocessing on one training fold and transform both the training
    and validation fold using the same fitted preprocessing.

    Fold-level target-construction columns are removed only after the target
    has already been calculated.
    """
    train_features = remove_fold_leakage_columns(train_df)
    valid_features = remove_fold_leakage_columns(valid_df)

    preprocessing, imputer = fit_preprocessor(
        train_features,
        y_train,
        pooled,
        model_feature_columns,
        quantile,
    )

    X_train = transform_features(train_features, preprocessing, imputer)
    X_valid = transform_features(valid_features, preprocessing, imputer)

    validate_model_features(X_train, pooled)
    validate_model_features(X_valid, pooled)

    return X_train, X_valid


def default_params():

    """Return the default XGBoost hyperparameters used when tuning is disabled."""

    return {
        'n_estimators': 200,
        'learning_rate': 0.015,
        'max_depth': 2,
        'min_child_weight': 1,
        'subsample': 0.5,
        'colsample_bytree': 0.4,
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
        'gamma': 0.0
    }


def make_model(quantile, params):

    """
    Create an XGBoost quantile-regression model using the requested quantile.
    """
    return xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=quantile,
        **params,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )


def tune_quantile(train_df, quantile, pooled, n_trials, model_feature_columns, n_folds):

    """
    Tune one quantile model using only the chronological training period.

    Each Optuna trial is evaluated across TimeSeriesSplit folds, with the
    fold-balanced objective combining pinball loss, coverage error, and
    fold-to-fold coverage consistency.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            'Import Optuna before using --tune.'
        ) from exc

    if len(train_df) <= n_folds:
        raise ValueError(
            'Not enough training rows for chronological Optuna CV.'
        )

    folds = list(
        TimeSeriesSplit(n_splits=n_folds).split(train_df)
    )

    def objective(trial):
        """
        Evaluate one Optuna hyperparameter configuration across all CV folds.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 30.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 10.0)
        }

        fold_results = []

        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            fold_train = train_df.iloc[train_idx].copy()
            fold_valid = train_df.iloc[valid_idx].copy()

            y_train, y_valid, _ = target_from_training(fold_train, fold_valid,)

            X_train, X_valid = make_fold_features(fold_train, fold_valid, y_train, pooled, model_feature_columns, quantile)

            model = make_model(quantile, params)
            model.fit(X_train, y_train)

            predictions = model.predict(X_valid)

            fold_metrics = calculate_metrics(y_valid, predictions, quantile)

            fold_metrics['fold'] = fold
            fold_metrics['n_train'] = len(fold_train)
            fold_metrics['n_valid'] = len(fold_valid)

            fold_results.append(fold_metrics)

        balanced = calculate_fold_balanced_score(fold_results)

        return balanced['score']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    return study


def walkforward(df, quantile, params, pooled, model_feature_columns, n_folds):

    """
    Run chronological walk-forward validation, fitting each fold only on
    earlier observations and evaluating on the following unseen observations.

    This is a historical backtest/diagnostic and is separate from the final
    chronological holdout evaluation. It does not change hyperparameters or
    production model selection.
    """
    results = []

    splitter = TimeSeriesSplit(n_splits=n_folds)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        y_train, y_test, route_median  = target_from_training(train_df, test_df)

        X_train, X_test = make_fold_features(
            train_df,
            test_df,
            y_train,
            pooled,
            model_feature_columns,
            quantile
        )

        model = make_model(quantile, params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        result = calculate_metrics(
            y_test,
            predictions,
            quantile
        )

        result.update({
            'fold': fold,
            'train_start': str(train_df['firstSeen'].min()),
            'train_end': str(train_df['firstSeen'].max()),
            'test_start': str(test_df['firstSeen'].min()),
            'test_end': str(test_df['firstSeen'].max()),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'route_median': route_median,
        })

        results.append(result)

    fold_balance = calculate_fold_balanced_score(results)

    return {
        'folds': results,
        'aggregate': fold_balance
    }

def fit_production_model(df, quantile, params, pooled, out_dir, model_feature_columns):

    """
    Fit the final deployable model on all available historical rows after
    tuning and final holdout evaluation have been completed.
    """
    y, _, route_median = target_from_training(df, df)

    clean_features = remove_fold_leakage_columns(df)

    preprocessing, imputer = fit_preprocessor(
        clean_features,
        y,
        pooled,
        model_feature_columns,
        quantile
    )

    X = transform_features(
        clean_features,
        preprocessing,
        imputer
    )

    validate_model_features(X, pooled)

    model = make_model(quantile, params)
    model.fit(X, y)

    preprocessing['route_median'] = route_median
    preprocessing['training_rows'] = len(df)
    preprocessing['trained_through'] = str(df['firstSeen'].max())

    model.save_model(
        model_path(out_dir, quantile)
    )

    joblib.dump(
        preprocessing,
        preprocessing_path(out_dir, quantile)
    )

    importance = dict(
        zip(
            X.columns,
            model.feature_importances_.astype(float)
        )
    )

    return {
        'model_path': model_path(out_dir, quantile),
        'preprocessing_path': preprocessing_path(out_dir, quantile),
        'model_features': list(X.columns),
        'route_median': route_median,
        'feature_importance': importance
    }


def append_tuning_history(route, entry):

    """
    Append one completed Optuna tuning run to the route's persistent history
    file, creating the file if it does not already exist.
    """
    path = tuning_history_path(route)
    history = []

    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)

    history.append(entry)

    with open(path, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def train_tier(df, route, tier_name, pooled, tune, n_trials, model_feature_columns, n_folds):

    """
    Train all configured quantiles for either the pooled carrier tier or
    one dedicated carrier tier, including evaluation, backtesting, and
    final production fitting.
    """
    out_dir = (
        all_carriers_dir(route)
        if pooled
        else os.path.join(by_carrier_dir(route), tier_name)
    )

    os.makedirs(out_dir, exist_ok=True)

    train_df, test_df = chronological_split(df)
    results = {}

    for quantile in config.QUANTILES:
        tier_label = 'ALL CARRIERS' if pooled else tier_name

        print(
            f"\n[{route}] [{tier_label}] P{int(quantile * 100)}"
        )

        if tune:
            study = tune_quantile(
                train_df,
                quantile,
                pooled,
                n_trials,
                model_feature_columns,
                n_folds
            )

            params = study.best_params

            append_tuning_history(
                route,
                {
                    'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'route': route,
                    'tier': 'all_carriers' if pooled else tier_name,
                    'quantile': quantile,
                    'n_trials': n_trials,
                    'best_value': study.best_value,
                    'best_params': params,
                    'optimization_metric': (
                        'mean_pinball_loss + scaled_mean_coverage_error + '
                        'scaled_fold_coverage_error_std'
                    ),
                    'coverage_error_definition': (
                        'absolute empirical coverage error per chronological fold'
                    ),
                    'fold_balance_definition': (
                        'standard deviation of absolute coverage error across folds'
                    ),
                    'coverage_error_weight': COVERAGE_ERROR_WEIGHT,
                    'fold_balance_weight': FOLD_BALANCE_WEIGHT,
                    'n_validation_folds': n_folds,
                    'model_features': (
                        model_feature_columns + ['airline']
                        if pooled
                        else model_feature_columns
                    ),
                    'optimization_period': {
                        'start': str(train_df['firstSeen'].min()),
                        'end': str(train_df['firstSeen'].max())
                    }
                }
            )
        else:
            params = default_params()

        y_train, y_test, _ = target_from_training(
            train_df,
            test_df
        )

        X_train, X_test = make_fold_features(
            train_df,
            test_df,
            y_train,
            pooled,
            model_feature_columns,
            quantile
        )

        validate_model_features(X_train, pooled)
        validate_model_features(X_test, pooled)

        evaluation_model = make_model(quantile, params)
        evaluation_model.fit(X_train, y_train)

        test_predictions = evaluation_model.predict(X_test)

        test_metrics = calculate_metrics(
            y_test,
            test_predictions,
            quantile
        )

        print(
            f"  HOLDOUT score={test_metrics['tuning_score']:.4f} "
            f"pinball={test_metrics['pinball_loss']:.4f} "
            f"coverage_error={test_metrics['coverage_error']:.4f} "
            f"MAE={test_metrics['mae']:.2f} "
            f"coverage={test_metrics['empirical_coverage_pct']:.1f}%"
        )

        wf = walkforward(
            df,
            quantile,
            params,
            pooled,
            model_feature_columns,
            n_folds
        )

        print(
            f"  BACKTEST balanced score={wf['aggregate']['score']:.4f} "
            f"mean_pinball={wf['aggregate']['mean_pinball_loss']:.4f} "
            f"mean_coverage_error={wf['aggregate']['mean_coverage_error']:.4f} "
            f"fold_coverage_error_std={wf['aggregate']['fold_coverage_error_std']:.4f}"
        )

        production = fit_production_model(
            df,
            quantile,
            params,
            pooled,
            out_dir,
            model_feature_columns
        )

        results[quantile] = {
            'hyperparameters': params,
            'model_features': production['model_features'],
            'final_holdout_metrics': test_metrics,
            'walkforward_backtest': wf,
            'production': production,
            'n_rows': len(df),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'n_validation_folds': n_folds,
            'train_start': str(train_df['firstSeen'].min()),
            'train_end': str(train_df['firstSeen'].max()),
            'holdout_start': str(test_df['firstSeen'].min()),
            'holdout_end': str(test_df['firstSeen'].max()),
            'production_start': str(df['firstSeen'].min()),
            'production_end': str(df['firstSeen'].max())
        }

    return {
        'route': route,
        'tier': 'all_carriers' if pooled else tier_name,
        'pooled': pooled,
        'trained': True,
        'n_rows': len(df),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_validation_folds': n_folds,
        'model_features': (
            model_feature_columns + ['airline']
            if pooled
            else model_feature_columns
        ),
        'quantiles': results
    }


def eligible_carriers(df):

    """
    Return carrier codes with enough historical observations to support
    training a dedicated carrier model.
    """
    counts = df['airline'].value_counts()

    return sorted(
        code
        for code, n in counts.items()
        if n >= MIN_CARRIER_ROWS
    )


def determine_carrier_validation_policy(carrier_df):

    """
    Determine whether a dedicated carrier dataset has enough independent
    temporal coverage for chronological validation.

    Carrier models are evaluated using the carrier's own complete calendar
    years rather than inheriting the pooled route's validation-fold count.
    """
    try:
        n_folds, complete_years = determine_cv_folds(carrier_df)
    except ValueError as exc:
        return {
            'feasible': False,
            'n_folds': 0,
            'complete_years': [],
            'reason': str(exc)
        }

    return {
        'feasible': True,
        'n_folds': n_folds,
        'complete_years': complete_years,
        'reason': None
    }


def write_manifest(
    route,
    dataset,
    pooled_result,
    carrier_results,
    model_feature_columns,
    removed_constant_features,
    tune,
    n_trials,
    pooled_n_folds
):
    """
    Write the route manifest and backtest summary containing model metadata,
    validation policy, feature selection, and trained model results.
    """
    manifest = {
        'route': route,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_policy': (
            f'Use dedicated carrier model when the requested carrier has '
            f'>={MIN_CARRIER_ROWS} historical flights and its own training '
            f'population supports chronological validation; otherwise use '
            f'the pooled all-carriers model.'
        ),
        'production_policy': (
            'Hyperparameters are selected using chronological CV within '
            'the training period only. The final chronological holdout is '
            'evaluated once and is not used for Optuna/model selection. '
            'After final holdout evaluation, selected hyperparameters are '
            'refit on all available historical rows for deployment.'
        ),
        'validation_policy': {
            'pooled_n_folds': pooled_n_folds,
            'carrier_folds_are_independent': True,
            'basis': 'number of complete calendar years in each model training population',
            'minimum_complete_years': MIN_COMPLETE_YEARS_FOR_CV
        },
        'final_holdout_policy': {
            'fraction': TEST_FRACTION,
            'method': 'chronological final holdout',
            'used_for_tuning': False,
            'used_for_model_selection': False,
            'evaluation_frequency': 'once per model and quantile'
        },
        'walkforward_backtest_policy': {
            'purpose': 'historical expanding walk-forward diagnostic',
            'used_for_tuning': False,
            'used_for_model_selection': False,
            'same_population_as_production_fit': True
        },
        'tuning_policy': {
            'metric': 'mean_pinball_loss + scaled_mean_coverage_error + scaled_fold_coverage_error_std',
            'coverage_error': 'absolute empirical coverage error',
            'fold_balance': 'standard deviation of coverage error across folds',
            'coverage_error_weight': COVERAGE_ERROR_WEIGHT,
            'fold_balance_weight': FOLD_BALANCE_WEIGHT
        },
        'dataset': dataset,
        'all_carriers': pooled_result,
        'by_carrier': {
            'carriers': carrier_results
        },
        'feature_selection': {
            'base_features': model_feature_columns,
            'removed_constant_features': removed_constant_features,
            'pooled_additional_feature': 'airline',
            'excluded_from_model': sorted(EXCLUDED_FEATURE_COLUMNS),
        },
        'feature_columns': model_feature_columns,
        'pooled_feature_columns': model_feature_columns + ['airline'],
        'target': 'flight_duration - route median'
    }

    with open(tier_manifest_path(route), 'w') as f:
        json.dump(
            manifest,
            f,
            indent=2,
            default=str
        )

    backtest = {
        'validation_policy': {
            'pooled_n_folds': pooled_n_folds,
            'carrier_folds_are_independent': True,
            'basis': 'complete calendar years within each training population'
        },
        'objective': {
            'metric': 'mean_pinball_loss + scaled_mean_coverage_error + scaled_fold_coverage_error_std',
            'coverage_error_weight': COVERAGE_ERROR_WEIGHT,
            'fold_balance_weight': FOLD_BALANCE_WEIGHT
        },
        'all_carriers': {
            'all_carriers': {
                str(q): result['walkforward_backtest']
                for q, result in pooled_result['quantiles'].items()
            }
        },
        'by_carrier': {
            code: {
                str(q): result['walkforward_backtest']
                for q, result in carrier_results[code]['quantiles'].items()
            }
            for code in carrier_results
            if carrier_results[code].get('trained', False)
        }
    }

    with open(backtest_path(route), 'w') as f:
        json.dump(
            backtest,
            f,
            indent=2,
            default=str
        )

    return manifest


def train_route(df, route, tune=False, n_trials=DEFAULT_TRIALS):

    """
    Train the pooled model and eligible dedicated carrier models for one
    route, then write the final route manifest and validation results.

    The pooled model and each dedicated carrier model determine their own
    chronological validation-fold count from their respective training
    populations.
    """
    os.makedirs(
        route_dir(route),
        exist_ok=True
    )

    train_df, test_df = chronological_split(df)

    model_feature_columns, removed_constant_features = select_model_features(df)

    pooled_n_folds, pooled_complete_years = determine_cv_folds(train_df)

    dataset = {
        'route': route,
        'total_rows': len(df),
        'dataset_start': str(df['firstSeen'].min()),
        'dataset_end': str(df['firstSeen'].max()),
        'train_start': str(train_df['firstSeen'].min()),
        'train_end': str(train_df['firstSeen'].max()),
        'holdout_start': str(test_df['firstSeen'].min()),
        'holdout_end': str(test_df['firstSeen'].max()),
        'train_rows': len(train_df),
        'holdout_rows': len(test_df),
        'complete_training_years_pooled': pooled_complete_years,
        'pooled_n_validation_folds': pooled_n_folds,
        'model_features': model_feature_columns,
        'removed_constant_features': removed_constant_features,
        'outlier_rule': {
            'duration_gt_minutes': OUTLIER_DURATION_MINUTES,
            'date_filter': None
        }
    }

    print(
        f'[{route}] Using {pooled_n_folds} pooled chronological '
        f'validation folds based on complete training years: '
        f'{pooled_complete_years}'
    )

    print(
        f'[{route}] Optuna objective weights: '
        f'coverage_error={COVERAGE_ERROR_WEIGHT}, '
        f'fold_balance={FOLD_BALANCE_WEIGHT}'
    )

    pooled_result = train_tier(
        df,
        route,
        'all_carriers',
        pooled=True,
        tune=tune,
        n_trials=n_trials,
        model_feature_columns=model_feature_columns,
        n_folds=pooled_n_folds
    )

    carrier_results = {}

    for code in eligible_carriers(df):
        carrier_df = df[df['airline'] == code].copy()

        carrier_train_df, carrier_test_df = chronological_split(carrier_df)

        carrier_validation = determine_carrier_validation_policy(
            carrier_train_df
        )

        if not carrier_validation['feasible']:
            carrier_results[code] = {
                'route': route,
                'tier': code,
                'pooled': False,
                'trained': False,
                'n_rows': len(carrier_df),
                'n_train': len(carrier_train_df),
                'n_test': len(carrier_test_df),
                'validation_policy': carrier_validation,
                'reason': carrier_validation['reason']
            }

            print(
                f'[{route}] [{code}] Dedicated model skipped: '
                f'{carrier_validation["reason"]}'
            )

            continue

        carrier_n_folds = carrier_validation['n_folds']
        carrier_complete_years = carrier_validation['complete_years']

        print(
            f'[{route}] [{code}] Using {carrier_n_folds} chronological '
            f'validation folds based on carrier-specific complete training '
            f'years: {carrier_complete_years}'
        )

        carrier_results[code] = train_tier(
            carrier_df,
            route,
            code,
            pooled=False,
            tune=tune,
            n_trials=n_trials,
            model_feature_columns=model_feature_columns,
            n_folds=carrier_n_folds
        )

        carrier_results[code]['validation_policy'] = {
            'n_folds': carrier_n_folds,
            'complete_training_years': carrier_complete_years,
            'basis': 'complete calendar years in carrier training population'
        }

    all_counts = df['airline'].value_counts()

    for code, count in all_counts.items():
        if code not in carrier_results:
            carrier_results[code] = {
                'route': route,
                'tier': code,
                'pooled': False,
                'trained': False,
                'n_rows': int(count),
                'reason': (
                    f'Only {int(count)} flights; dedicated model '
                    f'requires {MIN_CARRIER_ROWS}.'
                )
            }

    return write_manifest(
        route,
        dataset,
        pooled_result,
        carrier_results,
        model_feature_columns,
        removed_constant_features,
        tune,
        n_trials,
        pooled_n_folds
    )


def train_and_evaluate(tune=False, n_trials=DEFAULT_TRIALS):
    
    """
    Load the feature dataset, validate that exactly one route is present,
    train the route models, and record final pipeline statistics.
    """
    path = config.features_path()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Run `python -m src.features` first.'
        )

    flights = prepare_dataset(
        pd.read_parquet(path)
    )

    routes = sorted(
        flights['route'].dropna().unique()
    )

    if len(routes) != 1:
        raise ValueError(
            f'train_model.py expects exactly one route. Found: {routes}'
        )

    route = routes[0]

    flights = flights[
        flights['route'] == route
    ].copy()

    if len(flights) < config.MIN_ROWS_FOR_MODEL_FEASIBILITY:
        raise ValueError(
            f'{route} has {len(flights)} usable rows, below the configured '
            f'feasibility threshold of '
            f'{config.MIN_ROWS_FOR_MODEL_FEASIBILITY}.'
        )

    print(
        f"\n[{route}] {len(flights)} usable flights "
        f"{flights['firstSeen'].min().date()} -> "
        f"{flights['firstSeen'].max().date()}"
    )

    manifest = train_route(
        flights,
        route,
        tune=tune,
        n_trials=n_trials
    )

    model_features = manifest['feature_columns']

    record_stage_stats(
        '4_train_model',
        flights[
            model_features
            + ['airline', 'route', FLIGHT_DURATION_COLUMN]
        ],
        extra={
            'route': route,
            'tuning_enabled': tune,
            'n_trials': n_trials if tune else None,
            'dataset_start': str(flights['firstSeen'].min()),
            'dataset_end': str(flights['firstSeen'].max()),
            'pooled_model': True,
            'dedicated_carrier_models': eligible_carriers(flights),
            'feasibility_threshold': MIN_CARRIER_ROWS,
            'model_features': model_features,
            'pooled_model_features': model_features + ['airline'],
            'pooled_n_validation_folds': manifest['dataset']['pooled_n_validation_folds'],
            'complete_training_years_pooled': manifest['dataset']['complete_training_years_pooled'],
            'carrier_validation_policy': (
                'Each dedicated carrier model determines its own validation '
                'fold count from its own complete training years.'
            ),
            'tuning_metric': 'mean_pinball_loss + scaled_mean_coverage_error + scaled_fold_coverage_error_std',
            'coverage_error_weight': COVERAGE_ERROR_WEIGHT,
            'fold_balance_weight': FOLD_BALANCE_WEIGHT,
            'final_holdout_policy': (
                'Chronological final 20% evaluated once and never used '
                'for Optuna/model selection.'
            ),
            'walkforward_backtest_policy': (
                'Historical expanding walk-forward diagnostic; not used '
                'for Optuna/model selection.'
            ),
            'production_policy': (
                'After final holdout evaluation, selected hyperparameters '
                'are refit on all historical rows for deployment.'
            ),
            'outlier_rule': {
                'duration_gt_minutes': OUTLIER_DURATION_MINUTES,
                'date_filter': None
            }
        }
    )

    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train production flight-duration anomaly models.'
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run Optuna using chronological CV on the training period only.'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=DEFAULT_TRIALS,
        help=f'Optuna trials per quantile/model. Default: {DEFAULT_TRIALS}'
    )

    args = parser.parse_args()

    train_and_evaluate(
        tune=args.tune,
        n_trials=args.trials
    )