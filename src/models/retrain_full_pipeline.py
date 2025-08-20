import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_COLS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


def load_raw_data() -> pd.DataFrame:
    """Return the raw DataFrame with cnt column."""
    # adapt to your loader
    df = pd.read_csv("/app/data/raw/hour.csv")
    return df[RAW_COLS + ["cnt"]]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features required by the original pipeline."""
    # Temporal dummies
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_morning_rush"] = ((df["hr"] >= 7) & (df["hr"] <= 9)).astype(int)
    df["is_evening_rush"] = ((df["hr"] >= 17) & (df["hr"] <= 19)).astype(int)
    df["is_night"] = ((df["hr"] >= 22) | (df["hr"] <= 4)).astype(int)

    # Derived temperature
    df["feels_like_temp"] = df.apply(
        lambda r: r["temp"] - 0.1 * r["windspeed"] + 0.01 * (100 - r["hum"]),
        axis=1,
    )
    return df


def build_pipeline() -> Pipeline:
    """Return a single Pipeline that starts from raw 12 cols."""
    numeric_cols = ["temp", "atemp", "hum", "windspeed", "feels_like_temp"]
    cat_cols = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "is_weekend",
        "is_morning_rush",
        "is_evening_rush",
        "is_night",
    ]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
    )

    return Pipeline([("prep", preproc), ("model", model)])


def main():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("bike-sharing-full-pipeline")

    df = load_raw_data()
    df = add_engineered_features(df)

    X = df[RAW_COLS]  # only the 12 raw columns
    y = df["cnt"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="bike-sharing-model",
            input_example=X_train.iloc[:1],
        )

    # also save locally for the fallback file
    joblib.dump(pipe, "/app/models/bike_share_model.pkl")
    print("Full pipeline re-trained & registered.")


if __name__ == "__main__":
    main()
