from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


APP_VERSION = "artifacts-v3"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "dashboard_artifacts.joblib"
METADATA_PATH = ARTIFACT_DIR / "dashboard_artifacts_meta.joblib"


st.set_page_config(
    page_title="NBA Salary Forecast Studio",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class ModelArtifacts:
    forecast_df: pd.DataFrame
    feature_columns: list[str]
    summary_df: pd.DataFrame
    results_df: pd.DataFrame
    best_model_name: str
    best_pipeline: Pipeline
    holdout_df: pd.DataFrame
    feature_importance_df: pd.DataFrame


def format_feature_label(feature: str) -> str:
    label = feature
    if label.endswith("_rolling3"):
        label = f"{label[:-9]} (3yr avg)"
    elif label.endswith("_lag1"):
        label = f"{label[:-5]} (prev year)"

    replacements = {
        "GP": "Games played",
        "MIN": "Minutes",
        "PTS": "Points",
        "REB": "Rebounds",
        "AST": "Assists",
        "STL": "Steals",
        "BLK": "Blocks",
        "TOV": "Turnovers",
        "FG_PCT": "FG%",
        "FG3_PCT": "3PT%",
        "FT_PCT": "FT%",
        "FG3A_rate": "3PA rate",
        "FTA_rate": "FTA rate",
        "TS_proxy": "True shooting proxy",
        "AST_TOV_ratio": "AST/TOV ratio",
        "PLUS_MINUS": "Plus/minus",
        "DD2": "Double-doubles",
        "TD3": "Triple-doubles",
        "PTS_per36": "Points per 36",
        "REB_per36": "Rebounds per 36",
        "AST_per36": "Assists per 36",
        "STL_per36": "Steals per 36",
        "BLK_per36": "Blocks per 36",
        "TOV_per36": "Turnovers per 36",
        "AGE": "Age",
        "Salary": "Current salary",
        "Season": "Season",
        "seasons_played": "Years in league",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #ffffff;
            color: #000000;
            font-family: "Times New Roman", Times, serif;
        }
        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 1.5rem;
            max-width: 1000px;
        }
        .app-shell {
            padding: 0;
            margin-bottom: 0.75rem;
        }
        .app-kicker {
            font-family: "Times New Roman", Times, serif;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0;
            color: #000000;
            margin-bottom: 0.25rem;
        }
        .app-title {
            font-family: "Times New Roman", Times, serif;
            font-size: 1.9rem;
            line-height: 1.1;
            font-weight: 700;
            margin-bottom: 0.3rem;
            color: #000000;
        }
        .app-copy {
            color: #000000;
            font-size: 1rem;
            line-height: 1.35;
            max-width: 52rem;
        }
        .section-title {
            font-family: "Times New Roman", Times, serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #000000;
            margin: 0.6rem 0 0.2rem;
        }
        .section-copy {
            color: #000000;
            margin-bottom: 0.55rem;
        }
        div[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d0d0d0;
        }
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #000000;
        }
        div[data-testid="stSidebar"] .stSlider label,
        div[data-testid="stSidebar"] .stSelectbox label,
        div[data-testid="stSidebar"] .stTextInput label {
            color: #000000;
        }
        div[data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
            background: #000000;
        }
        div[data-testid="stSidebar"] [data-baseweb="select"] > div,
        div[data-testid="stSidebar"] input {
            background: #ffffff !important;
            border-color: #000000 !important;
            color: #000000 !important;
        }
        div[data-testid="stTabs"] button {
            font-family: "Times New Roman", Times, serif;
            font-weight: 700;
            color: #000000;
        }
        div[data-testid="stTabs"] {
            margin-top: 0.15rem;
        }
        .stMarkdown, .stText, .stCaption, label, p, div, span {
            font-family: "Times New Roman", Times, serif;
        }
        .stPlotlyChart, .stDataFrame {
            border: none;
            padding: 0;
            background: transparent;
        }
        h1, h2, h3 {
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_model_ready_stats(base_stats: pd.DataFrame) -> pd.DataFrame:
    stats = base_stats.rename(columns={"PLAYER_NAME": "Player", "season": "Season"}).copy()
    stats = stats.sort_values(["PLAYER_ID", "Season"]).reset_index(drop=True)

    per_36_stats = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
    for stat in per_36_stats:
        stats[f"{stat}_per36"] = np.where(stats["MIN"] > 0, stats[stat] / stats["MIN"] * 36, 0)

    stats["AST_TOV_ratio"] = np.where(stats["TOV"] > 0, stats["AST"] / stats["TOV"], stats["AST"])
    stats["FG3A_rate"] = np.where(stats["FGA"] > 0, stats["FG3A"] / stats["FGA"], 0)
    stats["FTA_rate"] = np.where(stats["FGA"] > 0, stats["FTA"] / stats["FGA"], 0)
    stats["TS_proxy"] = np.where(
        (stats["FGA"] + 0.44 * stats["FTA"]) > 0,
        stats["PTS"] / (2 * (stats["FGA"] + 0.44 * stats["FTA"])),
        0,
    )
    stats["seasons_played"] = stats.groupby("PLAYER_ID").cumcount()

    lag_features = [
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "PLUS_MINUS",
        "DD2",
        "TD3",
        "PTS_per36",
        "REB_per36",
        "AST_per36",
        "STL_per36",
        "BLK_per36",
        "TOV_per36",
        "AST_TOV_ratio",
        "FG3A_rate",
        "FTA_rate",
        "TS_proxy",
    ]
    for feature in lag_features:
        stats[f"{feature}_lag1"] = stats.groupby("PLAYER_ID")[feature].shift(1)
        stats[f"{feature}_rolling3"] = (
            stats.groupby("PLAYER_ID")[feature]
            .transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
        )

    ambiguous_name_seasons = (
        stats.groupby(["Player", "Season"])["PLAYER_ID"]
        .nunique()
        .reset_index(name="player_ids")
        .query("player_ids > 1")
    )
    stats = stats.merge(
        ambiguous_name_seasons[["Player", "Season"]].assign(_ambiguous_name=1),
        on=["Player", "Season"],
        how="left",
    )
    return stats[stats["_ambiguous_name"].isna()].drop(columns=["_ambiguous_name"])


@st.cache_data(show_spinner=False)
def load_stats_data() -> pd.DataFrame:
    try:
        return pd.read_csv("player_stats_model_ready.csv")
    except FileNotFoundError:
        # fall back to rebuilding the cleaned stats file if it is missing
        return build_model_ready_stats(pd.read_csv("player_stats.csv"))


@st.cache_data(show_spinner=False)
def load_salary_data() -> pd.DataFrame:
    return pd.read_csv("nba_salaries.csv")


def build_candidate_models() -> dict[str, object]:
    models: dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=20000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20000),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_depth=6,
            max_iter=220,
            l2_regularization=1.0,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=140,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=180,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(C=8.0, epsilon=0.08, kernel="rbf")),
            ]
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    return models


def train_dashboard_model() -> ModelArtifacts:
    stats = load_stats_data()
    salaries = load_salary_data()

    forecast_df = stats.merge(salaries, on=["Player", "Season"], how="inner")
    forecast_df = forecast_df.sort_values(["PLAYER_ID", "Season"]).reset_index(drop=True)
    forecast_df["target_season"] = forecast_df.groupby("PLAYER_ID")["Season"].shift(-1)
    forecast_df["next_salary"] = forecast_df.groupby("PLAYER_ID")["Salary"].shift(-1)
    forecast_df = forecast_df[forecast_df["target_season"] == forecast_df["Season"] + 1].copy()
    forecast_df["target_season"] = forecast_df["target_season"].astype(int)
    forecast_df["log_next_salary"] = np.log1p(forecast_df["next_salary"])

    feature_columns = [
        column
        for column in forecast_df.columns
        if column not in {"Player", "PLAYER_ID", "next_salary", "log_next_salary", "target_season"}
    ]

    X = forecast_df[feature_columns]
    y = forecast_df["log_next_salary"]
    eval_seasons = sorted(forecast_df["target_season"].unique())
    candidates = build_candidate_models()

    all_results: list[pd.DataFrame] = []
    best_name = ""
    best_score = float("inf")

    for name, estimator in candidates.items():
        model_rows = []
        for season in eval_seasons:
            train_mask = forecast_df["target_season"] < season
            test_mask = forecast_df["target_season"] == season
            if train_mask.sum() < 200 or test_mask.sum() == 0:
                continue

            pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
            )
            pipeline.fit(X.loc[train_mask], y.loc[train_mask])
            pred_salary = np.expm1(pipeline.predict(X.loc[test_mask]))
            actual_salary = forecast_df.loc[test_mask, "next_salary"]
            model_rows.append(
                {
                    "model": name,
                    "season": int(season),
                    "rows": int(test_mask.sum()),
                    "rmse": float(np.sqrt(mean_squared_error(actual_salary, pred_salary))),
                    "mae": float(mean_absolute_error(actual_salary, pred_salary)),
                    "r2": float(r2_score(actual_salary, pred_salary)),
                }
            )
        if model_rows:
            result = pd.DataFrame(model_rows)
            all_results.append(result)
            avg_mae = result["mae"].mean()
            if avg_mae < best_score:
                best_score = avg_mae
                best_name = name

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = (
        results_df.groupby("model")[["rmse", "mae", "r2"]]
        .mean()
        .sort_values(["mae", "rmse"], ascending=[True, True])
        .reset_index()
    )

    latest_holdout = eval_seasons[-1]
    final_train_mask = forecast_df["target_season"] < latest_holdout
    final_test_mask = forecast_df["target_season"] == latest_holdout

    best_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", build_candidate_models()[best_name])]
    )
    best_pipeline.fit(X.loc[final_train_mask], y.loc[final_train_mask])
    holdout_pred = np.expm1(best_pipeline.predict(X.loc[final_test_mask]))

    holdout_df = forecast_df.loc[
        final_test_mask, ["Player", "Season", "target_season", "Salary", "next_salary"]
    ].copy()
    holdout_df["predicted_next_salary"] = holdout_pred
    holdout_df["absolute_error"] = (
        holdout_df["next_salary"] - holdout_df["predicted_next_salary"]
    ).abs()
    holdout_df["error_pct"] = np.where(
        holdout_df["next_salary"] > 0,
        holdout_df["absolute_error"] / holdout_df["next_salary"],
        np.nan,
    )

    holdout_X = X.loc[final_test_mask]
    holdout_y = y.loc[final_test_mask]
    sample_n = min(len(holdout_X), 180)
    sampled_X = holdout_X.sample(sample_n, random_state=42) if len(holdout_X) > sample_n else holdout_X
    sampled_y = holdout_y.loc[sampled_X.index]

    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        importance_values = np.abs(best_pipeline.named_steps["model"].feature_importances_)
    else:
        try:
            perm = permutation_importance(
                best_pipeline,
                sampled_X,
                sampled_y,
                n_repeats=5,
                random_state=42,
            )
            importance_values = np.abs(perm.importances_mean)
        except Exception:
            coef = getattr(best_pipeline.named_steps["model"], "coef_", np.zeros(len(feature_columns)))
            importance_values = np.abs(np.ravel(coef))

    if float(np.nanmax(importance_values)) == 0.0:
        corr_frame = forecast_df[feature_columns + ["next_salary"]].corr(numeric_only=True)["next_salary"]
        importance_values = corr_frame.reindex(feature_columns).fillna(0).abs().to_numpy()

    feature_importance_df = (
        pd.DataFrame({"feature": feature_columns, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    feature_importance_df["display_feature"] = feature_importance_df["feature"].map(format_feature_label)
    total_importance = float(feature_importance_df["importance"].sum())
    if total_importance > 0:
        feature_importance_df["importance_pct"] = (
            feature_importance_df["importance"] / total_importance * 100
        )
    else:
        feature_importance_df["importance_pct"] = 0.0

    return ModelArtifacts(
        forecast_df=forecast_df,
        feature_columns=feature_columns,
        summary_df=summary_df,
        results_df=results_df,
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        holdout_df=holdout_df,
        feature_importance_df=feature_importance_df,
    )


def build_artifact_metadata() -> dict[str, object]:
    files = []
    for name in ["player_stats.csv", "player_stats_model_ready.csv", "nba_salaries.csv"]:
        path = Path(name)
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(
                {
                    "name": name,
                    "sha256": digest,
                }
            )
    return {"version": APP_VERSION, "files": files, "has_xgboost": HAS_XGBOOST}


def load_saved_artifacts() -> ModelArtifacts | None:
    if not ARTIFACT_PATH.exists() or not METADATA_PATH.exists():
        return None
    try:
        # if the inputs or app version changed, rebuild instead of using stale artifacts
        saved_meta = joblib.load(METADATA_PATH)
        current_meta = build_artifact_metadata()
        if saved_meta != current_meta:
            return None
        return joblib.load(ARTIFACT_PATH)
    except Exception:
        return None


def save_artifacts(artifacts: ModelArtifacts) -> None:
    try:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        joblib.dump(artifacts, ARTIFACT_PATH)
        joblib.dump(build_artifact_metadata(), METADATA_PATH)
    except Exception:
        # streamlit cloud can be picky about pickling app objects, so don't fail the app here
        return


@st.cache_resource(show_spinner=True)
def get_dashboard_artifacts() -> ModelArtifacts:
    saved = load_saved_artifacts()
    if saved is not None:
        return saved
    # training is slower, so only do it when the saved files are missing or outdated
    artifacts = train_dashboard_model()
    save_artifacts(artifacts)
    return artifacts


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value / 1_000_000:,.1f}M"


def safe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[column]) or cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].astype(str)
    return cleaned


def render_plain_table(df: pd.DataFrame, height: int | None = None) -> None:
    cleaned = df.copy()
    cleaned.columns = [str(column) for column in cleaned.columns]
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].map(lambda value: "" if pd.isna(value) else str(value))

    table_html = cleaned.to_html(index=False, border=1)
    if height is None:
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.components.v1.html(table_html, height=height, scrolling=True)


def build_simulator_display_table(simulator_df: pd.DataFrame) -> pd.DataFrame:
    row = simulator_df.iloc[0]
    return (
        row.rename_axis("feature")
        .reset_index(name="value")
        .assign(feature=lambda df: df["feature"].map(format_feature_label))
    )


def build_simulator_input(artifacts: ModelArtifacts) -> pd.DataFrame:
    reference = artifacts.forecast_df.sort_values("Season").iloc[-1]
    row = reference[artifacts.feature_columns].copy()

    def assign_if_present(column: str, value: float) -> None:
        if column in row.index:
            row[column] = value

    current_salary_m = st.sidebar.slider(
        "Current salary (millions)",
        0.8,
        65.0,
        float(reference["Salary"] / 1_000_000),
        0.5,
    )
    age = st.sidebar.slider("Age", 19, 40, int(round(float(reference["AGE"]))))
    years_in_league = st.sidebar.slider(
        "Years in league",
        0,
        20,
        int(min(max(age - 19, 0), 20)),
    )
    games = st.sidebar.slider("Games played", 5, 82, int(round(float(reference["GP"]))))
    minutes = st.sidebar.slider("Minutes per game", 5.0, 40.0, float(reference["MIN"]), 0.5)
    points = st.sidebar.slider("Points per game", 0.0, 35.0, float(reference["PTS"]), 0.5)
    rebounds = st.sidebar.slider("Rebounds per game", 0.0, 18.0, float(reference["REB"]), 0.5)
    assists = st.sidebar.slider("Assists per game", 0.0, 12.0, float(reference["AST"]), 0.5)
    steals = st.sidebar.slider("Steals per game", 0.0, 3.5, float(reference["STL"]), 0.1)
    blocks = st.sidebar.slider("Blocks per game", 0.0, 4.0, float(reference["BLK"]), 0.1)
    turnovers = st.sidebar.slider("Turnovers per game", 0.0, 6.0, float(reference["TOV"]), 0.1)
    fg_pct = st.sidebar.slider("FG%", 0.25, 0.75, float(reference["FG_PCT"]), 0.01)
    fg3_pct = st.sidebar.slider("3PT%", 0.15, 0.50, float(reference["FG3_PCT"]), 0.01)
    ft_pct = st.sidebar.slider("FT%", 0.40, 0.98, float(reference["FT_PCT"]), 0.01)
    plus_minus = st.sidebar.slider("Plus/minus", -15.0, 15.0, float(reference["PLUS_MINUS"]), 0.5)
    double_doubles = st.sidebar.slider("Double-doubles", 0, 60, int(round(float(reference["DD2"]))))
    triple_doubles = st.sidebar.slider("Triple-doubles", 0, 35, int(round(float(reference["TD3"]))))

    estimated_fga = max(points / max((2 * fg_pct * 1.08), 0.35), 1.0)
    estimated_fg3a = max(estimated_fga * 0.38, 0.0)
    estimated_fg3m = estimated_fg3a * fg3_pct
    estimated_fta = max(points * 0.22, 0.0)
    estimated_ftm = estimated_fta * ft_pct
    estimated_fgm = max((points - estimated_ftm - (estimated_fg3m * 3)) / 2 + estimated_fg3m, 0.0)
    estimated_oreb = max(rebounds * 0.22, 0.0)
    estimated_dreb = max(rebounds - estimated_oreb, 0.0)
    estimated_pf = max(1.5 + (40 - age) * 0.02, 1.0)

    pts_per36 = 0 if minutes == 0 else points / minutes * 36
    reb_per36 = 0 if minutes == 0 else rebounds / minutes * 36
    ast_per36 = 0 if minutes == 0 else assists / minutes * 36
    stl_per36 = 0 if minutes == 0 else steals / minutes * 36
    blk_per36 = 0 if minutes == 0 else blocks / minutes * 36
    tov_per36 = 0 if minutes == 0 else turnovers / minutes * 36
    ast_tov_ratio = assists if turnovers == 0 else assists / turnovers
    fg3a_rate = 0 if estimated_fga == 0 else estimated_fg3a / estimated_fga
    fta_rate = 0 if estimated_fga == 0 else estimated_fta / estimated_fga
    ts_proxy = 0 if (estimated_fga + 0.44 * estimated_fta) == 0 else points / (
        2 * (estimated_fga + 0.44 * estimated_fta)
    )

    assign_if_present("Salary", current_salary_m * 1_000_000)
    assign_if_present("AGE", age)
    assign_if_present("seasons_played", years_in_league)
    assign_if_present("GP", games)
    assign_if_present("MIN", minutes)
    assign_if_present("FGM", estimated_fgm)
    assign_if_present("FGA", estimated_fga)
    assign_if_present("PTS", points)
    assign_if_present("FG3M", estimated_fg3m)
    assign_if_present("FG3A", estimated_fg3a)
    assign_if_present("REB", rebounds)
    assign_if_present("AST", assists)
    assign_if_present("STL", steals)
    assign_if_present("BLK", blocks)
    assign_if_present("TOV", turnovers)
    assign_if_present("FTM", estimated_ftm)
    assign_if_present("FTA", estimated_fta)
    assign_if_present("FG_PCT", fg_pct)
    assign_if_present("FG3_PCT", fg3_pct)
    assign_if_present("FT_PCT", ft_pct)
    assign_if_present("OREB", estimated_oreb)
    assign_if_present("DREB", estimated_dreb)
    assign_if_present("PLUS_MINUS", plus_minus)
    assign_if_present("DD2", double_doubles)
    assign_if_present("TD3", triple_doubles)
    assign_if_present("PF", estimated_pf)
    assign_if_present("PTS_per36", pts_per36)
    assign_if_present("REB_per36", reb_per36)
    assign_if_present("AST_per36", ast_per36)
    assign_if_present("STL_per36", stl_per36)
    assign_if_present("BLK_per36", blk_per36)
    assign_if_present("TOV_per36", tov_per36)
    assign_if_present("AST_TOV_ratio", ast_tov_ratio)
    assign_if_present("FG3A_rate", fg3a_rate)
    assign_if_present("FTA_rate", fta_rate)
    assign_if_present("TS_proxy", ts_proxy)
    assign_if_present("PLUS_MINUS_lag1", plus_minus)
    assign_if_present("DD2_lag1", double_doubles)
    assign_if_present("TD3_lag1", triple_doubles)
    assign_if_present("PTS_lag1", points)
    assign_if_present("REB_lag1", rebounds)
    assign_if_present("AST_lag1", assists)
    assign_if_present("STL_lag1", steals)
    assign_if_present("BLK_lag1", blocks)
    assign_if_present("TOV_lag1", turnovers)
    assign_if_present("MIN_lag1", minutes)
    assign_if_present("GP_lag1", games)
    assign_if_present("FGM_lag1", estimated_fgm)
    assign_if_present("FGA_lag1", estimated_fga)
    assign_if_present("FG3M_lag1", estimated_fg3m)
    assign_if_present("FG3A_lag1", estimated_fg3a)
    assign_if_present("FTM_lag1", estimated_ftm)
    assign_if_present("FTA_lag1", estimated_fta)
    assign_if_present("FG_PCT_lag1", fg_pct)
    assign_if_present("FG3_PCT_lag1", fg3_pct)
    assign_if_present("FT_PCT_lag1", ft_pct)
    assign_if_present("PTS_per36_lag1", pts_per36)
    assign_if_present("REB_per36_lag1", reb_per36)
    assign_if_present("AST_per36_lag1", ast_per36)
    assign_if_present("STL_per36_lag1", stl_per36)
    assign_if_present("BLK_per36_lag1", blk_per36)
    assign_if_present("TOV_per36_lag1", tov_per36)
    assign_if_present("AST_TOV_ratio_lag1", ast_tov_ratio)
    assign_if_present("FG3A_rate_lag1", fg3a_rate)
    assign_if_present("FTA_rate_lag1", fta_rate)
    assign_if_present("TS_proxy_lag1", ts_proxy)

    for feature in [
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "PLUS_MINUS",
        "DD2",
        "TD3",
        "PTS_per36",
        "REB_per36",
        "AST_per36",
        "STL_per36",
        "BLK_per36",
        "TOV_per36",
        "AST_TOV_ratio",
        "FG3A_rate",
        "FTA_rate",
        "TS_proxy",
    ]:
        rolling_col = f"{feature}_rolling3"
        if rolling_col in row.index:
            row[rolling_col] = row[feature]

    return pd.DataFrame([row])[artifacts.feature_columns]


def main() -> None:
    inject_styles()
    artifacts = get_dashboard_artifacts()

    st.markdown(
        """
        <div class="app-shell">
            <div class="app-kicker">nba salary forecast dashboard</div>
            <div class="app-title">NBA Salary Forecast Studio</div>
            <div class="app-copy">
                Explore a next-season salary model built from player performance, historical trend features,
                and time-aware validation. Use the controls to test player profiles, inspect model accuracy,
                and see where the forecast is strongest or weakest.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    holdout = artifacts.holdout_df.copy()
    latest_season = int(holdout["target_season"].max())
    holdout_mae = mean_absolute_error(holdout["next_salary"], holdout["predicted_next_salary"])
    holdout_rmse = np.sqrt(mean_squared_error(holdout["next_salary"], holdout["predicted_next_salary"]))
    holdout_r2 = r2_score(holdout["next_salary"], holdout["predicted_next_salary"])

    st.write(f"Best model: {artifacts.best_model_name}")
    st.write(f"Holdout season: {latest_season}")
    st.write(f"Holdout MAE: {format_currency(holdout_mae)}")
    st.write(f"Holdout RMSE: {format_currency(holdout_rmse)}")
    st.write(f"Latest holdout R²: {holdout_r2:.3f}")
    st.caption(
        "Latest holdout data is for the most recent test season only. "
    )

    st.sidebar.markdown("## Salary Simulator")
    st.sidebar.caption("Shape a player season and project next year's salary.")
    simulator_df = build_simulator_input(artifacts)
    simulator_prediction = float(
        np.expm1(artifacts.best_pipeline.predict(simulator_df)[0])
    )
    st.sidebar.write(f"Predicted next salary: {format_currency(simulator_prediction)}")

    tabs = st.tabs(["Forecast Lab", "Model Pulse", "Player Explorer"])

    with tabs[0]:
        st.markdown('<div class="section-title">Forecast Lab</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Dial in a player profile, then compare it to the model\'s most important signals.</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.subheader("Simulator Output")
            st.write(f"Projected next-year salary: {format_currency(simulator_prediction)}")
            render_plain_table(
                safe_for_streamlit(build_simulator_display_table(simulator_df)),
                height=360,
            )

        with right:
            importance_chart_df = artifacts.feature_importance_df.sort_values(
                "importance_pct", ascending=True
            ).copy()
            importance_chart = px.bar(
                importance_chart_df,
                x="importance_pct",
                y="display_feature",
                orientation="h",
                color_discrete_sequence=["#caa46a"],
            )
            importance_chart.update_layout(
                title="Top Model Signals",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font_color="#000000",
                height=360,
                margin=dict(l=24, r=10, t=50, b=10),
                xaxis_title="share of displayed importance (%)",
                yaxis_title="",
            )
            importance_chart.update_traces(marker_color="#000000", marker_line_color="#000000", marker_line_width=1)
            importance_chart.update_yaxes(automargin=True)
            importance_chart.update_xaxes(ticksuffix="%")
            st.plotly_chart(importance_chart, use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="section-title">Model Pulse</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Review how each model performed on recent chronological holdout seasons.</div>',
            unsafe_allow_html=True,
        )

        holdout_chart = holdout.copy()
        holdout_chart["next_salary_m"] = pd.to_numeric(holdout_chart["next_salary"], errors="coerce") / 1_000_000
        holdout_chart["predicted_next_salary_m"] = pd.to_numeric(
            holdout_chart["predicted_next_salary"], errors="coerce"
        ) / 1_000_000
        holdout_chart["absolute_error_m"] = pd.to_numeric(
            holdout_chart["absolute_error"], errors="coerce"
        ) / 1_000_000
        holdout_chart = holdout_chart.dropna(
            subset=["next_salary_m", "predicted_next_salary_m", "absolute_error_m"]
        ).copy()
        scatter = px.scatter(
            holdout_chart,
            x="next_salary_m",
            y="predicted_next_salary_m",
            hover_name="Player",
            color="absolute_error_m",
            color_continuous_scale=["#1d4ed8", "#60a5fa", "#f59e0b"],
            labels={
                "next_salary_m": "Actual next salary (M)",
                "predicted_next_salary_m": "Predicted next salary (M)",
                "absolute_error_m": "Absolute error (M)",
            },
        )
        max_salary = max(
            holdout_chart["next_salary_m"].max(), holdout_chart["predicted_next_salary_m"].max()
        )
        scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_salary,
            y1=max_salary,
            line=dict(color="#000000", dash="dash"),
        )
        scatter.update_layout(
            title=f"Actual vs Predicted Salary, {latest_season}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font_color="#000000",
            height=500,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_tickformat=",.2f",
            yaxis_tickformat=",.2f",
            coloraxis_colorbar=dict(tickformat=",.2f", ticksuffix=" M"),
        )
        scatter.update_traces(
            marker=dict(size=8, opacity=0.75),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Actual next salary: %{x:,.2f} M<br>"
                "Predicted next salary: %{y:,.2f} M<br>"
                "Absolute error: %{marker.color:,.2f} M<extra></extra>"
            )
        )
        scatter.update_xaxes(ticksuffix=" M")
        scatter.update_yaxes(ticksuffix=" M")
        st.plotly_chart(scatter, use_container_width=True)

        results_chart = artifacts.results_df.copy()
        results_chart["season"] = pd.to_numeric(results_chart["season"], errors="coerce")
        results_chart["mae_m"] = pd.to_numeric(results_chart["mae"], errors="coerce") / 1_000_000
        results_chart = results_chart.dropna(subset=["season", "mae_m"]).copy()
        season_lines = px.line(
            results_chart,
            x="season",
            y="mae_m",
            color="model",
            markers=True,
            labels={"mae_m": "MAE (M)", "season": "Target season"},
            color_discrete_sequence=["#2563eb", "#dc2626", "#059669", "#d97706"],
        )
        season_lines.update_layout(
            title="Validation Error by Season",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font_color="#000000",
            height=500,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis_tickformat=",.2f",
        )
        season_lines.update_traces(
            hovertemplate="Season %{x}<br>MAE: %{y:,.2f} M<extra>%{fullData.name}</extra>"
        )
        season_lines.update_yaxes(ticksuffix=" M")
        st.plotly_chart(season_lines, use_container_width=True)

        st.caption(
            "The table below shows average validation metrics across all target seasons. "
            "The holdout numbers at the top of the page are only for the latest holdout season."
        )
        st.caption(
            "Negative R² is possible. It means that, for that validation slice, the model did worse "
            "than predicting the average salary for every player."
        )
        render_plain_table(
            safe_for_streamlit(
                artifacts.summary_df.assign(
                    model=lambda df: df["model"],
                    rmse=lambda df: df["rmse"].map(format_currency),
                    mae=lambda df: df["mae"].map(format_currency),
                    r2=lambda df: df["r2"].map(lambda value: f"{value:.3f}"),
                ).rename(
                    columns={
                        "model": "model",
                        "rmse": "avg_rmse",
                        "mae": "avg_mae",
                        "r2": "avg_r2",
                    }
                )
            )
        )

        top_n = st.slider("Show biggest misses", 5, 30, 12, 1)
        misses = holdout.sort_values("absolute_error", ascending=False).head(top_n).copy()
        misses["actual"] = misses["next_salary"].map(format_currency)
        misses["predicted"] = misses["predicted_next_salary"].map(format_currency)
        misses["absolute_error"] = misses["absolute_error"].map(format_currency)
        misses["error_pct"] = misses["error_pct"].map(lambda value: f"{value:.0%}")
        render_plain_table(
            safe_for_streamlit(
                misses[["Player", "target_season", "actual", "predicted", "absolute_error", "error_pct"]]
            )
        )

        accurate_n = st.slider("Show most accurate predictions", 5, 30, 12, 1)
        accurate = holdout.sort_values("absolute_error", ascending=True).head(accurate_n).copy()
        accurate["actual"] = accurate["next_salary"].map(format_currency)
        accurate["predicted"] = accurate["predicted_next_salary"].map(format_currency)
        accurate["absolute_error"] = accurate["absolute_error"].map(format_currency)
        accurate["error_pct"] = accurate["error_pct"].map(lambda value: f"{value:.0%}")
        render_plain_table(
            safe_for_streamlit(
                accurate[["Player", "target_season", "actual", "predicted", "absolute_error", "error_pct"]]
            )
        )

    with tabs[2]:
        st.markdown('<div class="section-title">Player Explorer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Filter the holdout season to inspect who the model priced well and where it struggled.</div>',
            unsafe_allow_html=True,
        )

        filter_cols = st.columns(4)
        min_actual = holdout["next_salary"].min() / 1_000_000
        max_actual = holdout["next_salary"].max() / 1_000_000
        min_minutes = float(artifacts.forecast_df["MIN"].min())
        max_minutes = float(artifacts.forecast_df["MIN"].max())

        with filter_cols[0]:
            player_query = st.text_input("Player search", placeholder="Type a player name")
        with filter_cols[1]:
            salary_range = st.slider(
                "Actual next salary (millions)",
                float(min_actual),
                float(max_actual),
                (float(min_actual), float(max_actual)),
                0.5,
            )
        with filter_cols[2]:
            minutes_range = st.slider(
                "Minutes per game",
                min_minutes,
                max_minutes,
                (min_minutes, max_minutes),
                0.5,
            )
        with filter_cols[3]:
            sort_by = st.selectbox("Sort by", ["absolute_error", "predicted_next_salary", "next_salary", "Player"])

        explorer_df = holdout.merge(
            artifacts.forecast_df[["Player", "Season", "MIN", "PTS", "REB", "AST"]],
            on=["Player", "Season"],
            how="left",
        )
        explorer_df = explorer_df[
            explorer_df["next_salary"].between(salary_range[0] * 1_000_000, salary_range[1] * 1_000_000)
        ]
        explorer_df = explorer_df[explorer_df["MIN"].between(minutes_range[0], minutes_range[1])]
        if player_query:
            explorer_df = explorer_df[
                explorer_df["Player"].str.contains(player_query, case=False, na=False)
            ]
        explorer_df = explorer_df.sort_values(sort_by, ascending=(sort_by == "Player"))

        display_df = explorer_df.copy()
        display_df["current_salary"] = display_df["Salary"].map(format_currency)
        display_df["actual_next_salary"] = display_df["next_salary"].map(format_currency)
        display_df["predicted_next_salary"] = display_df["predicted_next_salary"].map(format_currency)
        display_df["absolute_error"] = display_df["absolute_error"].map(format_currency)

        render_plain_table(
            safe_for_streamlit(
                display_df[
                    [
                        "Player",
                        "Season",
                        "target_season",
                        "MIN",
                        "PTS",
                        "REB",
                        "AST",
                        "current_salary",
                        "actual_next_salary",
                        "predicted_next_salary",
                        "absolute_error",
                    ]
                ]
            )
        )


if __name__ == "__main__":
    main()
