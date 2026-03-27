from pathlib import Path

import joblib

import app


def main() -> None:
    # run this once before deploying if you want faster first load
    artifacts = app.train_dashboard_model()
    app.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(artifacts, app.ARTIFACT_PATH)
    joblib.dump(app.build_artifact_metadata(), app.METADATA_PATH)
    print(f"saved artifacts to {Path(app.ARTIFACT_PATH).resolve()}")


if __name__ == "__main__":
    main()
