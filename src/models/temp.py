# copy the script into the container
docker cp - api:/tmp/train.py <<'PY'
from src.data.preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import joblib

preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()

pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("model", RandomForestRegressor(n_estimators=10, random_state=42))
])
pipe.fit(X_train, y_train)

rmse = root_mean_squared_error(y_test, pipe.predict(X_test)) ** 0.5
print(f"RMSE on test set: {rmse:.2f}")

joblib.dump(pipe, "/app/models/bike_share_model.pkl")
print("Saved fitted pipeline to /app/models/bike_share_model.pkl")
PY

# run it inside the container
docker exec api python /tmp/train.py

# restart the API to reload the model
docker-compose -f docker-compose-simple.yml restart api



