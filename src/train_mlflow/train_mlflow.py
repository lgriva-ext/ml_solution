import pandas as pd
import mlflow

registry_uri = f"databricks://modelregistery:modelregistery"
mlflow.set_registry_uri(registry_uri)
mlflow.set_experiment("/Shared//test_training")

wind_farm_data = pd.read_csv(
    "https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv",
    index_col=0,
)


def get_training_data():
    training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
    X = training_data.drop(columns="power")
    y = training_data["power"]
    return X, y


def get_validation_data():
    validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
    X = validation_data.drop(columns="power")
    y = validation_data["power"]
    return X, y


def get_weather_and_forecast():
    format_date = lambda pd_date: pd_date.date().strftime("%Y-%m-%d")
    today = pd.Timestamp("today").normalize()
    week_ago = today - pd.Timedelta(days=5)
    week_later = today + pd.Timedelta(days=5)

    past_power_output = pd.DataFrame(wind_farm_data)[
        format_date(week_ago) : format_date(today)
    ]
    weather_and_forecast = pd.DataFrame(wind_farm_data)[
        format_date(week_ago) : format_date(week_later)
    ]
    if len(weather_and_forecast) < 10:
        past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
        weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10:]

    return weather_and_forecast.drop(columns="power"), past_power_output["power"]


def train_keras_model(X, y):
    import tensorflow.keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(
        Dense(
            100,
            input_shape=(X_train.shape[-1],),
            activation="relu",
            name="hidden_layer",
        )
    )
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)
    return model


X_train, y_train = get_training_data()

with mlflow.start_run():
    # Automatically capture the model's parameters, metrics, artifacts,
    # and source code with the `autolog()` function
    mlflow.tensorflow.autolog()

    train_keras_model(X_train, y_train)
    run_id = mlflow.active_run().info.run_id

model_name = "power-forecasting-model"
artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(
    run_id=run_id, artifact_path=artifact_path
)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
