# hybrid_gwo_sarima_bilstm_spawn_safe.py
"""
Hybrid SARIMA + BiLSTM pipeline with:
 - Grey Wolf Optimizer (GWO) for ARIMA/SARIMA parameter tuning
 - Spawn-safe multiprocessing (suitable for TensorFlow)
 - Per-process TensorFlow imports + GPU memory growth config
 - Resource monitoring in main process
"""

# ========================
# IMPORTANT: set spawn BEFORE importing heavy native libs
# ========================
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# Now import light / general libs
import os
import time
import threading
import psutil
import math
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ------------------------
# Configuration
# ------------------------
DATA_PATH = "Processed_Data.csv"   # <-- change if needed
FORECAST_PERIOD = 180
NO_SEASONALITY_JUNCTIONS = {3, 7, 10, 13}
GWO_POPULATION = 8      # conservative default, increase for better search
GWO_ITERS = 20          # conservative default
GWO_SEED = 42
NUM_PROCESSES = min(6, max(1, os.cpu_count() - 1))  # keep some cores free

# ------------------------
# Utility: create supervised sequences for LSTM
# ------------------------
def create_dataset(data_array, time_steps=1):
    X, y = [], []
    for i in range(len(data_array) - time_steps):
        X.append(data_array[i:(i + time_steps), 0])
        y.append(data_array[i + time_steps, 0])
    return np.array(X), np.array(y)


# ========================
# GWO + ARIMA (SARIMAX) implementation
# ========================
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ArimaOptimization:
    """
    Callable fitness wrapper:
    - When use_seasonality is True, expects params: [p,d,q,P,D,Q]
    - When False, expects params: [p,d,q]
    Uses train_data (pd.Series) and evaluates on test_data (pd.Series)
    Minimizes MAPE.
    """
    def __init__(self, train_data, test_data, use_seasonality=True, m=7):
        self.train_data = train_data.astype(float)
        self.test_data = test_data.astype(float)
        self.use_seasonality = use_seasonality
        self.m = int(m) if (use_seasonality and (m is not None)) else None
        self.best_fitness = np.inf
        self.best_params = None
        self.best_model = None

    def __call__(self, params):
        # params may be floats -> round and cast to ints
        try:
            params_int = [int(max(0, round(p))) for p in params]
            if self.use_seasonality:
                if len(params_int) < 6:
                    return 1e9
                p, d, q, P, D, Q = params_int[:6]
                # enforce minimal valid orders
                seasonal_order = (P, D, Q, self.m) if (self.m and self.m > 1) else None
            else:
                if len(params_int) < 3:
                    return 1e9
                p, d, q = params_int[:3]
                seasonal_order = None

            # Build SARIMAX with/without seasonal_order
            if seasonal_order:
                model = SARIMAX(self.train_data,
                                order=(p, d, q),
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            else:
                model = SARIMAX(self.train_data,
                                order=(p, d, q),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

            fitted = model.fit(disp=False, method='lbfgs', maxiter=200)
            # Forecast for test horizon
            forecast = fitted.get_forecast(steps=len(self.test_data)).predicted_mean
            # Align lengths and compute MAPE robustly (avoid zeros)
            y_true = np.asarray(self.test_data).astype(float)
            y_pred = np.asarray(forecast).astype(float)
            # If any actuals are zero, add small epsilon to denominator
            eps = 1e-8
            mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

            # Keep best
            if mape < self.best_fitness:
                self.best_fitness = float(mape)
                self.best_params = {
                    "p": int(p),
                    "d": int(d),
                    "q": int(q),
                    "P": int(P) if seasonal_order else 0,
                    "D": int(D) if seasonal_order else 0,
                    "Q": int(Q) if seasonal_order else 0,
                    "m": int(self.m) if seasonal_order else None
                }
                self.best_model = fitted
            return float(mape)
        except Exception:
            # If model failed, return a very large fitness so it is ignored
            return 1e9


class GreyWolfOptimizer:
    """
    Simple Grey Wolf Optimizer implementation.
    - fitness_fn: callable that accepts a vector and returns scalar fitness (lower is better)
    - dim: dimensionality of solution
    - lower_bounds, upper_bounds: lists/arrays of same length
    """

    def __init__(self, population_size=8, seed=None):
        self.population_size = int(population_size)
        if seed is not None:
            np.random.seed(int(seed))

    def run(self, fitness_fn, dim, lower_bounds, upper_bounds, max_iters=20):
        lb = np.array(lower_bounds, dtype=float)
        ub = np.array(upper_bounds, dtype=float)
        # Initialize population uniformly
        wolves = np.random.uniform(lb, ub, (self.population_size, dim))

        # Evaluate initial population
        fitnesses = np.array([fitness_fn(w) for w in wolves], dtype=float)

        # Sort by fitness ascending
        idx_sorted = np.argsort(fitnesses)
        wolves = wolves[idx_sorted]
        fitnesses = fitnesses[idx_sorted]

        # Initialize alpha, beta, delta
        alpha = wolves[0].copy()
        beta = wolves[1].copy() if self.population_size > 1 else wolves[0].copy()
        delta = wolves[2].copy() if self.population_size > 2 else wolves[0].copy()
        alpha_score = fitnesses[0]
        beta_score = fitnesses[1] if self.population_size > 1 else fitnesses[0]
        delta_score = fitnesses[2] if self.population_size > 2 else fitnesses[0]

        # Iterations
        for t in range(max_iters):
            a = 2 - 2 * (t / max_iters)  # linearly decreasing from 2 to 0
            for i in range(self.population_size):
                for j in range(dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolves[i][j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolves[i][j])
                    X3 = delta[j] - A3 * D_delta

                    wolves[i][j] = np.clip((X1 + X2 + X3) / 3.0, lb[j], ub[j])

            # Evaluate population and update alpha,beta,delta
            fitnesses = np.array([fitness_fn(w) for w in wolves], dtype=float)
            idx_sorted = np.argsort(fitnesses)
            wolves = wolves[idx_sorted]
            fitnesses = fitnesses[idx_sorted]
            # update leaders
            alpha = wolves[0].copy()
            beta = wolves[1].copy() if self.population_size > 1 else wolves[0].copy()
            delta = wolves[2].copy() if self.population_size > 2 else wolves[0].copy()
            alpha_score = fitnesses[0]
            beta_score = fitnesses[1] if self.population_size > 1 else fitnesses[0]
            delta_score = fitnesses[2] if self.population_size > 2 else fitnesses[0]

        # Return the best params dict and best fitness (obtained from fitness function wrapper)
        # We assume fitness_fn (an ArimaOptimization instance) stored best_params and best_fitness
        # after being called during search.
        if hasattr(fitness_fn, "best_params") and fitness_fn.best_params is not None:
            return fitness_fn.best_params, float(fitness_fn.best_fitness)
        else:
            # Fallback: return array + alpha_score
            return {"params_vector": alpha.tolist()}, float(alpha_score)


# ========================
# Worker: process one junction
# ========================
def process_junction_worker(junction, data_path=DATA_PATH,
                            forecast_period=FORECAST_PERIOD,
                            no_seasonality=NO_SEASONALITY_JUNCTIONS,
                            gwo_population=GWO_POPULATION,
                            gwo_iters=GWO_ITERS,
                            gwo_seed=GWO_SEED):
    """
    This function runs in spawned subprocesses.
    It reads the CSV locally, fits SARIMAX tuned via GWO, computes residuals,
    trains BiLSTM on residuals, combines predictions, and returns a summary dict.
    """
    # Import TensorFlow inside the spawned worker and configure GPU growth
    try:
        # Reduce TF verbosity
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

        # Configure memory growth for GPUs present (prevents full allocation)
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    except Exception as e:
        return {
            "Junction": int(junction),
            "Status": "TFImportFailed",
            "Error": str(e)
        }

    pid = os.getpid()
    start_time_local = time.time()
    print(f"[PID {pid}] Starting Junction {junction}")

    # Load data locally to avoid pickling large DataFrame across processes
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"Junction": int(junction), "Status": "ReadFailed", "Error": str(e)}

    # Preprocess same as main: ensure Date index and Vehicles column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    else:
        # assume index is already date-like; fallback
        df.index = pd.to_datetime(df.index)

    # Filter junction and keep Vehicles
    subset = df[df["Junction"] == junction][["Vehicles"]].dropna()
    if subset.shape[0] < 30:
        return {"Junction": int(junction), "Status": "TooFewRows", "Rows": int(subset.shape[0])}

    subset.index = pd.to_datetime(subset.index).sort_values()
    use_seasonality = junction not in no_seasonality
    m = 7 if use_seasonality else None

    # Train-test split
    train_size = int(len(subset) * 0.8)
    train = subset.iloc[:train_size]["Vehicles"]
    test = subset.iloc[train_size:]["Vehicles"]

    # --------- Step 1: GWO optimization of ARIMA/SARIMA ----------
    fitness_wrapper = ArimaOptimization(train, test, use_seasonality, m if m is not None else 1)
    dim = 6 if use_seasonality else 3
    lower_bounds = [0] * dim
    # Conservative upper bounds; users can change if they want larger p/q
    default_upper = [5, 2, 5, 2, 1, 2]
    upper_bounds = default_upper[:dim]

    gwo = GreyWolfOptimizer(population_size=gwo_population, seed=gwo_seed)
    try:
        best_params_dict, best_mape = gwo.run(fitness_wrapper, dim, lower_bounds, upper_bounds, max_iters=gwo_iters)
    except Exception as e:
        # If GWO fails for some reason, fallback path will be used below
        best_params_dict, best_mape = None, None

    # If optimizer failed to produce a fitted model, try fallback: simple AR(1) persistence or auto_arima if available
    arima_fitted = None
    if hasattr(fitness_wrapper, "best_model") and fitness_wrapper.best_model is not None:
        arima_fitted = fitness_wrapper.best_model
        selected_order = fitness_wrapper.best_params
    else:
        # Fallback: try quick SARIMAX fit with auto small orders
        try:
            fallback_order = (1, 0, 1)
            if use_seasonality and m and m > 1:
                arima_fitted = SARIMAX(train, order=fallback_order,
                                       seasonal_order=(1, 0, 1, m),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit(disp=False)
                selected_order = {"p":1,"d":0,"q":1,"P":1,"D":0,"Q":1,"m":m}
            else:
                arima_fitted = SARIMAX(train, order=fallback_order,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit(disp=False)
                selected_order = {"p":1,"d":0,"q":1,"P":0,"D":0,"Q":0,"m":None}
        except Exception:
            # last resort: persistence (use last value)
            arima_fitted = None
            selected_order = None

    # Generate in-sample and test predictions for ARIMA
    if arima_fitted is not None:
        try:
            # in-sample predicted mean aligned to train index
            arima_train_pred = arima_fitted.get_prediction(start=train.index[0], end=train.index[-1]).predicted_mean
        except Exception:
            # if get_prediction fails, fallback to fitted.fittedvalues if available
            try:
                arima_train_pred = pd.Series(arima_fitted.fittedvalues, index=train.index)
            except Exception:
                arima_train_pred = pd.Series(train.values, index=train.index)

        try:
            arima_test_pred = pd.Series(arima_fitted.get_forecast(steps=len(test)).predicted_mean, index=test.index)
        except Exception:
            arima_test_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    else:
        # persistence
        arima_train_pred = pd.Series(train.values, index=train.index)
        arima_test_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)

    # --------- Step 2: Residuals ----------
    train_residuals = (train.values.astype(float) - np.asarray(arima_train_pred).astype(float))
    test_residuals = (test.values.astype(float) - np.asarray(arima_test_pred).astype(float))

    # Convert to DataFrame for scaler compatibility
    train_res_df = pd.DataFrame(train_residuals, index=train.index, columns=["Residuals"])
    test_res_df = pd.DataFrame(test_residuals, index=test.index, columns=["Residuals"])

    # --------- Step 3: Train BiLSTM on residuals ----------
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_res_df.values)
    test_scaled = scaler.transform(test_res_df.values)

    time_steps = 7 if use_seasonality else 1

    # If insufficient length for LSTM sequences, skip LSTM and return ARIMA-only forecast
    if len(train_scaled) <= time_steps or len(test_scaled) <= time_steps:
        # compute ARIMA-only MAPE for aligned test portion
        # aligned means we compare values from index time_steps onward
        aligned_arima_test = arima_test_pred.values[time_steps:]
        aligned_actual = test.values[time_steps:]
        if len(aligned_actual) == 0:
            arima_mape = float('nan')
        else:
            arima_mape = mean_absolute_percentage_error(aligned_actual, aligned_arima_test) * 100
        # compute forecast (ARIMA forecast)
        try:
            arima_forecast = arima_fitted.get_forecast(steps=forecast_period).predicted_mean if arima_fitted is not None else np.full(forecast_period, train.iloc[-1])
        except Exception:
            arima_forecast = np.full(forecast_period, train.iloc[-1])
        # Build hybrid_forecast = arima_forecast (no residuals)
        hybrid_forecast = np.asarray(arima_forecast)

        # Save quick plot
        forecast_dates = pd.date_range(start=subset.index[-1], periods=forecast_period + 1, freq='D')[1:]
        plt.figure(figsize=(12, 6))
        plt.plot(subset.index, subset["Vehicles"], label="Historical")
        plt.plot(forecast_dates, hybrid_forecast, label="ARIMA-only Forecast")
        plt.title(f"Junction {junction} ARIMA-only Forecast (PID {pid})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Junction_{junction}_arima_only_{pid}.png")
        plt.close()

        return {
            "Junction": int(junction),
            "PID": pid,
            "Status": "ARIMA-only",
            "ARIMA_MAPE": float(arima_mape),
            "Hybrid_MAPE": None,
            "ARIMA_Order": selected_order,
            "TimeSec": time.time() - start_time_local
        }

    # Prepare LSTM datasets
    X_train, y_train = create_dataset(train_scaled, time_steps)
    X_test, y_test = create_dataset(test_scaled, time_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build BiLSTM model
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

    lstm_model = Sequential([
        Bidirectional(LSTM(50, return_sequences=True), input_shape=(time_steps, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")

    # Fit LSTM (quiet)
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Predict residuals and invert scaling
    lstm_train_pred_scaled = lstm_model.predict(X_train)
    lstm_test_pred_scaled = lstm_model.predict(X_test)
    try:
        lstm_train_pred = scaler.inverse_transform(lstm_train_pred_scaled).flatten()
        lstm_test_pred = scaler.inverse_transform(lstm_test_pred_scaled).flatten()
    except Exception:
        # If inverse_transform fails, assume zero residuals
        lstm_train_pred = np.zeros(len(lstm_train_pred_scaled)).flatten()
        lstm_test_pred = np.zeros(len(lstm_test_pred_scaled)).flatten()

    # --------- Step 4: Combine ARIMA + LSTM residuals ----------
    # Align ARIMA predictions (they include the initial time_steps)
    arima_train_aligned = np.asarray(arima_train_pred).astype(float)[time_steps:]
    arima_test_aligned = np.asarray(arima_test_pred).astype(float)[time_steps:]

    hybrid_train_pred = arima_train_aligned + lstm_train_pred
    hybrid_test_pred = arima_test_aligned + lstm_test_pred

    # Actual aligned
    train_actual_aligned = train.values[time_steps:]
    test_actual_aligned = test.values[time_steps:]

    # Metrics
    try:
        arima_mape = mean_absolute_percentage_error(test_actual_aligned, arima_test_aligned) * 100
    except Exception:
        arima_mape = float('nan')
    try:
        hybrid_mape = mean_absolute_percentage_error(test_actual_aligned, hybrid_test_pred) * 100
    except Exception:
        hybrid_mape = float('nan')
    hybrid_acc = None if math.isnan(hybrid_mape) else (100.0 - hybrid_mape)

    # --------- Step 5: Forecast future horizon ----------
    # ARIMA forecast
    try:
        arima_forecast_vals = np.asarray(arima_fitted.get_forecast(steps=forecast_period).predicted_mean)
    except Exception:
        arima_forecast_vals = np.full(forecast_period, train.iloc[-1])

    # LSTM iterative forecasting on residuals
    last_seq_scaled = test_scaled[-time_steps:].reshape(1, time_steps, 1).copy()
    seq = last_seq_scaled.copy()
    lstm_forecasts_scaled = []
    for _ in range(forecast_period):
        nxt = lstm_model.predict(seq)
        lstm_forecasts_scaled.append(nxt[0, 0])
        # append nxt and drop first
        seq = np.append(seq[:, 1:, :], np.array([[[nxt[0, 0]]]]), axis=1)

    lstm_forecasts_scaled = np.array(lstm_forecasts_scaled).reshape(-1, 1)
    try:
        lstm_forecasts_inv = scaler.inverse_transform(lstm_forecasts_scaled).flatten()
    except Exception:
        lstm_forecasts_inv = np.zeros(forecast_period)

    hybrid_forecast = arima_forecast_vals + lstm_forecasts_inv

    # Save figures: historical + hybrid test predictions + hybrid forecast
    forecast_dates = pd.date_range(start=subset.index[-1], periods=forecast_period + 1, freq="D")[1:]
    plt.figure(figsize=(14, 7))
    plt.plot(subset.index, subset["Vehicles"], label="Historical", linewidth=1.2)
    # aligned test indices for hybrid predictions
    aligned_test_idx = test.index[time_steps: time_steps + len(hybrid_test_pred)]
    plt.plot(aligned_test_idx, hybrid_test_pred, label="Hybrid Test Predictions", linestyle='--')
    plt.plot(test.index, arima_test_pred.values, label="ARIMA Test Predictions", alpha=0.6)
    plt.plot(forecast_dates, hybrid_forecast, label="Hybrid Forecast (next {} days)".format(forecast_period), color="green")
    plt.title(f"Hybrid SARIMA + BiLSTM - Junction {junction} (PID {pid})")
    plt.xlabel("Date")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.tight_layout()
    fname = f"Junction_{junction}_hybrid_{pid}.png"
    plt.savefig(fname)
    plt.close()

    # Return summary
    return {
        "Junction": int(junction),
        "PID": pid,
        "Status": "OK",
        "ARIMA_Order": selected_order,
        "GWO_MAPE": float(best_mape) if best_mape is not None else None,
        "ARIMA_MAPE": float(arima_mape),
        "Hybrid_MAPE": float(hybrid_mape) if not math.isnan(hybrid_mape) else None,
        "Hybrid_Accuracy": float(hybrid_acc) if hybrid_acc is not None else None,
        "TimeSec": time.time() - start_time_local
    }


# ========================
# Main: orchestrate multiprocessing + resource monitoring
# ========================
def monitor_resources_thread(start_time, cpu_list, mem_list, ts_list, stop_event):
    while not stop_event.is_set():
        cpu_list.append(psutil.cpu_percent())
        mem_list.append(psutil.virtual_memory().percent)
        ts_list.append(time.time() - start_time)
        time.sleep(1)


def main():
    # Read data minimally here just to get junction list without loading full df into child processes
    try:
        df_main = pd.read_csv(DATA_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to read data file {DATA_PATH}: {e}")

    if "Date" in df_main.columns:
        df_main["Date"] = pd.to_datetime(df_main["Date"])
        df_main.set_index("Date", inplace=True, drop=False)

    junctions = sorted(df_main["Junction"].unique().tolist())

    # Resource monitoring containers
    cpu_list, mem_list, ts_list = [], [], []
    stop_event = threading.Event()
    start_time = time.time()
    monitor_thread = threading.Thread(target=monitor_resources_thread, args=(start_time, cpu_list, mem_list, ts_list, stop_event), daemon=True)
    monitor_thread.start()

    print(f"Launching processing for {len(junctions)} junctions using {NUM_PROCESSES} processes...")
    results = []
    # Use ProcessPoolExecutor with spawn-safe environment (we already set spawn)
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # map with args: use starmap-like pattern by passing tuples and wrapper
        futures = []
        for jun in junctions:
            futures.append(executor.submit(process_junction_worker, jun))

        # collect results as they finish
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                res = {"Junction": None, "Status": "FailedInWorker", "Error": str(e)}
            results.append(res)
            print("Collected:", res)

    # Stop resource monitoring
    stop_event.set()
    # allow monitor thread to finish quickly
    time.sleep(0.5)

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")

    # Save summary CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("hybrid_results_summary.csv", index=False)
    print("Saved hybrid_results_summary.csv")

    # Plot resource usage if data collected
    if len(ts_list) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(ts_list, cpu_list, label="CPU Usage (%)")
        plt.plot(ts_list, mem_list, label="Memory Usage (%)")
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Usage (%)")
        plt.legend()
        plt.title("CPU & Memory Usage During Run")
        plt.tight_layout()
        plt.savefig("resource_usage.png")
        plt.show()
    else:
        print("No resource monitoring data collected.")

    print("Done.")


if __name__ == "__main__":
    main()
