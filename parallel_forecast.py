import time
import os  # To get process id
import psutil # CPU & memory tracking
import concurrent.futures
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for plots
sns.set_style("whitegrid")

forecast_period = 180

# Read the CSV file into a DataFrame
final_data = pd.read_csv("Processed_Data.csv")

# Convert the date column to datetime and set it as the index
final_data["Date"] = pd.to_datetime(final_data["Date"])
final_data.set_index("Date", inplace=True)

# Get the unique junctions from the data
junctions = final_data["Junction"].unique()

# List of junctions with no seasonality
no_seasonality_junctions = {3, 7, 10, 13}

# Tracking resource usage
cpu_usage = []
memory_usage = []
time_stamps = []

def monitor_resources():
    """Monitor CPU and memory usage"""
    while True:
        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().percent)
        time_stamps.append(time.time() - start_time)
        time.sleep(1)  # Record every second

def process_junction(junction):
    import matplotlib.pyplot as plt
    from pmdarima import auto_arima
    import pandas as pd
    import numpy as np
    import os  # To get the process id

    pid = os.getpid()  # Get the current process ID

    print(f"\nProcessing Junction {junction} in process ID {pid}...")

    # Filter data for this junction
    subset = final_data[final_data['Junction'] == junction].copy()
    subset = subset[['Vehicles']].dropna()  # Keep only the 'Vehicles' column
    subset.index = pd.to_datetime(subset.index)

    # Set seasonality based on junction
    m = None if junction in no_seasonality_junctions else 7
    print(f"Junction {junction}: Using Seasonality Period (m): {m if m is not None else 'No Seasonality'}")

    # Split data into train and test sets (80/20 split)
    train_size = int(len(subset) * 0.8)
    train, test = subset.iloc[:train_size], subset.iloc[train_size:]

    
    try:
        model = auto_arima(
            train['Vehicles'],
            seasonal=(m is not None),
            m=m if m is not None else 1,
            trace=False,
            suppress_warnings=True,
            stepwise=True,
            max_p=None, max_q=None, max_P=None, max_Q=None,
            d=None, D=None
        )

        # Extract model parameters (for informational purposes)
        p, d, q = model.order
        if m is not None:
            P, D, Q, s = model.seasonal_order
        else:
            P, D, Q, s = (0, 0, 0, 0)

        print(f"Junction {junction}: Selected SARIMA Model: (p,d,q)=({p},{d},{q}), (P,D,Q,s)=({P},{D},{Q},{s})")

        
        predictions = model.predict(n_periods=len(test))

        # Compute evaluation metrics
        mape = np.mean(np.abs((list(test['Vehicles']) - predictions) / list(test['Vehicles']))) * 100
        acc = 100 - mape

        print(f"Junction {junction}: MAPE: {mape:.2f}%, Accuracy: {acc:.2f}%")

        # Forecasting for next forecast_period days
        forecast = model.predict(n_periods=forecast_period)
        forecast_dates = pd.date_range(start=subset.index[-1], periods=forecast_period+1, freq='D')[1:]

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(subset.index, subset['Vehicles'], label="Actual", color="blue")
        plt.plot(forecast_dates, forecast, label="Forecast (next 6 months)", color="green")
        plt.title(f"Forecasting for Junction {junction} (Process ID: {pid})")
        plt.xlabel("Date")
        plt.ylabel("Vehicles Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Junction_{junction}_forecast_{pid}.png")
        plt.close()

        return f"Junction {junction} processed (Process ID: {pid})"
    except Exception as e:
        print(f"Junction {junction}: Model failed: {e}, skipping this junction.")
        return f"Junction {junction} failed"

if __name__ == "__main__":
    import threading
    
    # Start timing
    start_time = time.time()

    # Start resource monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    # Set the number of cores to use
    num_processes = 14

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_junction, junctions))

    # Stop monitoring
    monitor_thread.join(timeout=2)
    
    # Total execution time
    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

    # Print results from each junction
    for res in results:
        print(res)

    # Plot CPU & Memory Usage
    plt.figure(figsize=(12, 6))
    plt.plot(time_stamps, cpu_usage, label="CPU Usage (%)", color="blue")
    plt.plot(time_stamps, memory_usage, label="Memory Usage (%)", color="red")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Usage (%)")
    plt.title("CPU and Memory Usage Over Execution Time")
    plt.legend()
    plt.savefig("resource_usage.png")
    plt.show()