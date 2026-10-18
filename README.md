# Residential Energy Consumption Forecasting Using Transformer Self-Attention

A deep learning mini-project that forecasts residential electricity consumption using a Transformer encoder with a self-attention mechanism.

## 1. Project Overview

Residential electricity consumption varies over time due to household activities and appliance usage. Forecasting future consumption from historical readings can help understand usage patterns and support energy planning.

This project uses historical household electricity measurements to predict the next hour’s active power consumption. A Transformer-based self-attention model learns temporal relationships from the previous 24 hours of data.

## 2. Objectives

- Prepare household electricity data for time-series forecasting.
- Convert minute-level measurements into hourly readings.
- Build a Transformer encoder using self-attention.
- Predict the next hour’s residential active power consumption.
- Evaluate the model using regression metrics.
- Compare the Transformer with a simple persistence baseline.

## 3. Dataset

**Dataset:** Individual Household Electric Power Consumption  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains measurements from a household in France, recorded approximately every minute over several years. It includes active power, reactive power, voltage, current intensity, and three sub-metering measurements.

### Target variable

`Global_active_power` — household active power, measured in kilowatts (kW).

### Download and setup

1. Download the dataset from the UCI repository.
2. Extract `household_power_consumption.txt`.
3. Place the file in `data/raw/`.

The raw dataset is not included in this Git repository. Download it separately before running the pipeline.

## 4. Project Workflow

1. Dataset collection
2. Data cleaning and missing-value handling
3. Timestamp creation from date and time
4. Resampling minute-level readings into hourly averages
5. Checking for missing hourly intervals
6. Creating 24-hour input sequences
7. Chronological training, validation, and test split
8. Feature standardization using training data
9. Transformer encoder training
10. Next-hour active power prediction
11. Evaluation against a persistence baseline
12. Actual-versus-predicted visualization

## 5. Model Architecture

The forecasting model is a Transformer encoder implemented using PyTorch.

- **Input:** Previous 24 hourly observations, each containing 7 features
- **Input shape:** `(24, 7)` per sample
- **Input projection:** Maps the seven features into a 64-dimensional representation
- **Positional embeddings:** Represent the order of the hourly time steps
- **Transformer encoder:** Two encoder layers with four attention heads
- **Feed-forward dimension:** 128
- **Output:** Predicted `Global_active_power` for the next hour

The self-attention mechanism allows the model to learn relationships between different time steps within the 24-hour history.

## 6. Training and Evaluation

The prepared sequences are divided chronologically:

| Split | Samples |
|---|---:|
| Training | 23,766 |
| Validation | 5,093 |
| Testing | 5,093 |
| **Total** | **33,952** |

The model is trained using mean squared error loss and the AdamW optimizer. Validation loss is used for early stopping and selecting the best model.

### Evaluation metrics

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error

## 7. Results

The model was evaluated on the held-out test set and compared with a persistence baseline, which predicts the next hour’s power as the most recently observed hour.

| Metric | Transformer | Persistence baseline |
|---|---:|---:|
| MAE | **0.3121 kW** | 0.3866 kW |
| RMSE | **0.4657 kW** | 0.5846 kW |
| MAPE | **37.18%** | 46.35% |

The Transformer achieved lower MAE and RMSE than the persistence baseline. The prediction plots show that it captures broad consumption patterns, although some sharp peaks are underestimated.

MAPE should be interpreted cautiously because percentage errors can become large when actual consumption is close to zero.

### Result visualizations

The generated plots are available in the `plots/` directory:

- `test_week_predictions.png` — actual consumption compared with Transformer and persistence predictions over a test-week window.
- `actual_vs_predicted.png` — scatter plot comparing actual values with Transformer predictions.

## 8. How to Run the Project

Run the following commands from the project root in Windows CMD.

### Create and activate a virtual environment

```cmd
py -3.13 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install dependencies

```cmd
pip install -r requirements.txt
```

The project was developed with PyTorch and CUDA support on an NVIDIA RTX 3050. If CUDA is unavailable in another environment, PyTorch can run the model on CPU, though training may be slower.

### Run the pipeline

After placing the downloaded dataset in `data/raw/`, run:

```cmd
python src\preprocess.py
python src\prepare_sequences.py
python src\train_transformer.py
python src\plot_results.py
```

The scripts generate the hourly dataset, forecasting sequences, trained model, test predictions, and result plots.

## 9. Project Structure

```text
residential-energy-forecasting/
│
├── data/
│   ├── raw/                 # Downloaded dataset (not tracked by Git)
│   └── processed/           # Generated hourly data and sequences
│
├── notebooks/               # Exploratory notebooks, if used
├── src/
│   ├── preprocess.py        # Clean data and create hourly readings
│   ├── prepare_sequences.py # Generate time-series input/target sequences
│   ├── train_transformer.py # Train and evaluate Transformer
│   └── plot_results.py      # Create evaluation plots
│
├── plots/                   # Result visualizations
├── models/                  # Saved trained model (generated locally)
├── tests/                   # Project tests
│
├── .gitignore
├── README.md
└── requirements.txt
```

Generated data and model files may be excluded from Git. They can be recreated by downloading the dataset and running the pipeline.

## 10. Limitations

- The dataset represents a single household in France and may not generalize to other households or regions.
- The current model forecasts only one hour ahead.
- Some sudden consumption peaks are underestimated.
- MAPE can be misleading when actual power values are very low.
- The model’s performance depends on the available historical data and selected input window.

## 11. Future Scope

- Forecast multiple future hours.
- Experiment with longer input windows.
- Compare the Transformer with LSTM, GRU, and other forecasting models.
- Improve peak-consumption prediction.
- Evaluate the model on data from more households or regions.
- Develop a simple interface for entering historical readings and viewing forecasts.

## 12. Technologies Used

- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Transformer encoder and self-attention
- Git and GitHub

## 13. Conclusion

This project demonstrates residential electricity consumption forecasting using a Transformer-based self-attention model. The model uses the previous 24 hours of household measurements to predict the next hour’s active power. On the test set, it outperformed a persistence baseline in MAE and RMSE, demonstrating the potential of attention-based models for short-term energy forecasting.