# AI Electricity Bill Analyzer

A Flask based machine learning project that predicts electricity units and bill amount from Yes/No appliance usage inputs.

## Features

- User registration and login.
- Yes/No input for TV, washing machine, refrigerator, AC, fan, LED bulbs, heater, and laptop.
- AI prediction for units consumed and bill amount using Random Forest Regression.
- Next month usage and bill estimate based on recent prediction history.
- High bill alert when the bill amount is above Rs. 1000.
- Simple electricity saving recommendation.
- User-wise prediction history stored in `data/prediction_history.csv`.
- Login data stored in SQLite at `data/app.db`.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Register a new account first, then login and analyze the electricity bill.

## Dataset Format

Place your dataset at:

```text
data/electricity_dataset.csv
```

Required columns:

```text
TV,Washing_Machine,Refrigerator,AC,Fan,LED_Bulbs,Heater,Laptop,Units_Consumed,Bill_Amount
```

Use `Yes` and `No` values for appliance columns.
