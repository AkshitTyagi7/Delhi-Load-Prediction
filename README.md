# Energy Consumption and Peak Load Prediction

This project predicts **Energy Consumption (MU)** and **Peak Load (MW)** for Delhi using weather data and public holiday information. Predictions are generated using machine learning models, specifically **Random Forest Regressors**, trained on historical data from 2021 to 2024.

## Overview

The project forecasts daily **Energy Consumption** and **Peak Load** values based on the following features:
- **Weather Parameters**: Temperature (min, max, avg), wind speed, wind direction, and precipitation.
- **Holidays**: Public holidays and Sundays are flagged and used as input to the model.

## Dataset

The dataset contains historical data for energy consumption and peak load, combined with weather data. Predictions are made for August 2023, based on trends from previous years.

## Predictions

Two models were created:
1. **Energy Consumption Model**: Predicts daily energy consumption in MU.
2. **Peak Load Model**: Predicts the daily peak load in MW.

Predictions were made using Random Forest and fitted to follow the actual consumption/load trends with a ±10% variation.

## Installation and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
