# 📊 DevelopersHub Corporation - AI/ML Engineering Internship Tasks

## 📁 Task 1: Exploring and Visualizing a Simple Dataset (Iris Dataset)

### 🎯 Task Objective
Perform Exploratory Data Analysis (EDA) on the Iris dataset to understand data structure, distributions, feature relationships, and detect outliers.

### 📂 Dataset Used
- **Source:** UCI Machine Learning Repository (via Kaggle)
- **URL:** https://www.kaggle.com/datasets/uciml/iris
- **Size:** 150 rows, 5 columns
- **Features:** sepal\_length, sepal\_width, petal\_length, petal\_width
- **Target:** species (Setosa, Versicolor, Virginica)

### 🧠 Models Applied
No models were applied. This is purely EDA using:
- `pandas` for data inspection
- `matplotlib` & `seaborn` for visualizations (scatter plots, histograms, box plots)

### 🔍 Key Results and Findings
- **Data Integrity:** The dataset is clean with 150 rows and 5 columns. No missing values were found.
- **Species Separation:** Setosa is completely separable from Versicolor and Virginica. However, Versicolor and Virginica show some overlap and cannot be perfectly separated.
- **Best Features:** Petal length and petal width are the most effective features for distinguishing between the three species. Sepal measurements are less discriminative.
- **Outliers:** Outliers were only detected in sepal\_width, with four values identified: 4.4, 4.1, 4.2, and 2.0.
- **Distributions:** Histograms show clear separation for petal measurements across all three species, while sepal measurements show more overlap.

### ✅ Conclusion 
The Iris dataset is clean and well-structured, with petal measurements being the best predictors for species classification. Only sepal\_width contains outliers that may need attention.

---

## 📈 Task 2: Predict Future Stock Prices (Short-Term)

### 🎯 Task Objective
Predict next-day closing price of a stock using historical data (no leakage).

### 📂 Dataset Used
- Apple (AAPL) stock data from Yahoo Finance (`yfinance`)
- **Period:** Jan 1, 2015 – Jan 1, 2024
- **Features:** Open, High, Low, Close, Volume, and lagged versions

### 🧠 Models Applied
- Linear Regression
- Random Forest Regressor

### 🔍 Key Results & Findings
- **Linear Regression (Test):** R² = 0.956, RMSE = 3.89
- **Random Forest (Test, Cell 8):** R² = 0.815, RMSE = 7.94
- **Random Forest (Test, Cell 10 - full training):** R² = 0.843 (better)
- **Time Series Cross Validation (5-fold):**
  - Linear Regression: R² = 0.965 (stable)
  - Random Forest: R² = -0.666 (unstable)
- **Small data CV (1800 samples):** Random Forest gave negative R² (-2.31), proving that too little training data causes poor CV performance.
- **Prediction:** Predicts Next Day Close Price.

### ✅ Conclusion
Linear Regression outperforms Random Forest for this task; RF shows instability in CV but improves with more data.

---

## 🏠 Task 6: House Price Prediction

### 🎯 Task Objective
Build a machine learning pipeline to predict house prices using a realistic pricing formula, evaluate model performance, and provide an interactive prediction tool.

### 📂 Dataset Used
- **URL:** https://www.kaggle.com/code/shayanzk/house-price-prediction-eda-regression/input
- **Note:** This dataset initially had uncorrelated prices; prices were reconstructed using a realistic formula based on area, bedrooms, bathrooms, floors, year built, location, condition, garage, and added market noise.

### 🧠 Models Applied
- **Baseline:** DummyRegressor (mean predictor)
- **Linear Regression**
- **Gradient Boosting Regressor** (300 estimators, max_depth=4, learning_rate=0.05)

### 🔍 Key Results and Findings
- **Performance Comparison:** Linear Regression performed slightly better than Gradient Boosting on test data (R² = 0.848 vs 0.819).
- **MAE:**
  - Linear Regression = $54,748
  - Gradient Boosting = $59,725
  - Baseline = $150,560
- **MAPE:**
  - Linear Regression = 11.7%
  - Gradient Boosting = 12.7%
- **Within ±10% accuracy:**
  - Linear Regression = 57.0%
  - Gradient Boosting = 54.2%
- **Cross-validation (5-fold) confirmed stable performance:**
  - LR CV R² mean ≈ 0.850
  - GB CV R² mean ≈ 0.826
- **Feature importance from GB** showed Area, AreaPerRoom, and Location (Downtown) as top predictors.
- **Live interactive prediction checker** was implemented, allowing user input for 8 house features to output predicted prices from both models and their average.

### ✅ Conclusion
Linear Regression slightly outperformed Gradient Boosting, achieving an R² of 0.848 and MAE of $54,748, demonstrating that a well-engineered linear model is both accurate and interpretable for this structured real-estate dataset.

---
