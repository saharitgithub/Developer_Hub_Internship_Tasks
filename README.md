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

# 📊 DevelopersHub Corporation — AI/ML Engineering Internship: Advanced Tasks

## 🛠️ Tech Stack & Requirements

| Category | Libraries / Tools |
|----------|-------------------|
| Deep Learning | `transformers`, `torch`, `datasets` |
| ML Pipeline | `scikit-learn`, `pandas`, `numpy` |
| Deployment | `gradio` |
| Environment | Google Colab (T4 GPU) |
| Language | Python 3.10+ |

---

## 📁 Advanced Task 1: News Topic Classifier Using BERT

### 🎯 Objective
Fine-tune a pre-trained DistilBERT (`distilbert-base-uncased`) model on the AG News dataset to build a News Topic Classifier that categorizes headlines into four classes: **World, Sports, Business, and Sci/Tech**.

### 📂 Dataset
| Property | Detail |
|----------|--------|
| Source | HuggingFace Datasets (`fancyzhx/ag_news`) |
| Total Size | 120,000 training + 7,600 test samples |
| Features | `text` (headline), `label` (0=World, 1=Sports, 2=Business, 3=Sci/Tech) |
| Split Used | Train: 108,000 · Validation: 12,000 · Test: 7,600 |

### 🧠 Models Applied
| Model | Description |
|-------|-------------|
| Zero-Shot (`bart-large-mnli`) | No training required; NLI-based classification |
| Fine-Tuned DistilBERT (`distilbert-base-uncased`) | Custom classification head; trained on AG News |

### 🔍 Results

**Fine-Tuned DistilBERT — Overall Performance**

| Metric | Value |
|--------|-------|
| Accuracy | 94.12% |
| Weighted F1 | 94.12% |
| Training Time | ~38 minutes (T4 GPU) |
| Model Size | ~438 MB (saved as safetensors) |

**Per-Class Performance**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| World | 0.9517 | 0.9532 | 0.9524 |
| Sports | 0.9894 | 0.9816 | **0.9855** ✅ Best |
| Business | 0.9204 | 0.9011 | **0.9106** ❌ Worst |
| Sci/Tech | 0.9042 | 0.9289 | 0.9164 |

### ✅ Conclusion
The fine-tuned DistilBERT model achieves excellent performance (94%+ accuracy) on news topic classification. **Sports** is the most easily distinguishable category (F1: 0.9855), while **Business** proves the most challenging (F1: 0.9106). The model was successfully deployed via Gradio with a public shareable link.

---

## 📈 Advanced Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

### 🎯 Objective
Build a complete, production-ready machine learning pipeline to predict customer churn using the IBM Telco Customer Churn dataset, covering data cleaning, preprocessing, model training, hyperparameter tuning, and full evaluation.

### 📂 Dataset
| Property | Detail |
|----------|--------|
| Source | IBM Telco Customer Churn Dataset |
| Size | 7,043 rows, 21 columns |
| Target | `Churn` (Yes/No → converted to 1/0) |
| Features | 3 numeric, 16 categorical (after cleaning) |

### 🧠 Models Applied
- Logistic Regression (with `class_weight='balanced'`)
- Random Forest Classifier (with `class_weight='balanced'`)

### 🔍 Results

**Test Set Performance Comparison**

| Model | Accuracy | Precision | Recall | F1 | Test AUC | CV AUC |
|-------|----------|-----------|--------|----|----------|--------|
| Logistic Regression | 0.7381 | 0.5043 | 0.7807 | 0.6128 | **0.8403** ✅ | 0.8456 |
| Random Forest | 0.7438 | 0.5113 | 0.7834 | 0.6188 | 0.8394 | 0.8445 |

**Key Insights**
- **Best Model:** Logistic Regression (highest Test AUC: 0.8403)
- **Churn Rate:** 26.5% — dataset is imbalanced; handled via `class_weight='balanced'`
- **Top Predictors (Random Forest Feature Importance):**

| Feature | Importance |
|---------|------------|
| Contract_Month-to-month | 0.1797 |
| tenure | 0.1182 |
| Contract_Two year | 0.1009 |
| OnlineSecurity_No | 0.0952 |
| InternetService_Fiber optic | 0.0765 |

- **Cross-validation:** Both models stable (CV AUC ~0.845)
- **Prediction Demo:**
  - High-risk customer (month-to-month contract, fiber optic) → **85.9% churn probability**
  - Low-risk customer (2-year contract, DSL) → **5.0% churn probability**

### ✅ Conclusion
Logistic Regression achieved the highest test AUC (0.8403) and was selected as the winning model. The pipeline includes full preprocessing (scaling + one-hot encoding), handles class imbalance, and supports both single and batch predictions. The saved pipeline is production-ready and can be loaded for inference in any environment.

---

## 🎫 Advanced Task 5: Auto-Tagging Support Tickets Using LLM

### 🎯 Objective
Implement and compare three approaches to automatically tag/classify customer support tickets into 8 categories: **Billing, Technical Issue, Login Problem, Account Access, Refund, Shipping, Cancellation,** and **Product Inquiry**.

### 📂 Dataset
| Property | Detail |
|----------|--------|
| Type | Synthetic but realistic support ticket dataset |
| Size | 240 samples (30 tickets per category) |
| Split | Train: 70% (168) · Validation: 15% (36) · Test: 15% (36) |

### 🧠 Models Applied

| Approach | Model | Use Case |
|----------|-------|----------|
| Zero-Shot | `facebook/bart-large-mnli` | No labeled data available |
| Few-Shot | `facebook/bart-large-mnli` + enriched prompts | Minimal labeled examples |
| Fine-Tuned | `distilbert-base-uncased` | Labeled data available |

### 🔍 Results

**Performance Comparison (on 36-sample test set)**

| Approach | Accuracy | F1-Score | Precision | Recall | Inference Time |
|----------|----------|----------|-----------|--------|----------------|
| Zero-Shot | 0.8333 | 0.8263 | 0.8302 | 0.8333 | ~112s |
| Few-Shot | 0.8611 | 0.8594 | 0.8643 | 0.8611 | ~106s |
| Fine-Tuned | **0.9444** | **0.9446** | **0.9455** | **0.9444** | **~22s** ✅ |

**Key Insights**
- Fine-Tuned DistilBERT outperforms both zero-shot and few-shot approaches by **+11% accuracy**
- Few-Shot improves over zero-shot by **+2.8% accuracy** through enriched label descriptions
- Fine-Tuned model is **5× faster** at inference compared to BART-based approaches
- DistilBERT is **40% smaller** than BART and runs **60% faster**
- Training time: ~8 minutes on T4 GPU (5 epochs, batch_size=16)
- Confusion matrix shows well-separated categories with minimal cross-category confusion

### ✅ Conclusion
Fine-tuned DistilBERT achieves the highest accuracy (94.4%) and is recommended for production use when labeled data is available. Zero-shot serves as an excellent cold-start solution when no training data exists. Few-shot provides a middle ground with modest gains over zero-shot. All three approaches are deployed via a unified Gradio interface for interactive testing.

---

## 📊 Overall Summary

| Task | Model | Key Metric | Score |
|------|-------|------------|-------|
| Task 1: News Classifier | Fine-Tuned DistilBERT | Accuracy | 94.12% |
| Task 2: Churn Prediction | Logistic Regression Pipeline | Test AUC | 0.8403 |
| Task 5: Ticket Auto-Tagging | Fine-Tuned DistilBERT | Accuracy | 94.44% |

---
