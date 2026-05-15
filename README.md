# 🏠 House Price Prediction using PyTorch ANN

A deep learning project that predicts house sale prices using Artificial Neural Networks (ANN) built with PyTorch. Three different model architectures are explored and compared to find the best-performing configuration.

---

## 📌 Problem Description

Predicting house prices is a classic regression problem in machine learning. Given a set of features describing a residential property (such as lot area, neighborhood, number of rooms, garage size, and more), the goal is to accurately predict the final **Sale Price** of the home.

This project uses the **Ames Housing Dataset**, which contains 79 explanatory variables covering almost every aspect of residential homes. The approach involves:

- Thorough **data preprocessing** (handling missing values, encoding categorical variables, feature scaling)
- Building and training **three ANN architectures** with different configurations
- Comparing results using **Mean Squared Error (MSE)**
- Generating final predictions on unseen test data

---

## 📂 Dataset

The dataset used is from the **Kaggle House Prices: Advanced Regression Techniques** competition.

🔗 **Dataset Link:** [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

### Files Required

| File | Description |
|------|-------------|
| `train.csv` | Training data with features and `SalePrice` target |
| `test.csv` | Test data with features only (no target) |

> Download both files and place them in the **same directory** as the script before running.

---

## 🧪 Experiments & Results

Three model architectures were trained and compared:

| Experiment | Architecture | Activation Function | Learning Rate | Final MSE |
|------------|-------------|---------------------|---------------|-----------|
| Exp 1 | 128 → 64 → 1 | ReLU | 0.001 | 0.021 |
| Exp 2 | 256 → 128 → 1 | Tanh | 0.0005 | 0.018 |
| **Exp 3** ✅ | **256 → 128 → 1 + Dropout** | **ReLU + Dropout(0.3)** | **0.001** | **0.015** |

### 🏆 Best Model: Experiment 3

**Experiment 3** achieved the lowest MSE of **0.015** by combining:
- A wider network (256 → 128 neurons)
- **Dropout regularization** (rate = 0.3) to reduce overfitting
- **ReLU** activation functions
- **Adam** optimizer with `lr=0.001`

---

## 🛠️ Project Structure

```
house-price-prediction/
│
├── train.csv               # Training dataset (download from Kaggle)
├── test.csv                # Test dataset (download from Kaggle)
├── house_price_ann.py      # Main Python script
├── predictions.csv         # Generated predictions (output)
└── README.md               # Project documentation
```

---

## ⚙️ Requirements

Install the required Python libraries before running:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Library Versions (Recommended)

| Library | Version |
|---------|---------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| pandas | 1.5+ |
| numpy | 1.23+ |
| scikit-learn | 1.2+ |
| matplotlib | 3.6+ |

---

## 🚀 How to Run the Project

### Step 1 — Clone or Download the Project

```bash
git clone https://github.com/Tadresfarag90/HPP_NN_Project
cd house-price-prediction
```

Or simply download the script and place it in a folder.

### Step 2 — Download the Dataset

1. Go to [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Accept the competition rules
3. Download `train.csv` and `test.csv`
4. Place both files in the **same folder** as the script

### Step 3 — Install Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Step 4 — Run the Script

```bash
python house_price_ann.py
```

### Step 5 — View Output

- **Training progress** is printed every 10 epochs
- **Loss curves** (Train vs Validation) are displayed as a plot
- **Experiment comparison table** is printed to the console
- **Predictions** are saved to `predictions.csv`

---

## 📈 Training Output Example

```
Epoch [10/100]
Train Loss: 0.0842
Validation Loss: 0.0791

Epoch [20/100]
Train Loss: 0.0523
Validation Loss: 0.0488
...

Final MSE: 0.015

Predictions saved successfully!
```

---

## 🔄 Pipeline Overview

```
Load Data (train.csv / test.csv)
        ↓
Explore & Check Missing Values
        ↓
Handle Missing Values (Median / "Missing")
        ↓
Encode Categorical Variables (One-Hot Encoding)
        ↓
Align Train & Test Features
        ↓
Feature Scaling (StandardScaler)
        ↓
Train/Validation Split (80/20)
        ↓
Convert to PyTorch Tensors
        ↓
Train ANN Model (Experiment 3)
        ↓
Evaluate on Validation Set (MSE)
        ↓
Generate & Save Predictions
```

---

## 📝 Key Design Decisions

- **Missing values** in numeric columns are filled with the **median** to avoid skew from outliers
- **Categorical columns** are filled with `"Missing"` before one-hot encoding
- Train and test features are **aligned** to ensure consistent columns after encoding
- **Dropout (0.3)** is applied after each hidden layer to prevent overfitting
- **Adam optimizer** is used for adaptive learning rate adjustments

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset provided by **Dean De Cock** and hosted on [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Built with [PyTorch](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/)
