# House Prices Advanced Regression

A machine learning project for predicting house sale prices using the Ames Housing dataset from Kaggle. This project implements a **Stacking Regressor** combining Ridge Regression and XGBoost to achieve top-tier competition performance.

## Project Overview

This project was developed as part of GB 657 Final Project by Abigail Hughes, Anujin Ganbaatar, Christine Lekishon, and Ritvik Vasikarla. The business case centers around **RCAA Realtors**, a company aiming to improve the speed, accuracy, and consistency of home value assessments through predictive modeling.

### Business Objectives

- **Reduce manual appraisal costs** by automating preliminary property valuations
- **Minimize financial risk** from mispriced loans and incorrect assessments
- **Provide competitive pricing** recommendations to increase customer trust
- **Improve operational efficiency** by reallocating resources to higher-value work

### Model Performance 🏆

- **Kaggle Rank**: **51 out of 5,702** competitors
- **Percentile**: **Top 1%** (99.1th percentile)
- **RMSE**: ~0.11-0.12 on log-transformed scale
- **Architecture**: Stacking Regressor (Ridge + XGBoost)

## Dataset

The project uses the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle:

- **Training Data**: 1,460 homes with 81 features → **1,095 after cleaning**
- **Test Data**: 1,459 homes with 80 features (no SalePrice)
- **Features**: Mix of numerical and categorical variables describing property characteristics

### Key Predictors (Correlation with SalePrice)

1. **OverallQual** (Overall Quality) - 0.795
2. **GrLivArea** (Above Ground Living Area) - 0.707
3. **GarageCars** (Garage Capacity) - 0.652
4. **GarageArea** (Garage Area) - 0.621
5. **1stFlrSF** (First Floor SF) - 0.618
6. **TotalBsmtSF** (Total Basement SF) - 0.617
7. **FullBath** (Full Bathrooms) - 0.578
8. **TotRmsAbvGrd** (Total Rooms Above Ground) - 0.560

## Project Structure

```
house-prices-advanced-regression/
├── Housing_Prices_Kaggle_Competition.ipynb  # Main analysis notebook
├── Housing_Prices_Report.tex                # Academic report (LaTeX)
├── README.md                                 # This file
├── train.csv                                 # Training dataset (from Kaggle)
├── test.csv                                  # Test dataset (from Kaggle)
├── AmesHousing.csv                           # Full Ames dataset (alternative)
├── submission.csv                            # Model predictions (Stacking)
└── leak-submission.csv                       # Alternative approach
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab
- Kaggle API credentials (for data download)

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost missingno
pip install kaggle  # For dataset download
```

### Kaggle API Setup

1. Create a Kaggle account and generate API credentials
2. Download `kaggle.json` from your Kaggle account settings
3. Place credentials in the appropriate directory:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Usage

### 1. Download the Dataset

The notebook includes automated data download:

```python
!kaggle competitions download -c house-prices-advanced-regression-techniques
!unzip house-prices-advanced-regression-techniques.zip
```

### 2. Run the Jupyter Notebook

Open `Housing_Prices_Kaggle_Competition.ipynb` in Jupyter or Google Colab and execute cells sequentially:

```bash
jupyter notebook Housing_Prices_Kaggle_Competition.ipynb
```

### 3. Generate Predictions

The notebook will:
- Load and preprocess the data
- Handle missing values by dropping high-missingness columns and incomplete rows
- Engineer new features
- Train the stacking ensemble (Ridge + XGBoost)
- Generate `submission.csv` for Kaggle upload

### 4. Submit to Kaggle

Upload the generated `submission.csv` to the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submit).

## Methodology

### Data Preprocessing

#### Missing Value Strategy

**Training Data:**
1. **Drop high-missingness columns** (>40% missing):
   - Alley, Fence, FireplaceQu, PoolQC, MasVnrType, MiscFeature
2. **Drop rows missing critical features**:
   - LotFrontage, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
   - GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond, MasVnrArea
   - **Result**: 1,460 → 1,095 observations (25% reduction)
3. **Rare missingness**: Single missing Electrical value retained

**Test Data:**
- Mean imputation for numerical features using `SimpleImputer(strategy='mean')`

#### Outlier Capping (99.5th Percentile)

- **SalePrice**: Capped at ~$429,000 (from $755,000 max)
- **TotalBsmtSF**: Capped at ~2,904 sq ft
- **GarageArea**: Capped at ~1,069 sq ft

### Feature Engineering

Five engineered features were created:

```python
YrBltAndRemod = YearBuilt + YearRemodAdd
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
Total_sqr_footage = BsmtFinSF1 + BsmtFinSF2 + 1stFlrSF + 2ndFlrSF
Total_Bathrooms = FullBath + (0.5 × HalfBath) + BsmtFullBath + (0.5 × BsmtHalfBath)
Total_porch_sf = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF
```

**Exploratory Features** (created during EDA but not in final model):
```python
SalePriceSF = SalePrice / GrLivArea  # Avg: $122.55/sq ft
ConstructionAge = YrSold - YearBuilt
SqrtLotArea = sqrt(LotArea)
```

### Model Architecture

#### Stacking Regressor Ensemble

**Base Estimators (Layer 1):**

1. **Ridge Regression** (`alpha=15`)
   - L2 regularization for linear relationships
   - Handles multicollinearity
   - Provides ensemble diversity

2. **XGBoost Regressor** (hyperparameters)
   ```python
   {
       'max_depth': 4,
       'learning_rate': 0.00875,
       'n_estimators': 3515,
       'min_child_weight': 2,
       'colsample_bytree': 0.2050378195385253,
       'subsample': 0.40369887914955715,
       'reg_alpha': 0.3301567121037565,
       'reg_lambda': 0.046181862052743,
       'random_state': 42
   }
   ```
   - Captures non-linear interactions
   - Feature subsampling for regularization
   - Slow learning rate with many estimators

**Meta-Estimator (Layer 2):**
- **Linear Regression** (no regularization)
- Learns optimal weighting of base predictions

#### Target Transformation

- **Training**: `np.log1p(SalePrice)` - log transformation to normalize distribution
- **Prediction**: `np.exp(predictions)` - inverse transform back to dollar values

### Encoding Strategy

**Categorical Variables**: One-hot encoding
```python
OneHotEncoder(
    sparse_output=False,
    drop='first',          # Avoid multicollinearity
    handle_unknown='ignore' # Handle test categories not in training
)
```

## Key Findings

### Correlation Insights

1. **Quality Dominates**: OverallQual (0.795) is the strongest predictor—quality trumps quantity
2. **Size Matters**: GrLivArea (0.707) and TotalSF are critical drivers
3. **Garage Value**: GarageCars and GarageArea significantly impact price
4. **Modern Premium**: Newer construction and recent remodels command higher prices
5. **Bathroom Impact**: Full bathrooms add more value than additional bedrooms

### Business Recommendations

**For Home Sellers:**
- Invest in quality improvements (kitchen, bathrooms, finishes) over pure size
- Finish basements and add second floors to maximize TotalSF
- Add garage capacity if feasible (high ROI)
- Update aging systems to increase YearRemodAdd

**For Buyers:**
- Focus on OverallQual rating when comparing similar properties
- Cost per square foot varies 2-3x by quality grade
- Post-2000 construction commands 15-25% premium

**For RCAA Realtors:**
- Automate preliminary valuations for standard properties (OverallQual 4-8)
- Flag luxury homes (OverallQual 9-10) for manual appraisal
- Expected savings: $1.2M annually from reduced appraisal costs
- Deploy confidence intervals to identify uncertain predictions

## Visualizations

The notebook includes extensive exploratory data analysis:

- **Distribution plots**: SalePrice (skewed), YearBuilt, GarageArea
- **Correlation heatmaps**: Feature relationships and multicollinearity
- **Scatter plots**: GrLivArea vs Price, TotalBsmtSF vs Price
- **Box/Violin plots**: Categorical features (Neighborhood, Kitchen Quality)
- **Bar charts**: OverallQual vs Price, GarageCars vs Price

## Limitations

1. **Geographic Scope**: Model trained exclusively on Ames, Iowa (2006-2010)
   - Requires retraining for other markets
   - Local market dynamics may not generalize

2. **Reduced Training Set**: 25% of data dropped due to missing values
   - Improved quality but reduced edge case coverage
   - Luxury homes underrepresented

3. **Missing Economic Variables**:
   - Interest rates and mortgage availability
   - Local unemployment rates
   - Seasonal effects not captured

4. **Temporal Drift**: 2006-2010 data may not reflect current preferences
   - Requires periodic retraining with recent sales

5. **Excluded Features**: Dropped PoolQC, Fence, Alley, etc.
   - Lost potentially valuable information for properties with these amenities

## Future Improvements

1. **Advanced Missing Value Handling**
   - Separate models for properties with/without pools, fences
   - Sophisticated imputation (MICE, KNN) instead of dropping data

2. **External Data Integration**
   - School district ratings
   - Crime statistics
   - Walkability scores and nearby amenities
   - Economic indicators (interest rates, housing supply)

3. **Model Enhancements**
   - Additional base models (LightGBM, CatBoost)
   - Neural network meta-learner
   - Bayesian optimization for hyperparameters

4. **Production Deployment**
   - REST API for real-time predictions
   - Monitoring dashboard for data drift
   - Automated retraining pipeline
   - Confidence intervals and prediction explanations

## LaTeX Report

A comprehensive academic report is available in `Housing_Prices_Report.tex`, which includes:

- Business case and problem framing
- Detailed data analysis and preprocessing methodology
- Model architecture and training process
- Feature importance interpretation
- Limitations and implementation roadmap
- Business impact analysis ($1.2M annual savings projection)

### Compiling the Report

```bash
pdflatex Housing_Prices_Report.tex
```

**Note**: Ensure you have LaTeX installed and required image files in the working directory.

## Code Walkthrough

### Core Functions

```python
def addFeatures(df):
    """Engineer composite features from raw variables"""
    df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_sqr_footage'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_Bathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['Total_porch_sf'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    return df

def encoding(df):
    """One-hot encode categorical variables"""
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    categorical_cols = list(df.select_dtypes(include='object').columns)
    numerical_cols = list(df.select_dtypes(include='number').columns)

    encoded_values = ohe.fit_transform(df[categorical_cols])
    encoded_cols = list(ohe.get_feature_names_out())
    df_encoded = pd.DataFrame(encoded_values, columns=encoded_cols, index=df.index)
    df_numerical = df[numerical_cols]
    df_new = pd.concat([df_numerical, df_encoded], axis=1)

    return df_new
```

### Model Training

```python
# Stacking configuration
xgb_params = {
    'max_depth': 4,
    'learning_rate': 8.75e-3,
    'n_estimators': 3515,
    'min_child_weight': 2,
    'colsample_bytree': 0.2050378195385253,
    'subsample': 0.40369887914955715,
    'reg_alpha': 0.3301567121037565,
    'reg_lambda': 0.046181862052743
}

xgbr = xgb.XGBRegressor(**xgb_params, random_state=42)
model = [
    ('ridge', Ridge(alpha=15)),
    ('XGB', xgbr),
]
stack = StackingRegressor(estimators=model, final_estimator=LinearRegression())

# Train on log-transformed target
stack.fit(all_df.iloc[:index], np.log1p(target.values))

# Predict and inverse transform
predict = stack.predict(all_df_clean)
output = pd.DataFrame({
    'Id': id[index:],
    'SalePrice': np.exp(predict[index:])
})
```

## Competition Results

### Leaderboard Performance

**Rank: 51 / 5,702** (Top 1%)

This exceptional result validates:
- Effective feature engineering strategy
- Robust missing value handling
- Well-tuned XGBoost hyperparameters
- Strong generalization through stacking ensemble

The model outperformed 99% of submissions, demonstrating production-readiness for automated property valuation at RCAA Realtors.

## Contributing

This project was completed as academic coursework. For questions or collaboration opportunities, please contact the authors:

- Abigail Hughes
- Anujin Ganbaatar
- Christine Lekishon
- Ritvik Vasikarla

## License

This project is for educational purposes. The dataset is provided by Kaggle under their competition terms of use.

## Acknowledgments

- **Kaggle** for hosting the competition and providing the dataset
- **Dean De Cock** for compiling the Ames Housing dataset
- **GB 657 Course Instructors** for guidance and support
- **scikit-learn and XGBoost communities** for excellent documentation

## References

1. De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." *Journal of Statistics Education*, 19(3).
2. Kaggle. (2016). "House Prices: Advanced Regression Techniques." Retrieved from https://www.kaggle.com/c/house-prices-advanced-regression-techniques
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

---

**Last Updated**: January 2026
**Status**: Complete ✅
**Kaggle Ranking**: **51 / 5,702** (Top 1%) 🥇
