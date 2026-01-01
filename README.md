# House Prices Advanced Regression

A comprehensive machine learning project for predicting house sale prices using the Ames Housing dataset from Kaggle. This project implements advanced regression techniques including Ridge Regression, XGBoost, and Stacking Ensemble methods to achieve accurate price predictions.

## Project Overview

This project was developed as part of GB 657 Final Project by Abigail Hughes, Anujin Ganbaatar, Christine Lekishon, and Ritvik Vasikarla. The business case centers around **RCAA Realtors**, a company aiming to improve the speed, accuracy, and consistency of home value assessments through predictive modeling.

### Business Objectives

- **Reduce manual appraisal costs** by automating preliminary property valuations
- **Minimize financial risk** from mispriced loans and incorrect assessments
- **Provide competitive pricing** recommendations to increase customer trust
- **Improve operational efficiency** by reallocating resources to higher-value work

### Model Performance

- **RMSE**: 0.111 on log-transformed scale (~10% average error in dollar terms)
- **Kaggle Ranking**: Top 10% (579th out of 6064 competitors)
- **Improvement**: 71.1% improvement over baseline model

## Dataset

The project uses the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle, which contains:

- **Training Data**: 1,460 homes with 81 features
- **Test Data**: 1,459 homes with 80 features (no SalePrice)
- **Features**: Mix of numerical and categorical variables describing property characteristics

### Key Features

The most important predictors identified include:

1. **OverallQual** (Overall Quality) - Correlation: 0.79
2. **GrLivArea** (Above Ground Living Area) - Correlation: 0.71
3. **GarageCars** (Garage Capacity) - Correlation: 0.65
4. **GarageArea** (Garage Area in sq ft) - Correlation: 0.62
5. **TotalBsmtSF** (Total Basement Square Footage) - Correlation: 0.62
6. **1stFlrSF** (First Floor Square Footage) - Correlation: 0.62

## Project Structure

```
house-prices-advanced-regression/
├── Housing_Prices_Kaggle_Competition.ipynb  # Main analysis notebook
├── Housing_Prices_Report.tex                # LaTeX report document
├── README.md                                 # This file
├── train.csv                                 # Training dataset (downloaded)
├── test.csv                                  # Test dataset (downloaded)
├── submission.csv                            # Model predictions for Kaggle
└── leak-submission.csv                       # Alternative submission approach
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
- Handle missing values appropriately
- Engineer new features
- Train ensemble models
- Generate `submission.csv` for Kaggle upload

### 4. Submit to Kaggle

Upload the generated `submission.csv` to the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submit).

## Methodology

### Data Preprocessing

#### Missing Value Treatment
- **High missingness features** (PoolQC, Alley, Fence, FireplaceQu, MasVnrType, MiscFeature): Dropped from training
- **Critical features** (LotFrontage, Basement, Garage features): Rows with missing values removed
- **Rare missingness** (Electrical): Imputed using most frequent value
- **Test set**: Mean imputation for numerical features

#### Outlier Handling
- **SalePrice**: Capped at 99.5th percentile
- **TotalBsmtSF**: Capped at 99.5th percentile
- **GarageArea**: Capped at 99.5th percentile

### Feature Engineering

The following engineered features were created:

```python
YrBltAndRemod = YearBuilt + YearRemodAdd
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
Total_sqr_footage = BsmtFinSF1 + BsmtFinSF2 + 1stFlrSF + 2ndFlrSF
Total_Bathrooms = FullBath + (0.5 × HalfBath) + BsmtFullBath + (0.5 × BsmtHalfBath)
Total_porch_sf = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF
SalePriceSF = SalePrice / GrLivArea
ConstructionAge = YrSold - YearBuilt
SqrtLotArea = sqrt(LotArea)
```

### Models

#### Ensemble Architecture

The final model uses a **Stacking Regressor** with:

**Base Estimators:**
1. **Ridge Regression** (alpha=15)
   - Linear baseline with L2 regularization
   - Handles multicollinearity effectively

2. **XGBoost Regressor** (optimized hyperparameters)
   - max_depth: 4
   - learning_rate: 0.00875
   - n_estimators: 3515
   - min_child_weight: 2
   - colsample_bytree: 0.205
   - subsample: 0.404
   - reg_alpha: 0.330
   - reg_lambda: 0.046

**Meta-Estimator:**
- **Linear Regression** (final layer combining base predictions)

#### Target Transformation
- Log transformation applied: `np.log1p(SalePrice)`
- Inverse transform for predictions: `np.exp(predictions)`

### Encoding Strategy

- **Categorical Variables**: One-hot encoding with `drop='first'`
- **Handle Unknown**: Categories in test set not in training are handled gracefully
- **Sparse Output**: Set to False for compatibility with sklearn models

## Key Findings

### Correlation Insights

1. **Quality Over Quantity**: Overall quality (OverallQual) is the strongest predictor (0.79 correlation)
2. **Size Matters**: Living area (GrLivArea) strongly correlates with price (0.71)
3. **Modern Homes Premium**: Year built and remodel dates positively impact price
4. **Garage Value**: Garage capacity and area significantly influence sale price
5. **Bathroom Impact**: Full bathrooms add more value than additional bedrooms

### Feature Importance Rankings

Top 10 most important features from XGBoost model:
1. OverallQual
2. TotalSF (engineered)
3. GrLivArea
4. GarageCars
5. Total_Bathrooms (engineered)
6. YearBuilt
7. GarageArea
8. TotalBsmtSF
9. 1stFlrSF
10. YearRemodAdd

### Business Recommendations

**For Home Sellers:**
- Invest in quality improvements (kitchen, bathrooms, finishes)
- Maximize usable square footage
- Add garage capacity if feasible
- Update aging infrastructure

**For Buyers:**
- Focus on overall quality ratings when comparing properties
- Consider cost per square foot in context of quality grade
- Newer construction and recent renovations command premium prices

**For RCAA Realtors:**
- Use model for preliminary automated valuations
- Flag properties with prediction confidence < 90% for manual appraisal
- Implement dashboard with real-time pricing for new listings
- Expected savings: $1.2M annually from reduced appraisal costs

## Visualizations

The notebook includes extensive exploratory data analysis with visualizations:

- **Distribution plots**: SalePrice, YearBuilt, GarageArea, etc.
- **Correlation heatmaps**: Feature relationships
- **Scatter plots**: Relationships between key predictors and price
- **Box plots & Violin plots**: Categorical feature analysis
- **Bar charts**: Neighborhood effects, Quality ratings

## Limitations

1. **Geographic Scope**: Model trained exclusively on Ames, Iowa data
   - May not generalize to other housing markets
   - Requires retraining for different locations

2. **Economic Variables**: Missing macroeconomic factors
   - Interest rates not included
   - Economic cycles not captured
   - Seasonal effects not fully modeled

3. **Luxury Home Data**: High-end properties underrepresented
   - Predictions for expensive homes may be less reliable
   - Limited training signal for luxury features

4. **Temporal Limitations**: Data from 2006-2010
   - Market conditions have changed
   - New construction standards not reflected

## Future Improvements

1. **Enhanced Feature Engineering**
   - Interaction terms between quality and size
   - Polynomial features for non-linear relationships
   - Time-series features for market trends

2. **Advanced Models**
   - Deep learning architectures
   - Gradient Boosting variants (CatBoost, LightGBM)
   - Neural network ensembles

3. **External Data Integration**
   - School district ratings
   - Crime statistics
   - Economic indicators
   - Nearby amenities (parks, shopping, transit)

4. **Production Deployment**
   - API endpoint for real-time predictions
   - Monitoring for data drift
   - Automated retraining pipeline
   - Confidence intervals for predictions

## LaTeX Report

A comprehensive academic report is available in `Housing_Prices_Report.tex`, which includes:

- Business case and problem framing
- Detailed data analysis methodology
- Model selection and evaluation
- Feature importance interpretation
- Limitations and future work
- Business impact analysis

### Compiling the Report

```bash
pdflatex Housing_Prices_Report.tex
```

Note: Ensure you have LaTeX installed and required image files in the working directory.

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

1. De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." Journal of Statistics Education, 19(3).
2. Kaggle. (2016). "House Prices: Advanced Regression Techniques." Retrieved from https://www.kaggle.com/c/house-prices-advanced-regression-techniques
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

---

**Last Updated**: January 2026
**Status**: Complete ✓
**Kaggle Ranking**: Top 10% 🏆
