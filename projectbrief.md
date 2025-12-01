# Project Brief: Online News Popularity Prediction System

## Project Overview
Develop a Machine Learning system in Julia to predict online news article popularity using binary classification. This is an ML final project demonstrating mastery of multiple ML techniques and scientific methodology.

## Dataset
**Name:** Online News Popularity  
**Source:** Mashable articles (2-year period)  
**Size:** 39,644 articles  
**Attributes:** 61 total
- 58 predictive features
- 2 non-predictive attributes (url, timedelta)
- 1 target variable (shares)

**Data Quality:** No missing values

## Problem Definition
**Task Type:** Binary Classification  
**Target Definition:** 
- Popular: shares > median(shares)
- Not Popular: shares ≤ median(shares)

```julia
median_shares = median(df[!, colname])
df.binary_class = df[!, colname] .> median_shares
```

## Core Requirements

### 1. ✅ Four Approaches Selected

**See `approaches.md` for detailed specifications.**

**Quick Summary:**
1. **Approach 1 - Baseline:** All 58 features, MinMax normalization, median split (balanced)
2. **Approach 2 - Feature Selection:** 15-20 selected features, MinMax, median split (balanced)
3. **Approach 3 - PCA:** 10-20 components, ZeroMean normalization, median split (balanced)
4. **Approach 4 - Imbalanced:** All 58 features, ZeroMean, 75th percentile split (25/75 imbalanced)

### 2. Four ML Techniques Required
Each approach must implement all four techniques:

1. **Artificial Neural Networks (ANNs)**
   - Test at least 8 different architectures
   - Use 1-2 hidden layers
   - Include validation split within training data
   - Use `ANNCrossValidation` from utils_ML1.jl

2. **Support Vector Machines (SVMs)**
   - Test at least 8 configurations
   - Vary kernels (linear, RBF, polynomial, sigmoid) and C values
   - Use `modelCrossValidation` with `:SVMClassifier`

3. **Decision Trees**
   - Test at least 6 different maximum depths
   - Use `modelCrossValidation` with `:DecisionTreeClassifier`

4. **k-Nearest Neighbors (kNN)**
   - Test at least 6 different k values
   - Use `modelCrossValidation` with `:KNeighborsClassifier`

### 3. Ensemble Methods
- Implement at least one ensemble technique
- Combine at least 3 individual models
- Options: majority voting, weighted voting, or stacking

### 4. Code Implementation
✅ **Verified:** All required functions available in `utils/utils_ML1.jl` (1538 lines)
- Normalization: `normalizeMinMax`, `normalizeZeroMean`
- Cross-validation: `modelCrossValidation`, `ANNCrossValidation`
- Evaluation: `confusionMatrix`, `accuracy`
- Ensemble: `VotingClassifier`

## Deliverables

### Folder Structure
```
/root_folder
├── Report.pdf        (50% of grade)
├── main.jl           (30% of grade - must be fully executable)
├── /datasets         (data files)
└── /utils            (utils_ML1.jl auxiliary code)
```

### Report Components (50% Total)

**Introduction (10%)**
- Problem description
- Dataset summary
- Evaluation metric justification
- Code structure explanation
- Bibliographic review (minimum 3 scientific publications)

**Development (25% per approach attempted)**
- Dataset variations description
- Data preprocessing justification
- Experimental setup (cross-validation, feature selection, etc.)
- Model experimentation results
- Ensemble method implementation
- Results and discussion with plots/confusion matrices/statistical tests

**Final Discussion (10%)**
- Overall process summary
- Cross-approach comparison
- Experimental conclusions

### Code Requirements (30% Total)
- Set random seed for reproducibility
- Load and preprocess dataset
- Extract relevant features
- Split data (train/test)
- Use `modelCrossValidation` function for parameter selection
- Train final models on full training set
- Evaluate on test set
- Include confusion matrices

## Success Criteria
- All code must be executable from beginning to end
- Results must be reproducible
- Report must clearly explain methodology and findings
- Statistical rigor in comparisons
- Professional academic writing with proper citations

## Timeline Constraints
This is a final project submission with academic deadlines.

