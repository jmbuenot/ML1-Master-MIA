# Approach 1 Results: Baseline with All Features

## Configuration

**Approach:** Baseline - Use all available features  
**Features:** 55 predictive features (excluding url, timedelta, shares)  
**Normalization:** MinMax [0, 1]  
**Target:** Binary classification via median split (threshold = 1,200 shares)  
**Class Distribution:** Balanced (48.5% popular / 51.5% not popular)  
**Random Seed:** 42  

## Dataset Splits

**Total Samples:** 39,644 articles  

**Training Set:** 31,716 samples (80%)
- Positive class: 15,382 (48.5%)
- Negative class: 16,334 (51.5%)

**Test Set:** 7,928 samples (20%)
- Positive class: 3,845 (48.5%)
- Negative class: 4,083 (51.5%)

## Cross-Validation Setup

- **Method:** 10-fold stratified cross-validation
- **Applied to:** Training set only (31,716 samples)
- **Function:** `modelCrossValidation` from utils_ML1.jl

## Model Results (Cross-Validation)

### 1. Artificial Neural Networks (ANNs)

Tested 8 different topologies with:
- Learning rate: 0.01
- Validation ratio: 0.2
- Number of executions: 5
- Max epochs: 100
- Max epochs validation: 20

| Topology | CV Accuracy | Std Dev |
|----------|-------------|---------|
| [10] | 61.48% | ±1.60% |
| [20] | 62.31% | ±1.22% |
| [50] | 62.73% | ±1.12% |
| [100] | 62.65% | ±1.17% |
| [20, 10] | 62.04% | ±1.46% |
| **[50, 25]** | **62.88%** | **±1.19%** ⭐ |
| [100, 50] | 62.57% | ±1.15% |
| [50, 50] | 62.71% | ±1.05% |

**Best Configuration:** [50, 25] topology with 62.88% accuracy

**Observations:**
- Performance relatively consistent across topologies (61-63%)
- Deeper networks don't significantly improve performance
- Two-layer [50, 25] architecture slightly better than single layer
- Low standard deviations (1-1.6%) indicate stable performance

### 2. Support Vector Machines (SVMs)

Tested 8 different configurations with various kernels:

| Configuration | CV Accuracy | Std Dev |
|---------------|-------------|---------|
| Linear, C=0.1 | 60.86% | ±1.29% |
| Linear, C=1.0 | 61.94% | ±1.32% |
| Linear, C=10.0 | 62.85% | ±1.27% |
| RBF, C=1.0, γ=0.001 | 60.87% | ±1.18% |
| RBF, C=1.0, γ=0.01 | 60.69% | ±1.13% |
| RBF, C=10.0, γ=0.01 | 61.81% | ±1.18% |
| Poly, C=1.0, degree=2 | 62.91% | ±1.08% |
| **Poly, C=1.0, degree=3** | **63.43%** | **±1.04%** ⭐ |

**Best Configuration:** Polynomial kernel (degree 3), C=1.0 with 63.43% accuracy

**Observations:**
- Polynomial kernel outperforms linear and RBF
- Linear kernel performance improves with higher C values
- RBF kernel shows poorest performance
- Polynomial degree 3 better than degree 2

### 3. Decision Trees

Tested 6 different maximum depths:

| Max Depth | CV Accuracy | Std Dev |
|-----------|-------------|---------|
| 3 | 67.27% | ±0.89% |
| 5 | 69.16% | ±1.02% |
| **7** | **69.58%** | **±0.85%** ⭐ |
| 10 | 69.48% | ±0.91% |
| 15 | 67.36% | ±0.70% |
| 20 | 65.42% | ±0.70% |

**Best Configuration:** Max depth = 7 with 69.58% accuracy

**Observations:**
- **Best performing model overall**
- Clear overfitting after depth 10 (accuracy drops)
- Optimal depth is 7
- Very stable predictions (low std dev)
- Suggests feature interactions and non-linear relationships are important

### 4. k-Nearest Neighbors (kNN)

Tested 6 different k values:

| k | CV Accuracy | Std Dev |
|---|-------------|---------|
| 1 | 56.65% | ±0.81% |
| 3 | 57.90% | ±0.85% |
| 5 | 58.89% | ±0.78% |
| 7 | 59.29% | ±0.85% |
| **9** | **59.78%** | **±1.12%** ⭐ |
| 11 | 59.77% | ±0.82% |

**Best Configuration:** k = 9 with 59.78% accuracy

**Observations:**
- Poorest performing method
- Performance improves with higher k (reducing overfitting)
- Plateaus around k=9-11
- Distance-based methods struggle with this 55-dimensional feature space

## Overall Performance Ranking

1. **Decision Trees (depth 7):** 69.58% ± 0.85% 🥇
2. **SVM (polynomial degree 3):** 63.43% ± 1.04% 🥈
3. **ANN ([50, 25]):** 62.88% ± 1.19% 🥉
4. **kNN (k=9):** 59.78% ± 1.12%

**Performance Gap:** 9.8 percentage points between best and worst

## Key Insights

### Model-Specific Insights

1. **Decision Trees dominate:** 6-7% better than neural networks
   - Suggests non-linear feature interactions are crucial
   - Tree-based methods naturally handle feature interactions
   - No need for feature scaling (inherent advantage)

2. **ANNs and SVMs comparable:** Both around 62-63%
   - Polynomial kernel captures non-linearity similar to ANNs
   - Both benefit from normalization

3. **kNN underperforms:** Curse of dimensionality
   - 55-dimensional space makes distance metrics less meaningful
   - Would likely improve with dimensionality reduction

### Dataset Insights

1. **Problem characteristics:**
   - Non-linear relationships dominate
   - Feature interactions important
   - 55 dimensions challenging for distance-based methods

2. **Class balance:** Perfect for accuracy as primary metric
   - 48.5% / 51.5% split very balanced
   - No need for class weighting

3. **Stability:** Low standard deviations across all models
   - 10-fold CV provides reliable estimates
   - Large dataset (31K training samples) helps stability

## Next Steps

### For Approach 1:
- [ ] Train final models on full training set with best hyperparameters
- [ ] Evaluate on held-out test set (7,928 samples)
- [ ] Generate confusion matrices for all 4 models
- [ ] Implement ensemble method (combine top 3 models)
- [ ] Create visualization plots

### For Subsequent Approaches:
- **Approach 2 (Feature Selection):** Test if 19 selected features can match performance
- **Approach 3 (PCA):** Test if dimensionality reduction helps kNN and SVMs
- **Approach 4 (Imbalanced):** Test performance on highly viral articles (75th percentile)

## Files

- **Results:** `/results/approach1.txt`
- **Code:** `/main.jl` (run with `julia main.jl 1`)
- **Dataset:** `/datasets/OnlineNewsPopularity.csv`
- **Utils:** `/utils/utils_ML1.jl`




