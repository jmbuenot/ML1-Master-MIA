
# Project Approaches: Detailed Specifications

## Overview
This project implements **four distinct approaches** to predict online news popularity. Each approach tests different hypotheses about feature representation, normalization, and problem formulation, providing comprehensive insights for the final report.

---

## Approach 1: Baseline (Full Features)

### Objective
Establish baseline performance using all available features without any dimensionality reduction.

### Configuration
- **Features:** All 58 predictive features
- **Normalization:** MinMax (scales features to [0, 1])
- **Target:** Binary classification via median split
  ```julia
  median_shares = median(df.shares)
  y = df.shares .> median_shares
  ```
- **Class Distribution:** Balanced (50% popular / 50% not popular)

### Rationale
- Provides reference performance for comparison
- No information loss from feature selection or transformation
- Tests whether all features contribute meaningfully
- MinMax normalization suitable for ANNs, SVMs, and kNN

### Models to Train
1. **ANNs:** Test 8+ architectures (1-2 hidden layers)
2. **SVMs:** Test 8+ configurations (different kernels: linear, RBF, polynomial; vary C and gamma)
3. **Decision Trees:** Test 6+ maximum depths
4. **kNN:** Test 6+ k values

### Expected Insights
- Baseline accuracy with full information
- Which model types work best with high-dimensional data
- Feature importance (if analyzed)
- Reference for evaluating dimensionality reduction benefits

---

## Approach 2: Feature Selection (Selected Important Features)

### Objective
Test hypothesis that a carefully selected subset of features can achieve comparable or better performance than using all features.

### Configuration
- **Features:** 15-20 selected meaningful features
- **Normalization:** MinMax (same as baseline for fair comparison)
- **Target:** Binary classification via median split (same as baseline)
- **Class Distribution:** Balanced (50/50)

### Feature Selection Strategy
Select features from multiple categories:

**Text & Content Features (4-5 features):**
- `n_tokens_title` - Title length
- `n_tokens_content` - Content length
- `num_keywords` - Number of keywords
- `average_token_length` - Word complexity

**Multimedia & Engagement (3-4 features):**
- `num_hrefs` - Links in article
- `num_imgs` - Images
- `num_videos` - Videos
- `num_self_hrefs` - Self-references

**Sentiment & Tone (4-5 features):**
- `global_sentiment_polarity` - Overall sentiment
- `global_subjectivity` - Subjectivity level
- `title_sentiment_polarity` - Title sentiment
- `rate_positive_words` - Positive word rate
- `avg_positive_polarity` - Average positive polarity

**Topic & Channel (3-4 features):**
- `LDA_00`, `LDA_01`, `LDA_02` - Top LDA topics
- One or two channel indicators (e.g., `data_channel_is_tech`)

**Temporal Features (1-2 features):**
- `is_weekend` - Weekend publication
- One weekday indicator

### Rationale
- Reduces computational complexity
- Tests feature importance hypothesis
- Potentially reduces overfitting
- Improves model interpretability
- Common practice in ML pipelines

### Expected Insights
- Can fewer features achieve similar performance?
- Which feature categories are most important?
- Trade-off between simplicity and performance
- Model sensitivity to feature selection

---

## Approach 3: PCA Dimensionality Reduction

### Objective
Transform features into principal components that capture maximum variance, testing whether linear combinations of features work better than original features.

### Configuration
- **Features:** Start with all 58 features
- **Normalization:** **ZeroMean (z-score)** - REQUIRED for PCA
  ```julia
  # Must use zero-mean normalization before PCA
  params = calculateZeroMeanNormalizationParameters(X_train)
  X_train_norm = normalizeZeroMean(X_train, params)
  X_test_norm = normalizeZeroMean(X_test, params)
  ```
- **Transformation:** PCA to reduce to 10 or 20 principal components
- **Target:** Binary classification via median split
- **Class Distribution:** Balanced (50/50)

### Implementation Notes
- PCA must be fitted on training data only
- Same transformation applied to test data
- Analyze explained variance ratio
- Compare 10 vs 20 components

### Rationale
- Different from feature selection (transforms rather than selects)
- Captures maximum variance in fewer dimensions
- Reduces multicollinearity
- Standard dimensionality reduction technique
- Tests if linear combinations outperform original features

### Expected Insights
- How much variance captured by top components?
- Performance vs dimensionality trade-off
- Whether decorrelated features improve models
- Difference between 10 and 20 components

---

## Approach 4: Imbalanced Classification (Highly Viral Articles)

### Objective
Change problem definition from "above average" to "highly viral" articles, introducing class imbalance and testing models' ability to identify truly exceptional content.

### Configuration
- **Features:** All 58 features (like baseline)
- **Normalization:** **ZeroMean (z-score)** - different from Approaches 1 & 2
- **Target:** Binary classification via 75th percentile threshold
  ```julia
  threshold_75 = quantile(df.shares, 0.75)
  y = df.shares .> threshold_75
  ```
- **Class Distribution:** Imbalanced (25% highly viral / 75% not highly viral)

### Rationale
- **Real-world relevance:** Publishers care most about identifying viral content
- **Different problem:** Tests "exceptional" vs "typical" rather than "above average"
- **Methodological diversity:** Introduces class imbalance handling
- **Different normalization:** Tests z-score vs MinMax
- **Evaluation complexity:** Requires different metrics (F1, AUC-ROC, precision/recall)

### Special Considerations
- **Evaluation Metrics:** 
  - Accuracy less meaningful with imbalance
  - F1-score balances precision and recall
  - AUC-ROC measures overall discrimination ability
  - Confusion matrix reveals class-specific performance
  
- **Model Training:**
  - May need class weights for some models
  - Decision threshold tuning may be beneficial
  - Different models handle imbalance differently

### Expected Insights
- Which models handle imbalance best?
- Precision vs recall trade-offs
- Comparison of balanced vs imbalanced performance
- Effect of normalization method (z-score vs MinMax)
- Practical value for real deployment

---

## Comparison Matrix

| Aspect | Approach 1 | Approach 2 | Approach 3 | Approach 4 |
|--------|-----------|-----------|-----------|-----------|
| **Features** | All (58) | Selected (15-20) | PCA (10-20) | All (58) |
| **Normalization** | MinMax | MinMax | ZeroMean | ZeroMean |
| **Target** | Median | Median | Median | 75th %ile |
| **Class Balance** | 50/50 | 50/50 | 50/50 | 25/75 |
| **Complexity** | High-dim | Low-dim | Low-dim | High-dim |
| **Primary Metric** | Accuracy | Accuracy | Accuracy | F1 / AUC |
| **Hypothesis** | Baseline | Selection helps | Transform helps | Imbalance reality |

---

## Cross-Cutting Requirements

### All Approaches Must Include:

1. **Model Experimentation:**
   - ANNs: 8+ architectures
   - SVMs: 8+ configurations
   - Decision Trees: 6+ depths
   - kNN: 6+ k values

2. **Cross-Validation:**
   - Use `modelCrossValidation` from utils_ML1.jl
   - Apply only to training data
   - Select best hyperparameters

3. **Final Evaluation:**
   - Train with best parameters on full training set
   - Evaluate on held-out test set
   - Generate confusion matrix

4. **Ensemble Method:**
   - Combine at least 3 best models
   - Use majority/weighted voting or stacking
   - Compare ensemble to individual models

5. **Documentation:**
   - Justify all preprocessing choices
   - Report all experiments with metrics
   - Include visualizations (plots, confusion matrices)
   - Discuss results with statistical comparisons

---

## Implementation Order

### Recommended Sequence:
1. **Start with Approach 1 (Baseline)**
   - Establishes data pipeline
   - Provides reference results
   - Validates all model implementations

2. **Continue with Approach 2 (Feature Selection)**
   - Reuses pipeline from Approach 1
   - Same normalization and target
   - Tests feature importance hypothesis

3. **Implement Approach 3 (PCA)**
   - Adds PCA transformation step
   - Different normalization (ZeroMean)
   - Same target definition

4. **Complete with Approach 4 (Imbalanced)**
   - Different target definition
   - Different evaluation focus
   - Most complex analysis

---

## Success Criteria

### Per Approach:
- ✅ All 4 model types trained and evaluated
- ✅ Multiple hyperparameter configurations tested
- ✅ Cross-validation results documented
- ✅ Test set evaluation with confusion matrix
- ✅ Ensemble method implemented
- ✅ Results clearly visualized and discussed

### Overall Project:
- ✅ All 4 approaches completed = 100% possible score
- ✅ Meaningful comparisons across approaches
- ✅ Insights about what works and why
- ✅ Reproducible results (fixed random seed)
- ✅ Professional academic report

