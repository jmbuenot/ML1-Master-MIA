# Active Context: Current Work Focus

## Current Status
**Phase:** Implementation In Progress - Approaches 1 & 2 Implemented  
**Date:** November 27, 2025  
**Focus:** Approach 1 complete with results. Approach 2 fully implemented and ready to execute.

## Recent Changes
- Created memory-bank/ directory structure
- Initialized core documentation files  
- Established project understanding from task.md
- **Selected and documented 4 distinct approaches**
- Verified utils_ML1.jl exists with all required functions (1538 lines)
- Verified dataset exists (OnlineNewsPopularity.csv, 39,646 rows)
- Documented available utility functions
- **Created main.jl with complete structure for all 4 approaches**
  - Command-line argument support (julia main.jl 1-4)
  - Data loading and preprocessing functions
  - Normalization (MinMax and ZeroMean)
  - PCA support for Approach 3
  - Model training functions (ANN, SVM, Decision Tree, kNN)
  - Ensemble placeholder
  - Stratified train/test split
- **Fixed bugs in utils_ML1.jl**
  - Added missing VotingClassifier struct definition
  - Exported VotingClassifier
- **Fixed type mismatches in main.jl**
  - Changed Vector{Bool} to AbstractVector{Bool} to handle BitVector
- **Optimized hyperparameters for development**
  - Reduced numExecutions from 50 to 5
  - Reduced maxEpochs from 1000 to 100
- **✅ COMPLETED: Approach 1 cross-validation and execution**
  - Dataset: 39,644 samples, 55 features (excluding url, timedelta, shares)
  - Train/test split: 31,716 / 7,928 (80/20)
  - Target: Median split at 1200 shares (48.5% / 51.5%)
  - Best models identified via 10-fold CV
  - Results saved to results/approach1.txt
- **✅ VERIFIED: Approach 2 fully implemented in main.jl**
  - Implementation location: lines 755-801 (run_approach_2 function)
  - Feature selection: 19 features across 5 categories (lines 113-141)
  - Complies with all specifications from approaches.md
  - Uses same pipeline as Approach 1 (stratified split, MinMax normalization, 10-fold CV)
  - All 4 model types + ensemble implemented
  - Ready to execute with: julia main.jl 2

## Current Work Focus

### Immediate Priorities
1. **✅ Project Setup** - COMPLETE
   - ✅ Verified folder structure (datasets/, utils/ exist)
   - ✅ Verified utils_ML1.jl exists (all functions available)
   - ✅ Verified dataset exists (OnlineNewsPopularity.csv)
   - ✅ Created main.jl structure
   - ✅ Fixed bugs in utils and main

2. **✅ Approach 1 Execution** - COMPLETE
   - ✅ Data loading and preprocessing working
   - ✅ 55 features extracted and normalized (MinMax)
   - ✅ Stratified train/test split (31,716 / 7,928)
   - ✅ ANNs: 8 topologies tested, best [50,25] @ 62.88%
   - ✅ SVMs: 8 configs tested, best poly-3 @ 63.43%
   - ✅ Decision Trees: 6 depths tested, best depth-7 @ 69.58%
   - ✅ kNN: 6 k values tested, best k=9 @ 59.78%
   - ✅ Final models trained and evaluated on test set
   - ✅ Ensemble method with majority voting
   - ✅ Results saved to results/approach1.txt

3. **✅ Approach 2 Implementation** - COMPLETE, READY TO RUN
   - ✅ Feature selection: 19 features selected
     - Text & Content: n_tokens_title, n_tokens_content, num_keywords, average_token_length
     - Multimedia: num_hrefs, num_imgs, num_videos, num_self_hrefs
     - Sentiment: global_sentiment_polarity, global_subjectivity, title_sentiment_polarity, rate_positive_words, avg_positive_polarity
     - Topic: LDA_00, LDA_01, LDA_02
     - Temporal: is_weekend, weekday_is_monday
   - ✅ Same pipeline as Approach 1 (MinMax, median split, 10-fold CV)
   - ✅ All utils_ML1.jl functions properly utilized
   - ⏳ READY TO EXECUTE: julia main.jl 2

4. **Next Steps** - PENDING
   - ⏳ Execute Approach 2 and save results
   - ⏳ Implement Approach 3 (PCA with 20 components)
   - ⏳ Implement Approach 4 (Imbalanced classification)
   - ⏳ Add visualization/plotting functions
   - ⏳ Comparative analysis across approaches

## Next Steps

### Short-term (Immediate)
1. **Execute Approach 2** - Run `julia main.jl 2` to generate results
2. Save Approach 2 results to results/approach2.txt
3. Compare Approach 2 vs Approach 1 performance
4. Document insights about feature selection impact

### Medium-term
1. **Implement Approach 3** - PCA dimensionality reduction (already templated in main.jl)
2. **Implement Approach 4** - Imbalanced classification (already templated in main.jl)
3. Execute both approaches and collect results
4. Generate comparative visualizations

### Long-term
1. Perform comprehensive statistical comparisons across all 4 approaches
2. Generate all plots and confusion matrices for report
3. Write complete report with citations (3+ papers)
4. Final verification and reproducibility testing

## Active Decisions

### ✅ DECIDED: Four Approaches Selected

**Approach 1 – Baseline**
- Use all 58 available features
- Apply MinMax normalization
- Binary target: median split of shares (balanced 50/50)
- Train all 4 required models (ANN, SVM, Decision Trees, kNN)

**Approach 2 – Feature Selection**
- Select subset of 15-20 meaningful features (text-related, sentiment, LDA topics, weekday)
- Apply MinMax normalization
- Binary target: median split (balanced 50/50)
- Train all 4 ML methods with reduced feature set

**Approach 3 – PCA Dimensionality Reduction**
- Start with full 58 features
- Apply ZeroMean (z-score) normalization (required for PCA)
- Perform PCA, reduce to 10 or 20 principal components
- Binary target: median split (balanced 50/50)
- Train all 4 models on PCA-transformed features

**Approach 4 – Imbalanced Classification**
- Use all 58 features
- Apply ZeroMean (z-score) normalization
- Binary target: 75th percentile threshold (imbalanced 25% positive / 75% negative)
- Train all 4 ML methods with class imbalance consideration
- Tests "highly viral" vs rest (more challenging, practical problem)

### ✅ Decisions Made
- **Cross-Validation:** ✅ 10-fold (CV_FOLDS = 10)
- **Feature Selection Details:** ✅ 19 features selected for Approach 2 (text, sentiment, topics, temporal)
- **PCA Components:** ✅ 20 components for Approach 3
- **Random Seed:** ✅ 42
- **Test Split:** ✅ 20% (0.2)
- **ANN Parameters:** ✅ numExecutions=5, maxEpochs=100, learningRate=0.01, validationRatio=0.2

### Pending Decisions
- **PCA Components Alternative:** Should we also try 10 components for comparison?
- **Ensemble Strategy:** Majority voting vs weighted voting vs stacking?
- **Final Hyperparameters:** Increase numExecutions and maxEpochs for final report?

### Established Patterns
- Use functions from utils_ML1.jl exclusively
- Fixed random seed for reproducibility
- Train/test split before any modeling
- Cross-validation only on training set
- Confusion matrices for all final models

## Important Considerations

### Technical Constraints
- Must work with Julia language
- Must use provided utils_ML1.jl functions
- Dataset has no missing values (simplifies preprocessing)
- Binary classification target (simplifies evaluation)

### Academic Requirements
- Minimum 3 scientific publications in bibliography
- Formal citation style required
- Statistical significance testing for comparisons
- Clear justification for all methodological choices

## ✅ Questions Resolved

1. **Does utils_ML1.jl already exist with required functions?**  
   ✅ YES - File exists with 1538 lines of code

2. **Is the dataset already downloaded in datasets/ folder?**  
   ✅ YES - OnlineNewsPopularity.csv exists (39,646 rows)

3. **What specific functions are available in utils_ML1.jl?**  
   ✅ Key functions identified:
   - **Normalization:** `calculateMinMaxNormalizationParameters`, `normalizeMinMax`, `calculateZeroMeanNormalizationParameters`, `normalizeZeroMean`
   - **Cross-validation:** `modelCrossValidation` (supports :ANN, :SVMClassifier, :DecisionTreeClassifier, :KNeighborsClassifier)
   - **ANN:** `buildClassANN`, `trainClassANN`, `ANNCrossValidation`
   - **Evaluation:** `confusionMatrix`, `printConfusionMatrix`, `accuracy`
   - **Data:** `oneHotEncoding`, `holdOut`
   - **Ensemble:** `VotingClassifier` available

4. **Should we focus on one comprehensive approach or multiple simpler approaches?**  
   ✅ DECIDED - Four distinct approaches selected (see Active Decisions)

## Learnings and Insights

### Project Structure
- Project follows academic ML best practices
- Emphasis on reproducibility and scientific rigor
- Report quality as important as code implementation
- Multiple opportunities to score (25% per approach) allows flexible strategy

### Technical Insights from Approach 1
- **Dataset:** 39,644 articles, 55 predictive features (url, timedelta, shares excluded)
- **Class Balance:** Near-perfect balance with median split (48.5% / 51.5%)
- **Best Model:** Decision Trees (69.58% CV accuracy)
  - Decision Trees significantly outperform other methods for this dataset
  - Suggests non-linear relationships and feature interactions are important
- **Model Rankings (CV Accuracy):**
  1. Decision Trees (depth 7): 69.58%
  2. SVMs (poly kernel deg 3): 63.43%
  3. ANNs ([50,25] topology): 62.88%
  4. kNN (k=9): 59.78%
- **ANN Performance:** Relatively consistent across topologies (61-63%)
  - Best: [50,25] with 62.88%
  - Suggests problem may not benefit much from deep architectures
- **SVM Kernels:** Polynomial > Linear > RBF for this dataset
- **Decision Tree Depth:** Overfitting occurs after depth 10 (CV accuracy drops)
- **kNN:** Poorest performer, suggesting distance-based methods struggle with this feature space

### Performance Considerations
- With reduced parameters (5 executions, 100 epochs), Approach 1 runs in reasonable time
- 10-fold CV provides good variance estimates (std around 1%)
- Stratified split maintains class balance across train/test sets

