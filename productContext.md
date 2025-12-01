# Product Context: Online News Popularity Prediction

## Purpose
This ML system serves as a comprehensive demonstration of machine learning capabilities for predicting online content popularity. It's designed as a final project to showcase proficiency in:
- Multiple ML algorithms
- Experimental methodology
- Scientific reporting
- Code organization and reproducibility

## Problem Being Solved

### Primary Challenge
Predict whether an online news article will be "popular" (high social media shares) based on its content and metadata characteristics before or shortly after publication.

### Real-World Context
- **Business Value:** Content creators and news platforms need to understand what makes articles viral
- **Decision Support:** Helps editorial teams prioritize content promotion
- **Pattern Discovery:** Reveals relationships between article features and audience engagement

### Why Binary Classification?
Instead of predicting exact share counts (regression), we classify articles as:
- **Popular** (above median shares) - worthy of promotion investment
- **Not Popular** (at or below median) - standard treatment

This simplifies the problem while maintaining practical value for decision-making.

## Dataset Characteristics

### Feature Categories

**Content Features**
- Word counts (title, content)
- Token rates (unique, non-stop words)
- Multimedia elements (links, images, videos)
- Keywords metadata

**Channel Information**
- Data channel indicators (Lifestyle, Entertainment, Business, Social Media, Tech, World)

**Timing Features**
- Weekday published
- Weekend indicator

**Topic Modeling**
- LDA topic closeness (5 topics: LDA_00 through LDA_04)

**Sentiment Analysis**
- Global sentiment polarity and subjectivity
- Positive/negative word rates
- Title sentiment features

**Reference Metrics**
- Self-referencing shares statistics
- Keyword performance metrics

## How It Should Work

### User Experience Flow
1. **Data Loading:** Seamless import of CSV dataset
2. **Preprocessing:** Automated feature engineering and normalization
3. **Experimentation:** Systematic testing of multiple ML models
4. **Comparison:** Clear visualization of model performance
5. **Results:** Reproducible outputs matching report figures

### Expected Outputs
- Training and test accuracy metrics
- Confusion matrices for each model
- Cross-validation results for parameter selection
- Ensemble model performance
- Comparative analysis across approaches

## Quality Goals

### Code Quality
- **Reproducibility:** Same results every run (fixed random seed)
- **Modularity:** Reusable functions from utils_ML1.jl
- **Clarity:** Well-organized, commented code
- **Completeness:** Executable from start to finish

### Scientific Rigor
- **Methodology:** Proper train/test splits, cross-validation
- **Statistical Testing:** Significance tests for model comparisons
- **Documentation:** Clear explanation of all decisions
- **Citations:** Grounded in published research

### Reporting Excellence
- **Clarity:** Non-technical readers can understand approach
- **Depth:** Technical readers can reproduce work
- **Visualization:** Effective plots and confusion matrices
- **Academic Style:** Formal, properly cited

## Success Indicators
- All four ML techniques successfully implemented for each approach
- Ensemble methods show improvement or insights
- Results are statistically analyzed and clearly presented
- Code runs without errors and reproduces report results
- Report meets academic publication standards

