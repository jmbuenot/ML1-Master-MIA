# System Patterns: Architecture and Design

## System Architecture

### High-Level Structure
```
┌─────────────────────────────────────────────────────────┐
│                      main.jl                            │
│                 (Orchestration Layer)                   │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Data    │   │  Model   │   │  Utils   │
    │  Layer   │   │  Layer   │   │  Layer   │
    └──────────┘   └──────────┘   └──────────┘
          │               │               │
          │               │               │
    [datasets/]    [ML Models]    [utils_ML1.jl]
```

## Key Components

### 1. Data Pipeline
```julia
# Expected flow:
Raw CSV → Load → Preprocess → Feature Engineering → Train/Test Split
```

**Responsibilities:**
- Load data from datasets/ folder
- Create binary classification target
- Handle feature extraction
- Normalize/standardize features
- Split into training and test sets

### 2. Model Training Pipeline
```julia
# For each ML technique:
Training Data → Cross-Validation (modelCrossValidation) → 
    Best Parameters → Final Training → Test Evaluation
```

**Responsibilities:**
- Parameter tuning via cross-validation
- Model training with optimal parameters
- Performance evaluation
- Confusion matrix generation

### 3. Utility Layer (utils_ML1.jl)
Expected to contain:
- `modelCrossValidation`: Cross-validation function
- Data normalization functions
- Model training wrappers for each technique
- Evaluation metric calculations
- Confusion matrix utilities

## Design Patterns

### Pattern 1: Experiment Runner
Each approach should follow consistent structure:
```julia
# 1. Data Setup
data = load_and_preprocess()

# 2. Experimentation Loop
for each_model_type
    for each_hyperparameter_set
        results = cross_validate(model, params, data)
        store_results(results)
    end
end

# 3. Best Model Selection
best_params = select_best(results)

# 4. Final Evaluation
final_model = train_final(best_params, training_data)
test_results = evaluate(final_model, test_data)
```

### Pattern 2: Reproducible Research
```julia
# Always start with:
using Random
Random.seed!(42)  # Or chosen seed value

# Document all parameters
const RANDOM_SEED = 42
const TEST_SIZE = 0.2
const CV_FOLDS = 10
```

### Pattern 3: Model Registry
Track all experiments systematically:
```julia
# Structure for storing results
results = Dict(
    "ANNs" => [],
    "SVMs" => [],
    "DecisionTrees" => [],
    "kNN" => [],
    "Ensemble" => []
)
```

## Critical Implementation Paths

### Path 1: ANN Training (Validation Split Required)
```julia
# ANNs need special handling:
training_data → split → train_subset + validation_subset
               ↓
        Cross-validation on train_subset
               ↓
        Train final model with validation monitoring
               ↓
        Evaluate on held-out test set
```

### Path 2: Cross-Validation Strategy
Must use `modelCrossValidation` from utils:
- Input: training data only (never test data)
- Output: performance metrics for parameter selection
- Process: k-fold split within training set
- Result: best hyperparameters for final model

### Path 3: Ensemble Creation
```julia
# Combine at least 3 individual models:
model1 = best_ann
model2 = best_svm
model3 = best_tree

# Voting strategy:
predictions = majority_vote([model1, model2, model3], test_data)
# OR
predictions = weighted_vote([model1, model2, model3], weights, test_data)
# OR
predictions = stacking_ensemble([model1, model2, model3], meta_learner, test_data)
```

## Component Relationships

### Data Dependencies
```
datasets/ → Data Loading → Feature Matrix X, Target Vector y
                ↓
        Binary Target Creation (median threshold)
                ↓
        Train/Test Split (preserve class balance)
                ↓
        Normalization (fit on train, apply to test)
```

### Model Dependencies
```
utils_ML1.jl functions
        ↓
    modelCrossValidation (parameter selection)
        ↓
    Hyperparameter Grid Search
        ↓
    Final Model Training
        ↓
    Evaluation & Confusion Matrix
```

## Technical Decisions

### Decision 1: Train/Test Split
- **Timing:** Before any modeling or cross-validation
- **Rationale:** Prevent data leakage
- **Typical Ratio:** 80/20 or 70/30

### Decision 2: Normalization
- **Timing:** After train/test split
- **Fit:** On training data only
- **Apply:** Transform both train and test
- **Methods:** To be determined (min-max, z-score, etc.)

### Decision 3: Cross-Validation
- **Scope:** Training data only
- **Purpose:** Hyperparameter tuning
- **Function:** modelCrossValidation from utils
- **Folds:** Likely 5 or 10

### Decision 4: Evaluation Metrics
- **Primary:** To be determined based on class balance
- **Options:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Required:** Confusion matrix for all final models

## Code Organization Strategy

### Modular Approach
```julia
# main.jl structure:
# 1. Imports and Setup
# 2. Data Loading Functions
# 3. Preprocessing Functions
# 4. Model Training Functions (one per technique)
# 5. Ensemble Functions
# 6. Evaluation Functions
# 7. Main Execution Block
```

### Function Naming Convention
- `load_*`: Data loading functions
- `preprocess_*`: Data preprocessing
- `train_*`: Model training
- `evaluate_*`: Model evaluation
- `plot_*`: Visualization functions

