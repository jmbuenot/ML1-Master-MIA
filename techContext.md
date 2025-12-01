# Technical Context: Technologies and Setup

## Core Technologies

### Programming Language
**Julia**
- Version: To be determined (check project environment)
- Reason: High-performance numerical computing
- ML ecosystem: Native ML libraries

### Required Packages (Expected)
```julia
# Data manipulation
using DataFrames
using CSV
using Statistics

# Machine Learning
using Flux           # For ANNs
using ScikitLearn    # For SVMs, Decision Trees, kNN
# OR
using DecisionTree   # Alternative for decision trees
using NearestNeighbors  # Alternative for kNN

# Utilities
using Random         # For reproducibility
using LinearAlgebra  # For matrix operations

# Visualization (likely needed)
using Plots
# OR
using PyPlot
# OR  
using Makie
```

## Development Setup

### Project Structure
```
/home/azaliia/projects/ML1/ML_final_project/
├── main.jl              # Main executable script
├── Report.pdf           # Final report (to be created)
├── task.md              # Project specification
├── .gitignore           # Git ignore rules
├── .clinerules          # Memory bank system instructions
├── memory-bank/         # Project documentation (this system)
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── activeContext.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   └── progress.md
├── datasets/            # Data files (to be verified)
│   └── OnlineNewsPopularity.csv (expected)
└── utils/               # Auxiliary code (to be verified)
    └── utils_ML1.jl     # Required utility functions
```

### Environment Setup
```julia
# Likely need Project.toml or similar for dependencies
# Check if exists, create if needed
# Activate project environment
# ] activate .
```

## Technical Constraints

### Must Use utils_ML1.jl
All ML operations should use functions from the provided auxiliary code:
- Cannot import external model training code directly
- Must wrap/adapt to use provided functions
- Ensures consistency with course materials

### ✅ Verified Functions in utils_ML1.jl (1538 lines)

1. **modelCrossValidation**
   - Purpose: Cross-validation for parameter selection
   - Signature: `modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple, crossValidationIndices::Array{Int64,1})`
   - Supported models: `:ANN`, `:SVMClassifier`, `:DecisionTreeClassifier`, `:KNeighborsClassifier`
   - Returns: Tuple of (accuracy, error, sensitivity, specificity, ppv, npv, f1, confusion_matrix)
   - Each metric returns (mean, std) except confusion matrix

2. **Normalization Functions**
   - `calculateMinMaxNormalizationParameters(dataset)` - returns (min, max) per column
   - `calculateZeroMeanNormalizationParameters(dataset)` - returns (mean, std) per column
   - `normalizeMinMax!(dataset, params)` - in-place min-max normalization
   - `normalizeMinMax(dataset, params)` - returns normalized copy
   - `normalizeZeroMean!(dataset, params)` - in-place z-score normalization
   - `normalizeZeroMean(dataset, params)` - returns normalized copy

3. **ANN-Specific Functions**
   - `buildClassANN(numInputs, topology, numOutputs; transferFunctions)` - builds ANN architecture
   - `trainClassANN(topology, dataset; validationDataset, testDataset, ...)` - trains ANN
   - `ANNCrossValidation(topology, dataset, crossValidationIndices; ...)` - ANN cross-validation

4. **Evaluation Utilities**
   - `confusionMatrix(predictions, targets, classes)` - returns 8-tuple of metrics
   - `printConfusionMatrix(confusion)` - pretty prints confusion matrix
   - `accuracy(outputs, targets)` - calculates accuracy

5. **Data Utilities**
   - `oneHotEncoding(feature, classes)` - encodes categorical variables
   - `holdOut(N, P)` - creates train/test splits
   - `classifyOutputs(outputs)` - converts probabilities to class predictions

6. **Ensemble Methods**
   - `VotingClassifier(models, voting, weights)` - ensemble voting classifier

## Data Format

### Input Format
```
CSV file with columns:
- url (non-predictive)
- timedelta (non-predictive)  
- 58 predictive features (numeric)
- shares (target, numeric)
```

### Expected Data Structure
```julia
# After loading:
df = DataFrame(...)  # 39,644 rows × 61 columns

# Feature matrix
X = Matrix{Float64}  # 39,644 × 58 (excluding non-predictive + target)

# Binary target
y = Vector{Bool}     # 39,644 × 1 (popular vs not popular)
```

## Model Specifications

### 1. Artificial Neural Networks (ANNs)
**Framework:** Likely Flux.jl
**Requirements:**
- Test 8+ architectures
- 1-2 hidden layers
- Validation split within training

**Architecture Examples:**
```julia
# Single hidden layer variations:
[58, 10, 1]    # 10 neurons
[58, 20, 1]    # 20 neurons
[58, 50, 1]    # 50 neurons
[58, 100, 1]   # 100 neurons

# Two hidden layer variations:
[58, 20, 10, 1]
[58, 50, 20, 1]
[58, 100, 50, 1]
[58, 50, 50, 1]
```

### 2. Support Vector Machines (SVMs)
**Framework:** Likely ScikitLearn.jl
**Requirements:**
- Test 8+ configurations
- Vary kernels and C values

**Configuration Examples:**
```julia
# Linear kernel
C ∈ [0.1, 1.0, 10.0, 100.0]

# RBF kernel  
C ∈ [0.1, 1.0, 10.0]
γ ∈ [0.001, 0.01, 0.1]

# Polynomial kernel
degree ∈ [2, 3]
C ∈ [1.0, 10.0]
```

### 3. Decision Trees
**Framework:** DecisionTree.jl or ScikitLearn.jl
**Requirements:**
- Test 6+ maximum depths

**Depth Examples:**
```julia
max_depth ∈ [3, 5, 7, 10, 15, 20]
# OR
max_depth ∈ [2, 4, 6, 8, 10, 12]
```

### 4. k-Nearest Neighbors (kNN)
**Framework:** NearestNeighbors.jl or ScikitLearn.jl
**Requirements:**
- Test 6+ k values

**k Examples:**
```julia
k ∈ [1, 3, 5, 7, 9, 11]
# OR
k ∈ [3, 5, 10, 15, 20, 25]
```

## Reproducibility Requirements

### Random Seed
```julia
using Random
Random.seed!(SEED_VALUE)

# Set before:
# - Train/test split
# - Cross-validation splits
# - ANN weight initialization
# - Any stochastic operations
```

### Documentation Requirements
All parameters must be documented:
- Seed value
- Train/test split ratio
- Cross-validation folds
- Normalization method and parameters
- All hyperparameters tested
- Final selected hyperparameters

## Performance Considerations

### Julia-Specific Optimizations
- Use in-place operations where possible
- Leverage vectorization
- Avoid global variables in hot loops
- Pre-allocate arrays when size is known

### Memory Management
- Large dataset (40K rows × 58 features)
- Consider memory usage in cross-validation
- May need to manage intermediate results

## Development Workflow

### Iteration Process
1. Develop in main.jl
2. Test small sections incrementally
3. Document results as they emerge
4. Save intermediate results
5. Generate plots and confusion matrices
6. Update memory bank with findings

### Debugging Strategy
- Use small data subset for rapid iteration
- Test each model type independently
- Verify cross-validation logic carefully
- Check for data leakage (test data in training)
- Validate confusion matrix calculations

## Tool Usage Patterns

### Data Inspection
```julia
# Use DataFrames exploration
describe(df)
first(df, 5)
names(df)
size(df)
```

### Visualization
```julia
# Plot training curves (ANNs)
# Plot confusion matrices
# Plot feature importance (if applicable)
# Plot cross-validation results
```

## ✅ Verified Information
- ✅ utils_ML1.jl exists with 1538 lines
- ✅ Dataset exists: OnlineNewsPopularity.csv (39,646 rows)
- ✅ All required functions available and documented above
- ✅ Module uses: Flux, MLJ, LIBSVM, DecisionTree, NearestNeighborModels

## Unknown/To Be Determined
- Exact Julia version (can check when running code)
- Exact package versions (check Project.toml if exists)

