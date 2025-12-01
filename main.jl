import Pkg
using Pkg
Pkg.activate(".") 
"""
Online News Popularity Prediction - ML Final Project
====================================================

Usage:
    julia main.jl <approach_number>
    
    approach_number: 1, 2, 3, or 4
        1 - Baseline (all features, MinMax, median split)
        2 - Feature Selection (15-20 features, MinMax, median split)
        3 - PCA (10-20 components, ZeroMean, median split)
        4 - Imbalanced (all features, ZeroMean, 75th percentile split)

Example:
    julia main.jl 1
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

using CSV
using DataFrames
using Statistics
using Random
using LinearAlgebra
using MultivariateStats  # For PCA
using Plots
using MLJ
using LIBSVM
using CategoricalArrays
using MLJModelInterface

# Import utility functions
include("utils/utils_ML1.jl")
using .UtilsML1

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

const RANDOM_SEED = 42
const TEST_SIZE = 0.2  # 20% for testing
const CV_FOLDS = 10    # 10-fold cross-validation
const DATASET_PATH = "datasets/OnlineNewsPopularity.csv"
const DEV_MODE = true  # Set to false for final run
const DEV_SAMPLE_SIZE = 5000  # Use smaller subset for testing

# ============================================================================
# DATA LOADING AND BASIC PREPROCESSING
# ============================================================================

"""
Load the Online News Popularity dataset.
Returns DataFrame with all columns.
"""
function load_data(filepath::String)
    println("\n" * "="^70)
    println("Loading dataset from: $filepath")
    println("="^70)
    
    df = CSV.read(filepath, DataFrame)
    if DEV_MODE
        df = df[1:min(DEV_SAMPLE_SIZE, nrow(df)), :]
        println("⚠️  DEV MODE: Using only $(nrow(df)) samples")
    end
    println("Dataset loaded successfully!")
    println("  Total samples: $(nrow(df))")
    println("  Total columns: $(ncol(df))")
    
    return df
end

"""
Prepare features and target for a given approach.
Returns (X, y, feature_names) where:
    - X: feature matrix
    - y: binary target vector
    - feature_names: names of features used
"""
function prepare_data(df::DataFrame, approach::Int)
    println("\n" * "="^70)
    println("Preparing data for Approach $approach")
    println("="^70)
    
    # Get all feature columns (exclude only url, timedelta, and the target "shares")
    # Note: We must NOT exclude self_reference_*_shares columns (they are valid features)
    all_cols = names(df)
    
    # Find the exact target column (can be "shares" or " shares")
    target_col_idx = findfirst(col -> strip(col) == "shares", all_cols)
    @assert target_col_idx !== nothing "Target column 'shares' not found!"
    target_col_name = all_cols[target_col_idx]
    
    # Filter: exclude only url, timedelta, and the exact target column
    feature_cols = filter(col -> 
        strip(col) != "url" && 
        strip(col) != "timedelta" && 
        col != target_col_name,  # Only exclude the exact target column, not columns containing "shares"
        all_cols)
    
    if approach == 1
        # Approach 1: Use all features
        println("Approach 1: Using all $(length(feature_cols)) features")
        X = Matrix{Float64}(df[:, feature_cols])
        feature_names = feature_cols
        
    elseif approach == 2
        # Approach 2: Select 15-20 important features
        selected_features = [
            # Text & Content (4 features)
            " n_tokens_title",
            " n_tokens_content", 
            " num_keywords",
            " average_token_length",
            
            # Multimedia & Engagement (4 features)
            " num_hrefs",
            " num_imgs",
            " num_videos",
            " num_self_hrefs",
            
            # Sentiment & Tone (5 features)
            " global_sentiment_polarity",
            " global_subjectivity",
            " title_sentiment_polarity",
            " rate_positive_words",
            " avg_positive_polarity",
            
            # Topic (3 features)
            " LDA_00",
            " LDA_01", 
            " LDA_02",
            
            # Temporal (2 features)
            " is_weekend",
            " weekday_is_monday"
        ]
        
        # Filter to only include columns that exist
        selected_features = filter(f -> f in feature_cols, selected_features)
        
        println("Approach 2: Using $(length(selected_features)) selected features")
        X = Matrix{Float64}(df[:, selected_features])
        feature_names = selected_features
        
    elseif approach == 3
        # Approach 3: Will apply PCA (start with all features)
        println("Approach 3: Starting with all $(length(feature_cols)) features")
        println("            PCA will be applied after normalization")
        X = Matrix{Float64}(df[:, feature_cols])
        feature_names = feature_cols  # Will be replaced with PC names after PCA
        
    elseif approach == 4
        # Approach 4: Use all features (like Approach 1)
        println("Approach 4: Using all $(length(feature_cols)) features")
        X = Matrix{Float64}(df[:, feature_cols])
        feature_names = feature_cols
        
    else
        error("Invalid approach number: $approach. Must be 1, 2, 3, or 4.")
    end
    
    # Create target variable based on approach
    shares = df[:, target_col_name]
    
    if approach == 4
        # Approach 4: 75th percentile threshold (imbalanced)
        threshold = quantile(shares, 0.75)
        y = shares .> threshold
        println("Target: 75th percentile threshold = $threshold")
        println("        Positive class (highly viral): $(sum(y)) ($(round(100*mean(y), digits=1))%)")
        println("        Negative class: $(sum(.!y)) ($(round(100*mean(.!y), digits=1))%)")
    else
        # Approaches 1, 2, 3: Median split (balanced)
        threshold = median(shares)
        y = shares .> threshold
        println("Target: Median threshold = $threshold")
        println("        Positive class (popular): $(sum(y)) ($(round(100*mean(y), digits=1))%)")
        println("        Negative class: $(sum(.!y)) ($(round(100*mean(.!y), digits=1))%)")
    end
    
    return X, y, feature_names
end

"""
Split data into training and test sets with stratification.
"""
function split_train_test(X::Matrix{Float64}, y::AbstractVector{Bool}, test_size::Float64)
    println("\n" * "="^70)
    println("Splitting data into train/test sets")
    println("="^70)
    
    Random.seed!(RANDOM_SEED)
    
    n_samples = size(X, 1)
    n_test = Int(round(n_samples * test_size))
    
    # Stratified split - separate positive and negative samples
    pos_indices = findall(y)
    neg_indices = findall(.!y)
    
    # Shuffle each class
    shuffle!(pos_indices)
    shuffle!(neg_indices)
    
    # Split each class
    n_test_pos = Int(round(length(pos_indices) * test_size))
    n_test_neg = Int(round(length(neg_indices) * test_size))
    
    test_pos = pos_indices[1:n_test_pos]
    train_pos = pos_indices[n_test_pos+1:end]
    
    test_neg = neg_indices[1:n_test_neg]
    train_neg = neg_indices[n_test_neg+1:end]
    
    # Combine and shuffle
    train_indices = vcat(train_pos, train_neg)
    test_indices = vcat(test_pos, test_neg)
    
    shuffle!(train_indices)
    shuffle!(test_indices)
    
    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    println("Training set: $(size(X_train, 1)) samples")
    println("  Positive: $(sum(y_train)) ($(round(100*mean(y_train), digits=1))%)")
    println("  Negative: $(sum(.!y_train)) ($(round(100*mean(.!y_train), digits=1))%)")
    println("Test set: $(size(X_test, 1)) samples")
    println("  Positive: $(sum(y_test)) ($(round(100*mean(y_test), digits=1))%)")
    println("  Negative: $(sum(.!y_test)) ($(round(100*mean(.!y_test), digits=1))%)")
    
    return X_train, X_test, y_train, y_test
end

"""
Apply normalization based on approach.
"""
function apply_normalization(X_train::Matrix{Float64}, X_test::Matrix{Float64}, approach::Int)
    println("\n" * "="^70)
    println("Applying normalization")
    println("="^70)
    
    if approach == 1 || approach == 2
        # MinMax normalization for Approaches 1 and 2
        println("Method: MinMax normalization [0, 1]")
        params = calculateMinMaxNormalizationParameters(X_train)
        X_train_norm = normalizeMinMax(X_train, params)
        X_test_norm = normalizeMinMax(X_test, params)
        
    elseif approach == 3 || approach == 4
        # Zero-mean (z-score) normalization for Approaches 3 and 4
        println("Method: ZeroMean (z-score) normalization")
        params = calculateZeroMeanNormalizationParameters(X_train)
        X_train_norm = normalizeZeroMean(X_train, params)
        X_test_norm = normalizeZeroMean(X_test, params)
        
    end
    
    println("Normalization complete!")
    
    return X_train_norm, X_test_norm
end

"""
Apply PCA transformation (Approach 3 only).
"""
function apply_pca(X_train::Matrix{Float64}, X_test::Matrix{Float64}, n_components::Int=20)
    println("\n" * "="^70)
    println("Applying PCA")
    println("="^70)
    
    # Fit PCA on training data
    pca_model = fit(PCA, X_train'; maxoutdim=n_components, pratio=1.0)
    
    # Transform both train and test
    X_train_pca = MultivariateStats.transform(pca_model, X_train')'
    X_test_pca = MultivariateStats.transform(pca_model, X_test')'
    
    # Get explained variance
    explained_var = principalvars(pca_model)
    total_var = var(pca_model)
    explained_ratio = explained_var ./ total_var
    cumulative_var = cumsum(explained_ratio)
    
    println("PCA complete!")
    println("  Components: $n_components")
    println("  Explained variance: $(round(100*cumulative_var[end], digits=2))%")
    println("  First 5 components explain: $(round(100*cumulative_var[min(5, n_components)], digits=2))%")
    
    return X_train_pca, X_test_pca, pca_model
end

"""
Create cross-validation indices for k-fold CV.
"""
function create_cv_indices(n_samples::Int, k_folds::Int)
    Random.seed!(RANDOM_SEED)
    
    indices = repeat(1:k_folds, outer=ceil(Int, n_samples/k_folds))
    indices = indices[1:n_samples]
    shuffle!(indices)
    
    return indices
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
Train a final model on full training set and evaluate on test set.
Reuses model creation logic from modelCrossValidation.
"""
function train_and_evaluate_final_model(modelType::Symbol, best_params::Dict,
                                       X_train, y_train, X_test, y_test)
    
    # Load MLJ models
    SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
    kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
    DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    
    # Convert targets to strings
    y_train_str = string.(y_train)
    y_test_str = string.(y_test)
    
    # Create model using same logic as modelCrossValidation
    if modelType == :SVMClassifier
        kernel_str = get(best_params, "kernel", "rbf")
        C = Float64(get(best_params, "C", 1.0))
        gamma = Float64(get(best_params, "gamma", 0.1))
        degree = Int32(get(best_params, "degree", 3))
        coef0 = Float64(get(best_params, "coef0", 0.0))
        
        if kernel_str == "linear"
            model = SVMClassifier(kernel = LIBSVM.Kernel.Linear, cost = C)
        elseif kernel_str == "poly" || kernel_str == "polynomial"
            model = SVMClassifier(kernel = LIBSVM.Kernel.Polynomial, cost = C, 
                                gamma = gamma, degree = degree, coef0 = coef0)
        elseif kernel_str == "sigmoid"
            model = SVMClassifier(kernel = LIBSVM.Kernel.Sigmoid, cost = C, 
                                gamma = gamma, coef0 = coef0)
        else  # rbf
            model = SVMClassifier(kernel = LIBSVM.Kernel.RadialBasis, cost = C, gamma = gamma)
        end
        
    elseif modelType == :DecisionTreeClassifier
        max_depth = get(best_params, "max_depth", -1)
        model = DTClassifier(max_depth = max_depth, rng = Random.MersenneTwister(RANDOM_SEED))
        
    elseif modelType == :KNeighborsClassifier
        K = get(best_params, "K", 3)
        model = kNNClassifier(K = K)
    else
        error("Unknown model type: $modelType")
    end
    
    # Train on full training set
    mach = MLJ.machine(model, MLJ.table(X_train), categorical(y_train_str))
    MLJ.fit!(mach, verbosity=0)
    
    # Predict on test set
    if modelType == :SVMClassifier
        predictions = string.(MLJ.predict(mach, MLJ.table(X_test)))
    else
        predictions = string.(mode.(MLJ.predict(mach, MLJ.table(X_test))))
    end
    
    # Calculate metrics
    classes = unique(y_train_str)
    test_metrics = confusionMatrix(predictions, y_test_str, classes)
    test_accuracy = test_metrics[1]
    confusion = test_metrics[8]
    
    return mach, test_accuracy, confusion
end

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

"""
Train and evaluate ANNs with different topologies.
"""
function train_anns(X_train::Matrix{Float64}, y_train::AbstractVector{Bool}, 
                   X_test::Matrix{Float64}, y_test::AbstractVector{Bool}, cv_indices::Vector{Int})
    println("\n" * "="^70)
    println("TRAINING ARTIFICIAL NEURAL NETWORKS (ANNs)")
    println("="^70)
    
    n_inputs = size(X_train, 2)
    
    # Define topologies to test (8+ architectures)
    topologies = [
        [10],           # Single layer: 10 neurons
        [20],           # Single layer: 20 neurons
        [50],           # Single layer: 50 neurons
        [100],          # Single layer: 100 neurons
        [20, 10],       # Two layers: 20 -> 10
        [50, 25],       # Two layers: 50 -> 25
        [100, 50],      # Two layers: 100 -> 50
        [50, 50],       # Two layers: 50 -> 50
    ]
    
    results = []
    
    for (idx, topology) in enumerate(topologies)
        println("\nTopology $idx: $topology")
        
        # Prepare dataset for ANNCrossValidation
        dataset = (X_train, y_train)
        
        # Cross-validation
        params = Dict(
            :topology => topology,
            :learningRate => 0.01,
            :validationRatio => 0.2,
            :numExecutions => 5,
            :maxEpochs => 100,
            :maxEpochsVal => 20
        )
        
        cv_results = modelCrossValidation(:ANN, params, dataset, cv_indices)
        
        acc_mean, acc_std = cv_results[1]
        println("  CV Accuracy: $(round(acc_mean*100, digits=2))% ± $(round(acc_std*100, digits=2))%")
        
        push!(results, (topology=topology, cv_accuracy=acc_mean, cv_std=acc_std))
    end
    
    # Select best topology
    best_idx = argmax([r.cv_accuracy for r in results])
    best_topology = results[best_idx].topology
    
    println("\n" * "-"^70)
    println("Best ANN topology: $best_topology")
    println("  CV Accuracy: $(round(results[best_idx].cv_accuracy*100, digits=2))%")
    
    # Train final model on full training set using trainClassANN
    println("\nTraining final ANN model on full training set...")
    dataset_train = (X_train, y_train)
    dataset_test = (X_test, y_test)
    
    # trainClassANN returns (model, trainingLosses, validationLosses, testLosses)
    ann_model, train_losses, val_losses, test_losses = trainClassANN(
        best_topology,
        dataset_train;
        testDataset = dataset_test,
        maxEpochs = 100,
        learningRate = 0.01,
        maxEpochsVal = 20,
        showText = false
    )
    
    # Evaluate on test set
    test_outputs = ann_model(X_test')'
    test_predictions = vec(test_outputs .> 0.5)
    test_metrics = confusionMatrix(test_predictions, y_test)
    test_accuracy = test_metrics[1]
    confusion = test_metrics[8]
    
    println("Test Set Evaluation:")
    println("  Accuracy: $(round(test_accuracy*100, digits=2))%")
    println("  Confusion Matrix:")
    printConfusionMatrix(test_predictions, y_test)
    
    return results, best_topology, ann_model, test_accuracy, confusion
end

"""
Train and evaluate SVMs with different configurations.
"""
function train_svms(X_train::Matrix{Float64}, y_train::AbstractVector{Bool}, 
                   X_test::Matrix{Float64}, y_test::AbstractVector{Bool}, cv_indices::Vector{Int})
    println("\n" * "="^70)
    println("TRAINING SUPPORT VECTOR MACHINES (SVMs)")
    println("="^70)
    
    # Define configurations to test (8+ configurations)
    configurations = [
        Dict("kernel" => "linear", "C" => 0.1),
        Dict("kernel" => "linear", "C" => 1.0),
        Dict("kernel" => "linear", "C" => 10.0),
        Dict("kernel" => "rbf", "C" => 1.0, "gamma" => 0.001),
        Dict("kernel" => "rbf", "C" => 1.0, "gamma" => 0.01),
        Dict("kernel" => "rbf", "C" => 10.0, "gamma" => 0.01),
        Dict("kernel" => "poly", "C" => 1.0, "degree" => 2),
        Dict("kernel" => "poly", "C" => 1.0, "degree" => 3),
    ]
    
    results = []
    dataset = (X_train, y_train)
    
    for (idx, config) in enumerate(configurations)
        println("\nConfiguration $idx: $(config)")
        
        cv_results = modelCrossValidation(:SVMClassifier, config, dataset, cv_indices)
        
        acc_mean, acc_std = cv_results[1]
        println("  CV Accuracy: $(round(acc_mean*100, digits=2))% ± $(round(acc_std*100, digits=2))%")
        
        push!(results, (config=config, cv_accuracy=acc_mean, cv_std=acc_std))
    end
    
    # Select best configuration
    best_idx = argmax([r.cv_accuracy for r in results])
    best_config = results[best_idx].config
    
    println("\n" * "-"^70)
    println("Best SVM configuration: $best_config")
    println("  CV Accuracy: $(round(results[best_idx].cv_accuracy*100, digits=2))%")
    
    # Train final model on full training set using helper function
    println("\nTraining final SVM model on full training set...")
    mach, test_accuracy, confusion = train_and_evaluate_final_model(
        :SVMClassifier, best_config, X_train, y_train, X_test, y_test
    )
    
    println("Test Set Evaluation:")
    println("  Accuracy: $(round(test_accuracy*100, digits=2))%")
    println("  Confusion Matrix:")
    y_test_str = string.(y_test)
    classes = unique(y_test_str)
    printConfusionMatrix(string.(MLJ.predict(mach, MLJ.table(X_test))), y_test_str, classes)
    
    return results, best_config, mach, test_accuracy, confusion
end

"""
Train and evaluate Decision Trees with different max depths.
"""
function train_decision_trees(X_train::Matrix{Float64}, y_train::AbstractVector{Bool}, 
                             X_test::Matrix{Float64}, y_test::AbstractVector{Bool}, cv_indices::Vector{Int})
    println("\n" * "="^70)
    println("TRAINING DECISION TREES")
    println("="^70)
    
    # Define max depths to test (6+ values)
    max_depths = [3, 5, 7, 10, 15, 20]
    
    results = []
    dataset = (X_train, y_train)
    
    for depth in max_depths
        println("\nMax depth: $depth")
        
        params = Dict("max_depth" => depth)
        cv_results = modelCrossValidation(:DecisionTreeClassifier, params, dataset, cv_indices)
        
        acc_mean, acc_std = cv_results[1]
        println("  CV Accuracy: $(round(acc_mean*100, digits=2))% ± $(round(acc_std*100, digits=2))%")
        
        push!(results, (max_depth=depth, cv_accuracy=acc_mean, cv_std=acc_std))
    end
    
    # Select best depth
    best_idx = argmax([r.cv_accuracy for r in results])
    best_depth = results[best_idx].max_depth
    
    println("\n" * "-"^70)
    println("Best max depth: $best_depth")
    println("  CV Accuracy: $(round(results[best_idx].cv_accuracy*100, digits=2))%")
    
    # Train final model on full training set using helper function
    println("\nTraining final Decision Tree model on full training set...")
    params = Dict("max_depth" => best_depth)
    mach, test_accuracy, confusion = train_and_evaluate_final_model(
        :DecisionTreeClassifier, params, X_train, y_train, X_test, y_test
    )
    
    println("Test Set Evaluation:")
    println("  Accuracy: $(round(test_accuracy*100, digits=2))%")
    println("  Confusion Matrix:")
    y_test_str = string.(y_test)
    classes = unique(y_test_str)
    printConfusionMatrix(string.(mode.(MLJ.predict(mach, MLJ.table(X_test)))), y_test_str, classes)
    
    return results, best_depth, mach, test_accuracy, confusion
end

"""
Train and evaluate kNN with different k values.
"""
function train_knn(X_train::Matrix{Float64}, y_train::AbstractVector{Bool}, 
                  X_test::Matrix{Float64}, y_test::AbstractVector{Bool}, cv_indices::Vector{Int})
    println("\n" * "="^70)
    println("TRAINING k-NEAREST NEIGHBORS (kNN)")
    println("="^70)
    
    # Define k values to test (6+ values)
    k_values = [1, 3, 5, 7, 9, 11]
    
    results = []
    dataset = (X_train, y_train)
    
    for k in k_values
        println("\nk = $k")
        
        params = Dict("K" => k)
        cv_results = modelCrossValidation(:KNeighborsClassifier, params, dataset, cv_indices)
        
        acc_mean, acc_std = cv_results[1]
        println("  CV Accuracy: $(round(acc_mean*100, digits=2))% ± $(round(acc_std*100, digits=2))%")
        
        push!(results, (k=k, cv_accuracy=acc_mean, cv_std=acc_std))
    end
    
    # Select best k
    best_idx = argmax([r.cv_accuracy for r in results])
    best_k = results[best_idx].k
    
    println("\n" * "-"^70)
    println("Best k: $best_k")
    println("  CV Accuracy: $(round(results[best_idx].cv_accuracy*100, digits=2))%")
    
    # Train final model on full training set using helper function
    println("\nTraining final kNN model on full training set...")
    params = Dict("K" => best_k)
    mach, test_accuracy, confusion = train_and_evaluate_final_model(
        :KNeighborsClassifier, params, X_train, y_train, X_test, y_test
    )
    
    println("Test Set Evaluation:")
    println("  Accuracy: $(round(test_accuracy*100, digits=2))%")
    println("  Confusion Matrix:")
    y_test_str = string.(y_test)
    classes = unique(y_test_str)
    printConfusionMatrix(string.(mode.(MLJ.predict(mach, MLJ.table(X_test)))), y_test_str, classes)
    
    return results, best_k, mach, test_accuracy, confusion
end

"""
Create and evaluate ensemble model combining the best 3 models.
"""
function train_ensemble(X_train::Matrix{Float64}, y_train::AbstractVector{Bool}, 
                       X_test::Matrix{Float64}, y_test::AbstractVector{Bool},
                       svm_mach, dt_mach, knn_mach)
    println("\n" * "="^70)
    println("TRAINING ENSEMBLE MODEL (Majority Voting)")
    println("="^70)
    
    # Get predictions from each model on test set
    y_train_str = string.(y_train)
    y_test_str = string.(y_test)
    
    # SVM predictions
    svm_pred = string.(MLJ.predict(svm_mach, MLJ.table(X_test)))
    
    # Decision Tree predictions
    dt_pred = string.(mode.(MLJ.predict(dt_mach, MLJ.table(X_test))))
    
    # kNN predictions
    knn_pred = string.(mode.(MLJ.predict(knn_mach, MLJ.table(X_test))))
    
    println("Combining predictions from:")
    println("  - SVM (poly degree 3)")
    println("  - Decision Tree (best performer)")
    println("  - kNN")
    
    # Majority voting
    ensemble_predictions = similar(svm_pred)
    for i in 1:length(svm_pred)
        # Count votes for each class
        votes = [svm_pred[i], dt_pred[i], knn_pred[i]]
        
        # Simple majority vote (most common prediction)
        vote_counts = Dict{String, Int}()
        for vote in votes
            vote_counts[vote] = get(vote_counts, vote, 0) + 1
        end
        
        # Select class with most votes
        ensemble_predictions[i] = argmax(vote_counts)
    end
    
    # Calculate metrics
    classes = unique(y_train_str)
    ensemble_metrics = confusionMatrix(ensemble_predictions, y_test_str, classes)
    ensemble_accuracy = ensemble_metrics[1]
    confusion = ensemble_metrics[8]
    
    println("\nEnsemble Test Set Evaluation:")
    println("  Accuracy: $(round(ensemble_accuracy*100, digits=2))%")
    println("  Confusion Matrix:")
    printConfusionMatrix(ensemble_predictions, y_test_str, classes)
    
    return ensemble_predictions, ensemble_accuracy, confusion
end

# ============================================================================
# MAIN APPROACH FUNCTIONS
# ============================================================================

"""
Execute Approach 1: Baseline with all features.
"""
function run_approach_1()
    println("\n" * "■"^70)
    println("■  APPROACH 1: BASELINE (ALL FEATURES)")
    println("■"^70)
    
    # Load data
    df = load_data(DATASET_PATH)
    
    # Prepare data
    X, y, feature_names = prepare_data(df, 1)
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE)
    
    # Normalize
    X_train_norm, X_test_norm = apply_normalization(X_train, X_test, 1)
    
    # Create CV indices
    cv_indices = create_cv_indices(size(X_train_norm, 1), CV_FOLDS)
    
    # Train models
    ann_results, best_ann, ann_model, ann_test_acc, ann_confusion = train_anns(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    svm_results, best_svm, svm_mach, svm_test_acc, svm_confusion = train_svms(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    dt_results, best_dt, dt_mach, dt_test_acc, dt_confusion = train_decision_trees(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    knn_results, best_knn, knn_mach, knn_test_acc, knn_confusion = train_knn(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    
    # Train ensemble
    ensemble_preds, ensemble_acc, ensemble_confusion = train_ensemble(X_train_norm, y_train, X_test_norm, y_test,
                                                                       svm_mach, dt_mach, knn_mach)
    
    # Print summary
    println("\n" * "="^70)
    println("APPROACH 1 - FINAL TEST SET RESULTS SUMMARY")
    println("="^70)
    println("Model                          Test Accuracy")
    println("-"^70)
    println("ANNs ([50, 25])                $(round(ann_test_acc*100, digits=2))%")
    println("SVM (poly degree 3)            $(round(svm_test_acc*100, digits=2))%")
    println("Decision Tree (depth 7)        $(round(dt_test_acc*100, digits=2))%")
    println("kNN (k=9)                      $(round(knn_test_acc*100, digits=2))%")
    println("Ensemble (Majority Voting)     $(round(ensemble_acc*100, digits=2))%")
    println("="^70)
    
    println("\n" * "■"^70)
    println("■  APPROACH 1 COMPLETE")
    println("■"^70)
end

"""
Execute Approach 2: Feature selection.
"""
function run_approach_2()
    println("\n" * "■"^70)
    println("■  APPROACH 2: FEATURE SELECTION")
    println("■"^70)
    
    # Load data
    df = load_data(DATASET_PATH)
    
    # Prepare data (with feature selection)
    X, y, feature_names = prepare_data(df, 2)
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE)
    
    # Normalize
    X_train_norm, X_test_norm = apply_normalization(X_train, X_test, 2)
    
    # Create CV indices
    cv_indices = create_cv_indices(size(X_train_norm, 1), CV_FOLDS)
    
    # Train models
    ann_results, best_ann, ann_model, ann_test_acc, ann_confusion = train_anns(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    svm_results, best_svm, svm_mach, svm_test_acc, svm_confusion = train_svms(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    dt_results, best_dt, dt_mach, dt_test_acc, dt_confusion = train_decision_trees(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    knn_results, best_knn, knn_mach, knn_test_acc, knn_confusion = train_knn(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    
    # Train ensemble
    ensemble_preds, ensemble_acc, ensemble_confusion = train_ensemble(X_train_norm, y_train, X_test_norm, y_test,
                                                                       svm_mach, dt_mach, knn_mach)
    
    # Print summary
    println("\n" * "="^70)
    println("APPROACH 2 - FINAL TEST SET RESULTS SUMMARY")
    println("="^70)
    println("Model                          Test Accuracy")
    println("-"^70)
    println("ANNs                           $(round(ann_test_acc*100, digits=2))%")
    println("SVM                            $(round(svm_test_acc*100, digits=2))%")
    println("Decision Tree                  $(round(dt_test_acc*100, digits=2))%")
    println("kNN                            $(round(knn_test_acc*100, digits=2))%")
    println("Ensemble (Majority Voting)     $(round(ensemble_acc*100, digits=2))%")
    println("="^70)
    
    println("\n" * "■"^70)
    println("■  APPROACH 2 COMPLETE")
    println("■"^70)
end

"""
Execute Approach 3: PCA dimensionality reduction.
"""
function run_approach_3()
    println("\n" * "■"^70)
    println("■  APPROACH 3: PCA DIMENSIONALITY REDUCTION")
    println("■"^70)
    
    # Load data
    df = load_data(DATASET_PATH)
    
    # Prepare data (all features initially)
    X, y, feature_names = prepare_data(df, 3)
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE)
    
    # Normalize (must use ZeroMean for PCA)
    X_train_norm, X_test_norm = apply_normalization(X_train, X_test, 3)
    
    # Apply PCA
    n_components = 20  # Can try 10 and 20
    X_train_pca, X_test_pca, pca_model = apply_pca(X_train_norm, X_test_norm, n_components)
    
    # Create CV indices
    cv_indices = create_cv_indices(size(X_train_pca, 1), CV_FOLDS)
    
    # Train models
    ann_results, best_ann, ann_model, ann_test_acc, ann_confusion = train_anns(X_train_pca, y_train, X_test_pca, y_test, cv_indices)
    svm_results, best_svm, svm_mach, svm_test_acc, svm_confusion = train_svms(X_train_pca, y_train, X_test_pca, y_test, cv_indices)
    dt_results, best_dt, dt_mach, dt_test_acc, dt_confusion = train_decision_trees(X_train_pca, y_train, X_test_pca, y_test, cv_indices)
    knn_results, best_knn, knn_mach, knn_test_acc, knn_confusion = train_knn(X_train_pca, y_train, X_test_pca, y_test, cv_indices)
    
    # Train ensemble
    ensemble_preds, ensemble_acc, ensemble_confusion = train_ensemble(X_train_pca, y_train, X_test_pca, y_test,
                                                                       svm_mach, dt_mach, knn_mach)
    
    # Print summary
    println("\n" * "="^70)
    println("APPROACH 3 - FINAL TEST SET RESULTS SUMMARY")
    println("="^70)
    println("Model                          Test Accuracy")
    println("-"^70)
    println("ANNs                           $(round(ann_test_acc*100, digits=2))%")
    println("SVM                            $(round(svm_test_acc*100, digits=2))%")
    println("Decision Tree                  $(round(dt_test_acc*100, digits=2))%")
    println("kNN                            $(round(knn_test_acc*100, digits=2))%")
    println("Ensemble (Majority Voting)     $(round(ensemble_acc*100, digits=2))%")
    println("="^70)
    
    println("\n" * "■"^70)
    println("■  APPROACH 3 COMPLETE")
    println("■"^70)
end

"""
Execute Approach 4: Imbalanced classification.
"""
function run_approach_4()
    println("\n" * "■"^70)
    println("■  APPROACH 4: IMBALANCED CLASSIFICATION")
    println("■"^70)
    
    # Load data
    df = load_data(DATASET_PATH)
    
    # Prepare data (75th percentile threshold)
    X, y, feature_names = prepare_data(df, 4)
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE)
    
    # Normalize
    X_train_norm, X_test_norm = apply_normalization(X_train, X_test, 4)
    
    # Create CV indices
    cv_indices = create_cv_indices(size(X_train_norm, 1), CV_FOLDS)
    
    # Train models
    ann_results, best_ann, ann_model, ann_test_acc, ann_confusion = train_anns(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    svm_results, best_svm, svm_mach, svm_test_acc, svm_confusion = train_svms(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    dt_results, best_dt, dt_mach, dt_test_acc, dt_confusion = train_decision_trees(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    knn_results, best_knn, knn_mach, knn_test_acc, knn_confusion = train_knn(X_train_norm, y_train, X_test_norm, y_test, cv_indices)
    
    # Train ensemble
    ensemble_preds, ensemble_acc, ensemble_confusion = train_ensemble(X_train_norm, y_train, X_test_norm, y_test,
                                                                       svm_mach, dt_mach, knn_mach)
    
    # Print summary
    println("\n" * "="^70)
    println("APPROACH 4 - FINAL TEST SET RESULTS SUMMARY")
    println("="^70)
    println("Model                          Test Accuracy")
    println("-"^70)
    println("ANNs                           $(round(ann_test_acc*100, digits=2))%")
    println("SVM                            $(round(svm_test_acc*100, digits=2))%")
    println("Decision Tree                  $(round(dt_test_acc*100, digits=2))%")
    println("kNN                            $(round(knn_test_acc*100, digits=2))%")
    println("Ensemble (Majority Voting)     $(round(ensemble_acc*100, digits=2))%")
    println("="^70)
    
    println("\n" * "■"^70)
    println("■  APPROACH 4 COMPLETE")
    println("■"^70)
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    # Set random seed for reproducibility
    Random.seed!(RANDOM_SEED)
    
    println("="^70)
    println("ONLINE NEWS POPULARITY PREDICTION")
    println("ML Final Project - Four Approaches")
    println("="^70)
    println("Random seed: $RANDOM_SEED")
    println("Test size: $(Int(TEST_SIZE*100))%")
    println("Cross-validation folds: $CV_FOLDS")
    
    # Parse command line argument
    if length(ARGS) < 1
        println("\nERROR: No approach specified!")
        println("\nUsage: julia main.jl <approach_number>")
        println("  approach_number: 1, 2, 3, or 4")
        println("\nApproaches:")
        println("  1 - Baseline (all features, MinMax, median split)")
        println("  2 - Feature Selection (15-20 features, MinMax, median split)")
        println("  3 - PCA (10-20 components, ZeroMean, median split)")
        println("  4 - Imbalanced (all features, ZeroMean, 75th percentile split)")
        return
    end
    
    approach = parse(Int, ARGS[1])
    
    if !(approach in [1, 2, 3, 4])
        println("\nERROR: Invalid approach number: $approach")
        println("Must be 1, 2, 3, or 4")
        return
    end

    # Execute the specified approach
    if approach == 1
        run_approach_1()
    elseif approach == 2
        run_approach_2()
    elseif approach == 3
        run_approach_3()
    elseif approach == 4
        run_approach_4()
    end
    
    println("\n" * "="^70)
    println("EXECUTION COMPLETE")
    println("="^70)
end

# Run main function
main()

