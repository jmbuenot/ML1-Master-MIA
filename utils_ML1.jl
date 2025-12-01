module UtilsML1

using Flux
using Flux.Losses
using Statistics
using Random
using MLJ
using LIBSVM
using DecisionTree
using MLJLIBSVMInterface
using MLJModelInterface
using MLJDecisionTreeInterface
using NearestNeighborModels
using CategoricalArrays

const NaN32 = Float32(NaN)
const Inf32 = Float32(Inf)

export oneHotEncoding,
       calculateMinMaxNormalizationParameters,
       calculateZeroMeanNormalizationParameters,
       normalizeMinMax!,
       normalizeMinMax,
       normalizeZeroMean!,
       normalizeZeroMean,
       classifyOutputs,
       confusionMatrix,
       printConfusionMatrix,
       accuracy,
       buildClassANN,
       trainClassANN,
       add,
       mse,
       avgGreaterThan0,
       rango_columnas,
       loss_regression,
       loss_binary,
       loss_multiclass,
       loss,
       holdOut,
       ANNCrossValidation,
       modelCrossValidation,
       VotingClassifier

"""Encode a categorical vector as one-hot columns or a boolean column."""
function oneHotEncoding(feature::AbstractArray{<:Any,1},
                        classes::AbstractArray{<:Any,1})
    @assert all(in(value, classes) for value in feature) "Found values outside the provided class set"

    numClasses = length(classes)
    @assert numClasses > 1 "At least two classes are required for encoding"

    if numClasses == 2
        return reshape(feature .== classes[1], :, 1)
    else
        oneHot = BitArray{2}(undef, length(feature), numClasses)
        for (idx, cls) in enumerate(classes)
            oneHot[:, idx] .= feature .== cls
        end
        return oneHot
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

"""Compute per-column minima and maxima."""
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims = 1), maximum(dataset, dims = 1)
end

"""Compute per-column means and standard deviations."""
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims = 1), std(dataset, dims = 1)
end

"""Normalize in-place with min-max using precomputed parameters."""
function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
                          normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    dataset[:, vec(minValues .== maxValues)] .= 0
    return dataset
end

"""Normalize in-place with min-max computing parameters on the fly."""
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

"""Return a min-max normalised copy using supplied parameters."""
function normalizeMinMax(dataset::AbstractArray{<:Real,2},
                         normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeMinMax!(copy(dataset), normalizationParameters)
end

"""Return a min-max normalised copy computing the parameters."""
function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
end

"""Standardise in-place using precomputed parameters."""
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters
    dataset .-= avgValues
    dataset ./= stdValues
    dataset[:, vec(stdValues .== 0)] .= 0
    return dataset
end

"""Standardise in-place computing means and standard deviations."""
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end

"""Return a standardised copy using supplied parameters."""
function normalizeZeroMean(dataset::AbstractArray{<:Real,2},
                           normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeZeroMean!(copy(dataset), normalizationParameters)
end

"""Return a standardised copy computing means and standard deviations."""
function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))
end

"""Classify continuous outputs into boolean labels."""
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real = 0.5)
    numOutputs = size(outputs, 2)
    @assert numOutputs >= 1 "Output matrix must contain at least one column"

    if numOutputs == 1
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims = 2)
        classified = falses(size(outputs))
        classified[indicesMaxEachInstance] .= true
        @assert all(sum(classified, dims = 2) .== 1) "Each pattern must be assigned to exactly one class"
        return classified
    end
end

"""Compute accuracy between boolean predictions and targets."""
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert all(size(outputs) .== size(targets)) "Output and target matrices must share the same dimensions"

    if size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        return mean(all(targets .== outputs, dims = 2))
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real = 0.5)
    return accuracy(outputs .>= threshold, targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real = 0.5)
    @assert all(size(outputs) .== size(targets)) "Output and target matrices must share the same dimensions"

    if size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1]; threshold = threshold)
    else
        return accuracy(classifyOutputs(outputs; threshold = threshold), targets)
    end
end

"""Compute binary metrics and confusion matrix without loops."""
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length"

    total = length(outputs)
    @assert total > 0 "At least one pattern is required to build a confusion matrix"

    tn = sum((.!outputs) .& (.!targets))
    tp = sum(outputs .& targets)
    fp = sum(outputs .& (.!targets))
    fn = sum((.!outputs) .& targets)

    confusion = Matrix{Int64}([tn fp; fn tp])

    accuracy_val = (tn + tp) / total
    error_rate = (fp + fn) / total

    all_true_negative = tn == total
    all_true_positive = tp == total

    sensitivity = all_true_negative ? 1.0 : ((tp + fn) == 0 ? 0.0 : tp / (tp + fn))
    positive_predictive_value = all_true_negative ? 1.0 : ((tp + fp) == 0 ? 0.0 : tp / (tp + fp))

    specificity = all_true_positive ? 1.0 : ((tn + fp) == 0 ? 0.0 : tn / (tn + fp))
    negative_predictive_value = all_true_positive ? 1.0 : ((tn + fn) == 0 ? 0.0 : tn / (tn + fn))

    fscore = (sensitivity + positive_predictive_value) == 0 ? 0.0 :
             2 * sensitivity * positive_predictive_value / (sensitivity + positive_predictive_value)

    return (accuracy_val,
            error_rate,
            sensitivity,
            specificity,
            positive_predictive_value,
            negative_predictive_value,
            fscore,
            confusion)
end

"""Apply a threshold to real-valued outputs and delegate to the binary `confusionMatrix`."""
function confusionMatrix(outputs::AbstractArray{<:Real,1},
                         targets::AbstractArray{Bool,1};
                         threshold::Real = 0.5)
    return confusionMatrix(outputs .>= threshold, targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2},
                         targets::AbstractArray{Bool,2};
                         weighted::Bool = true)
    @assert size(outputs) == size(targets) "'outputs' and 'targets' must share the same shape"

    num_classes = size(outputs, 2)
    if num_classes == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    end

    @assert num_classes != 2 "Use the binary version of `confusionMatrix` for two classes"

    sensitivities = zeros(Float64, num_classes)
    specificities = zeros(Float64, num_classes)
    ppvs = zeros(Float64, num_classes)
    npvs = zeros(Float64, num_classes)
    f1_scores = zeros(Float64, num_classes)

    class_counts = vec(sum(targets, dims = 1))

    for class_idx in 1:num_classes
        if class_counts[class_idx] > 0
            _, _, sens, spec, ppv, npv, f1, _ = confusionMatrix(outputs[:, class_idx],
                                                                targets[:, class_idx])
            sensitivities[class_idx] = sens
            specificities[class_idx] = spec
            ppvs[class_idx] = ppv
            npvs[class_idx] = npv
            f1_scores[class_idx] = f1
        end
    end

    confusion = zeros(Int64, num_classes, num_classes)
    for actual_idx in 1:num_classes
        for predicted_idx in 1:num_classes
            confusion[actual_idx, predicted_idx] = sum(targets[:, actual_idx] .& outputs[:, predicted_idx])
        end
    end

    valid_mask = class_counts .> 0
    valid_counts = class_counts[valid_mask]

    aggregate(metric_vector) = if !any(valid_mask)
        0.0
    elseif weighted
        total = sum(valid_counts)
        total == 0 ? 0.0 : sum(metric_vector[valid_mask] .* valid_counts) / total
    else
        mean(metric_vector[valid_mask])
    end

    agg_sensitivity = aggregate(sensitivities)
    agg_specificity = aggregate(specificities)
    agg_ppv = aggregate(ppvs)
    agg_npv = aggregate(npvs)
    agg_f1 = aggregate(f1_scores)

    overall_accuracy = accuracy(outputs, targets)
    error_rate = 1 - overall_accuracy

    return (overall_accuracy, error_rate, agg_sensitivity, agg_specificity,
            agg_ppv, agg_npv, agg_f1, confusion)
end

"""Convert continuous outputs into labels and delegate to the boolean version."""
function confusionMatrix(outputs::AbstractArray{<:Real,2},
                         targets::AbstractArray{Bool,2};
                         threshold::Real = 0.5,
                         weighted::Bool = true)
    classified = classifyOutputs(outputs; threshold = threshold)
    return confusionMatrix(classified, targets; weighted = weighted)
end

"""Encode arbitrary labels and delegate to the boolean version."""
function confusionMatrix(outputs::AbstractArray{<:Any,1},
                         targets::AbstractArray{<:Any,1},
                         classes::AbstractArray{<:Any,1};
                         weighted::Bool = true)
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length"

    classes_vec = collect(classes)
    @assert !isempty(classes_vec) "At least one class must be declared"
    @assert length(unique(classes_vec)) == length(classes_vec) "`classes` must not contain duplicated values"
    @assert all(in(value, classes_vec) for value in outputs) "`outputs` contains labels outside `classes`"
    @assert all(in(value, classes_vec) for value in targets) "`targets` contains labels outside `classes`"

    if length(classes_vec) == 1
        cls = classes_vec[1]
        encoded_outputs = reshape(outputs .== cls, :, 1)
        encoded_targets = reshape(targets .== cls, :, 1)
    else
        encoded_outputs = oneHotEncoding(outputs, classes_vec)
        encoded_targets = oneHotEncoding(targets, classes_vec)
    end

    return confusionMatrix(encoded_outputs, encoded_targets; weighted = weighted)
end

"""Generate multiclass metrics by inferring the class set."""
function confusionMatrix(outputs::AbstractArray{<:Any,1},
                         targets::AbstractArray{<:Any,1};
                         weighted::Bool = true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes; weighted = weighted)
end

"""Print metrics and confusion matrix for boolean outputs."""
function _print_confusion_summary(metrics; aggregation::Union{Nothing,String} = nothing)
    accuracy_val,
    error_rate,
    sensitivity,
    specificity,
    positive_predictive_value,
    negative_predictive_value,
    fscore,
    confusion = metrics

    println("Accuracy: $accuracy_val")
    println("Error rate: $error_rate")
    println("Sensitivity (Recall): $sensitivity")
    println("Specificity: $specificity")
    println("Positive predictive value (Precision): $positive_predictive_value")
    println("Negative predictive value: $negative_predictive_value")
    println("F-score: $fscore")
    if aggregation !== nothing
    println("Aggregation strategy: $aggregation")
    end
    println("Confusion matrix:")
    println(confusion)
    return nothing
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    metrics = confusionMatrix(outputs, targets)
    _print_confusion_summary(metrics)
    return nothing
end

"""Print metrics for real outputs after applying a threshold."""
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
                              targets::AbstractArray{Bool,1};
                              threshold::Real = 0.5)
    return printConfusionMatrix(outputs .>= threshold, targets)
end

"""Print multiclass metrics for boolean matrices."""
function printConfusionMatrix(outputs::AbstractArray{Bool,2},
                              targets::AbstractArray{Bool,2};
                              weighted::Bool = true)
    metrics = confusionMatrix(outputs, targets; weighted = weighted)
    aggregation = weighted ? "weighted" : "macro"
    _print_confusion_summary(metrics; aggregation = aggregation)
    return nothing
end

"""Print multiclass metrics for real-valued outputs."""
function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
                              targets::AbstractArray{Bool,2};
                              weighted::Bool = true)
    metrics = confusionMatrix(outputs, targets; weighted = weighted)
    aggregation = weighted ? "weighted" : "macro"
    _print_confusion_summary(metrics; aggregation = aggregation)
    return nothing
end

"""Print multiclass metrics from labels and an explicit catalogue."""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
                              targets::AbstractArray{<:Any,1},
                              classes::AbstractArray{<:Any,1};
                              weighted::Bool = true)
    metrics = confusionMatrix(outputs, targets, classes; weighted = weighted)
    aggregation = weighted ? "weighted" : "macro"
    _print_confusion_summary(metrics; aggregation = aggregation)
    return nothing
end

"""Print multiclass metrics inferring the class catalogue."""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
                              targets::AbstractArray{<:Any,1};
                              weighted::Bool = true)
    metrics = confusionMatrix(outputs, targets; weighted = weighted)
    aggregation = weighted ? "weighted" : "macro"
    _print_confusion_summary(metrics; aggregation = aggregation)
    return nothing
end

"""Build a classification ANN with an arbitrary topology."""
function buildClassANN(numInputs::Int,
                       topology::AbstractArray{<:Int,1},
                       numOutputs::Int;
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)))
    @assert length(transferFunctions) == length(topology) "Provide one activation function per hidden layer"

    ann = Chain()
    numInputsLayer = numInputs
    for (idx, numNeurons) in enumerate(topology)
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[idx]))
        numInputsLayer = numNeurons
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end

    return ann
end

"""Train a classification ANN with patterns stored in rows."""
function trainClassANN(topology::AbstractArray{<:Int,1},
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                       validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} =
                           (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
                       testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} =
                           (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
                       maxEpochs::Int = 1000,
                       minLoss::Real = 0.0,
                       learningRate::Real = 0.01,
                       maxEpochsVal::Int = 20,
                       showText::Bool = false)
    inputs, targets = trainingDataset
    @assert size(inputs, 1) == size(targets, 1) "The number of patterns must match"

    ann = buildClassANN(size(inputs, 2), topology, size(targets, 2);
                        transferFunctions = transferFunctions)

    function loss_fn(model, x, y)
        if size(y, 1) == 1
            return Losses.binarycrossentropy(model(x), y)
        else
            return Losses.crossentropy(model(x), y)
        end
    end

    inputs_t = permutedims(Float32.(inputs))
    targets_t = permutedims(Float32.(targets))
    training_batch = [(inputs_t, targets_t)]

    val_inputs, val_targets = validationDataset
    has_validation = !isempty(val_inputs) && !isempty(val_targets)
    val_inputs_t = has_validation ? permutedims(Float32.(val_inputs)) : zeros(Float32, size(inputs_t, 1), 0)
    val_targets_t = has_validation ? permutedims(Float32.(val_targets)) : zeros(Float32, size(targets_t, 1), 0)

    test_inputs, test_targets = testDataset
    has_test = !isempty(test_inputs) && !isempty(test_targets)
    test_inputs_t = has_test ? permutedims(Float32.(test_inputs)) : zeros(Float32, size(inputs_t, 1), 0)
    test_targets_t = has_test ? permutedims(Float32.(test_targets)) : zeros(Float32, size(targets_t, 1), 0)

    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]

    trainingLoss = Float32(loss_fn(ann, inputs_t, targets_t))
    push!(trainingLosses, trainingLoss)

    valLoss = has_validation ? Float32(loss_fn(ann, val_inputs_t, val_targets_t)) : NaN32
    push!(validationLosses, valLoss)

    testLoss = has_test ? Float32(loss_fn(ann, test_inputs_t, test_targets_t)) : NaN32
    push!(testLosses, testLoss)

    showText && println("Epoch 0: loss train=$(trainingLoss)" *
                        (has_validation ? ", val=$(valLoss)" : "") *
                        (has_test ? ", test=$(testLoss)" : ""))

    opt_state = Flux.setup(Adam(learningRate), ann)

    bestValLoss = has_validation ? valLoss : Inf32
    bestAnn = has_validation ? deepcopy(ann) : nothing
    bestEpoch = 0
    epochsWithoutImprovement = 0

    epoch = 0
    while (epoch < maxEpochs) && (trainingLoss > minLoss)
        Flux.train!(loss_fn, ann, training_batch, opt_state)
        epoch += 1

        trainingLoss = Float32(loss_fn(ann, inputs_t, targets_t))
        push!(trainingLosses, trainingLoss)

        if has_validation
            valLoss = Float32(loss_fn(ann, val_inputs_t, val_targets_t))
            push!(validationLosses, valLoss)

            if valLoss < bestValLoss
                bestValLoss = valLoss
                bestAnn = deepcopy(ann)
                bestEpoch = epoch
                epochsWithoutImprovement = 0
            else
                epochsWithoutImprovement += 1
            end
        else
            push!(validationLosses, NaN32)
        end

        if has_test
            testLoss = Float32(loss_fn(ann, test_inputs_t, test_targets_t))
            push!(testLosses, testLoss)
        else
            push!(testLosses, NaN32)
        end

        showText && println("Epoch $epoch: loss train=$(trainingLoss)" *
                             (has_validation ? ", val=$(validationLosses[end])" : "") *
                             (has_test ? ", test=$(testLosses[end])" : ""))

        if has_validation && epochsWithoutImprovement >= maxEpochsVal
            showText && println("Early stopping after $epochsWithoutImprovement epochs without improvement (best epoch: $bestEpoch)")
            break
        end
    end

    finalAnn = (has_validation && bestAnn !== nothing) ? bestAnn : ann
    return finalAnn, trainingLosses, validationLosses, testLosses
end

function trainClassANN(topology::AbstractArray{<:Int,1},
                       dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
                       validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} =
                           (Array{eltype(dataset[1]),2}(undef, 0, 0), falses(0)),
                       testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} =
                           (Array{eltype(dataset[1]),2}(undef, 0, 0), falses(0)),
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
                       maxEpochs::Int = 1000,
                       minLoss::Real = 0.0,
                       learningRate::Real = 0.01,
                       maxEpochsVal::Int = 20,
                       showText::Bool = false)
    inputs, targets_vec = dataset
    targets_matrix = reshape(targets_vec, length(targets_vec), 1)

    val_inputs, val_targets_vec = validationDataset
    val_targets_matrix = reshape(val_targets_vec, length(val_targets_vec), 1)
    val_dataset_matrix = (val_inputs, val_targets_matrix)

    test_inputs, test_targets_vec = testDataset
    test_targets_matrix = reshape(test_targets_vec, length(test_targets_vec), 1)
    test_dataset_matrix = (test_inputs, test_targets_matrix)

    return trainClassANN(topology, (inputs, targets_matrix);
                          validationDataset = val_dataset_matrix,
                          testDataset = test_dataset_matrix,
                          transferFunctions = transferFunctions,
                          maxEpochs = maxEpochs,
                          minLoss = minLoss,
                          learningRate = learningRate,
                          maxEpochsVal = maxEpochsVal,
                          showText = showText)
end

"""Specialised sum for Float32."""
add(x::Float32, y::Float32) = x + y

"""Generic sum for real numbers."""
add(x::Real, y::Real) = x + y

"""Specialised mean squared error."""
mse(outputs::Array{Float32,1}, targets::Array{Float32,1}) = mean((targets .- outputs) .^ 2)

"""Generic mean squared error."""
mse(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{<:Real,1}) = mean((targets .- outputs) .^ 2)

"""Average of positive values with the option to return the mask."""
function avgGreaterThan0(valores::AbstractArray{<:Real,1}; return_mask::Bool = false)
    positivos = valores .> 0
    promedio = mean(valores[positivos])
    return return_mask ? (positivos, promedio) : promedio
end

"""Column-wise range (max - min)."""
function rango_columnas(matriz::AbstractArray{<:Real,2})
    min_col = minimum(matriz, dims = 1)
    max_col = maximum(matriz, dims = 1)
    return max_col .- min_col
end

"""Loss function for regression."""
loss_regression(m, x, y) = Losses.mse(m(x), y)

"""Loss function for binary classification."""
loss_binary(m, x, y) = Losses.binarycrossentropy(m(x), y)

"""Loss function for multiclass classification."""
loss_multiclass(m, x, y) = Losses.crossentropy(m(x), y)

"""Automatically select the loss according to the number of outputs."""
loss(m, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(m(x), y) : Losses.crossentropy(m(x), y)

"""Split indices into train and test following the hold-out strategy."""
function holdOut(N::Int, P::Real)
    @assert N >= 0 "The number of patterns cannot be negative"
    @assert 0 <= P <= 1 "The proportion must lie within the interval [0, 1]"

    num_test = round(Int, P * N)
    ordering = randperm(N)

    test_indices = num_test == 0 ? Int[] : collect(ordering[1:num_test])
    train_indices = num_test == N ? Int[] : collect(ordering[(num_test + 1):end])

    return train_indices, test_indices
end

"""Split indices into train, validation, and test sets."""
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert N >= 0 "The number of patterns cannot be negative"
    @assert 0 <= Pval <= 1 "The validation proportion must lie within the interval [0, 1]"
    @assert 0 <= Ptest <= 1 "The test proportion must lie within the interval [0, 1]"
    @assert Pval + Ptest <= 1 "The sum of validation and test proportions cannot exceed 1"

    train_val_indices, test_indices = holdOut(N, Ptest)

    remaining = length(train_val_indices)
    num_val = round(Int, Pval * N)
    num_val = min(num_val, remaining)

    if remaining == 0
        @assert num_val == 0 "No patterns remain for validation"
        return Int[], Int[], test_indices
    elseif num_val == 0
        return train_val_indices, Int[], test_indices
    elseif num_val == remaining
        return Int[], train_val_indices, test_indices
    else
        proportion_val = num_val / remaining
        train_subset_rel, val_subset_rel = holdOut(remaining, proportion_val)
        train_indices = train_val_indices[train_subset_rel]
        val_indices = train_val_indices[val_subset_rel]
        return train_indices, val_indices, test_indices
    end
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
                            dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                            crossValidationIndices::Array{Int64,1};
                            numExecutions::Int = 50,
                            transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
                            maxEpochs::Int = 1000,
                            minLoss::Real = 0.0,
                            learningRate::Real = 0.01,
                            validationRatio::Real = 0,
                            maxEpochsVal::Int = 20)

    @assert numExecutions > 0 "At least one execution per fold is required"

    inputs, raw_targets = dataset
    num_instances = size(inputs, 1)
    @assert length(raw_targets) == num_instances "Targets must match the number of instances"
    @assert length(crossValidationIndices) == num_instances "Fold assignments must match the dataset size"

    classes = unique(raw_targets)
    encoded_targets = oneHotEncoding(raw_targets, classes)
    num_output_columns = size(encoded_targets, 2)
    confusion_dim = num_output_columns == 1 ? 2 : num_output_columns

    numFolds = maximum(crossValidationIndices)
    @assert numFolds > 0 "At least one fold is required"

    feature_count = size(inputs, 2)

    accuracy_folds = Float64[]
    error_folds = Float64[]
    sensitivity_folds = Float64[]
    specificity_folds = Float64[]
    ppv_folds = Float64[]
    npv_folds = Float64[]
    f1_folds = Float64[]

    global_confusion = zeros(Float64, confusion_dim, confusion_dim)

    for fold in 1:numFolds
        test_mask = crossValidationIndices .== fold
        train_mask = .!test_mask

        @assert any(test_mask) "Fold $fold does not contain any test instances"
        @assert any(train_mask) "Fold $fold leaves no instances for training"

        train_indices = collect(findall(train_mask))
        test_indices = collect(findall(test_mask))

        base_train_inputs = inputs[train_indices, :]
        base_train_targets = encoded_targets[train_indices, :]
        test_inputs = inputs[test_indices, :]
        test_targets = encoded_targets[test_indices, :]

        accuracy_exec = Float64[]
        error_exec = Float64[]
        sensitivity_exec = Float64[]
        specificity_exec = Float64[]
        ppv_exec = Float64[]
        npv_exec = Float64[]
        f1_exec = Float64[]

        confusion_stack = zeros(Float64, confusion_dim, confusion_dim, numExecutions)

        for execution in 1:numExecutions
            train_inputs_exec = base_train_inputs
            train_targets_exec = base_train_targets
            val_inputs_exec = zeros(eltype(inputs), 0, feature_count)
            val_targets_exec = falses(0, num_output_columns)

            if validationRatio > 0 && length(train_indices) > 1
                desired_val = round(Int, validationRatio * num_instances)
                max_allowed = length(train_indices) - 1
                effective_val = clamp(desired_val, 0, max_allowed)

                if effective_val > 0
                    ratio_subset = clamp(effective_val / length(train_indices), 0.0, 1.0)
                    train_rel, val_rel = holdOut(length(train_indices), ratio_subset)

                    if !isempty(val_rel)
                        train_inputs_exec = inputs[train_indices[train_rel], :]
                        train_targets_exec = encoded_targets[train_indices[train_rel], :]
                        val_inputs_exec = inputs[train_indices[val_rel], :]
                        val_targets_exec = encoded_targets[train_indices[val_rel], :]
                    end
                end
            end

            ann, _, _, _ = trainClassANN(
                topology,
                (train_inputs_exec, train_targets_exec);
                validationDataset = (val_inputs_exec, val_targets_exec),
                transferFunctions = transferFunctions,
                maxEpochs = maxEpochs,
                minLoss = minLoss,
                learningRate = learningRate,
                maxEpochsVal = maxEpochsVal,
                showText = false
            )

            raw_outputs = ann(permutedims(Float32.(test_inputs)))
            test_outputs = permutedims(Array(raw_outputs))

            metrics = confusionMatrix(test_outputs, test_targets; weighted = true)
            acc, err, sens, spec, ppv, npv, f1, confusion = metrics

            push!(accuracy_exec, Float64(acc))
            push!(error_exec, Float64(err))
            push!(sensitivity_exec, Float64(sens))
            push!(specificity_exec, Float64(spec))
            push!(ppv_exec, Float64(ppv))
            push!(npv_exec, Float64(npv))
            push!(f1_exec, Float64(f1))

            confusion_stack[:, :, execution] .= Float64.(confusion)
        end

        push!(accuracy_folds, mean(accuracy_exec))
        push!(error_folds, mean(error_exec))
        push!(sensitivity_folds, mean(sensitivity_exec))
        push!(specificity_folds, mean(specificity_exec))
        push!(ppv_folds, mean(ppv_exec))
        push!(npv_folds, mean(npv_exec))
        push!(f1_folds, mean(f1_exec))

        mean_confusion = dropdims(mean(confusion_stack, dims = 3), dims = 3)
        global_confusion .+= mean_confusion
    end

    stats(vector) = (mean(vector), std(vector; corrected = false))

    return (stats(accuracy_folds),
            stats(error_folds),
            stats(sensitivity_folds),
            stats(specificity_folds),
            stats(ppv_folds),
            stats(npv_folds),
            stats(f1_folds),
            global_confusion)
end

"""Perform cross-validation for different model types (ANN, SVM, Decision Tree, kNN)."""
function modelCrossValidation(
        modelType::Symbol, 
        modelHyperparameters::Dict,
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1})
    
    # Load MLJ models if not already loaded
    SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
    kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
    DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    
    # Helper function to get parameter value with flexible key types (String or Symbol)
    function getParam(dict, key_str, key_sym, default)
        if haskey(dict, key_str)
            return dict[key_str]
        elseif haskey(dict, key_sym)
            return dict[key_sym]
        else
            return default
        end
    end
    
    # Normalize model type symbols - accept both variants
    normalizedModelType = modelType
    if modelType == :SVC
        normalizedModelType = :SVMClassifier
    elseif modelType == :kNN
        normalizedModelType = :KNeighborsClassifier
    end
    
    # Handle ANN case - delegate to ANNCrossValidation
    if normalizedModelType == :ANN
        # Extract ANN hyperparameters
        topology = getParam(modelHyperparameters, "topology", :topology, nothing)
        @assert topology !== nothing "Topology is required for ANN"
        
        learningRate = getParam(modelHyperparameters, "learningRate", :learningRate, 0.01)
        validationRatio = getParam(modelHyperparameters, "validationRatio", :validationRatio, 0.0)
        numExecutions = getParam(modelHyperparameters, "numExecutions", :numExecutions, 50)
        maxEpochs = getParam(modelHyperparameters, "maxEpochs", :maxEpochs, 1000)
        maxEpochsVal = getParam(modelHyperparameters, "maxEpochsVal", :maxEpochsVal, 20)
        
        # Get transfer functions if provided
        transferFunctions = getParam(modelHyperparameters, "transferFunctions", :transferFunctions, fill(σ, length(topology)))
        
        # Call ANNCrossValidation
        return ANNCrossValidation(
            topology,
            dataset,
            crossValidationIndices;
            numExecutions = numExecutions,
            transferFunctions = transferFunctions,
            maxEpochs = maxEpochs,
            minLoss = 0.0,
            learningRate = learningRate,
            validationRatio = validationRatio,
            maxEpochsVal = maxEpochsVal
        )
    end
    
    # Handle MLJ models (SVM, Decision Tree, kNN)
    inputs, targets = dataset
    
    # Convert targets to strings to prevent type issues
    targets = string.(targets)
    
    # Get unique classes
    classes = unique(targets)
    
    # Initialize metric vectors for each fold
    accuracy_folds = Float64[]
    error_folds = Float64[]
    sensitivity_folds = Float64[]
    specificity_folds = Float64[]
    ppv_folds = Float64[]
    npv_folds = Float64[]
    f1_folds = Float64[]
    
    # Initialize global confusion matrix
    global_confusion = zeros(Int64, length(classes), length(classes))
    
    # Get number of folds
    numFolds = maximum(crossValidationIndices)
    
    # Cross-validation loop
    for fold in 1:numFolds
        # Split data into train and test for this fold
        test_mask = crossValidationIndices .== fold
        train_mask = .!test_mask
        
        train_inputs = inputs[train_mask, :]
        train_targets = targets[train_mask]
        test_inputs = inputs[test_mask, :]
        test_targets = targets[test_mask]
        
        # Create model based on model type
        model = nothing
        
        if normalizedModelType == :SVMClassifier
            # Extract SVM hyperparameters
            C = getParam(modelHyperparameters, "C", :C, 1.0)
            kernel_str = getParam(modelHyperparameters, "kernel", :kernel, "rbf")
            gamma = getParam(modelHyperparameters, "gamma", :gamma, 0.1)
            degree = getParam(modelHyperparameters, "degree", :degree, 3)
            coef0 = getParam(modelHyperparameters, "coef0", :coef0, 0.0)
            
            # Map kernel string to LIBSVM kernel type
            kernel = if kernel_str == "linear"
                LIBSVM.Kernel.Linear
            elseif kernel_str == "rbf"
                LIBSVM.Kernel.RadialBasis
            elseif kernel_str == "sigmoid"
                LIBSVM.Kernel.Sigmoid
            elseif kernel_str == "poly" || kernel_str == "polynomial"
                LIBSVM.Kernel.Polynomial
            else
                LIBSVM.Kernel.RadialBasis  # default
            end
            
        # Create SVM model with kernel-specific parameters
        model = if kernel_str == "linear"
            # Linear kernel: only C
            SVMClassifier(
                kernel = LIBSVM.Kernel.Linear,
                cost = Float64(C)
            )
        elseif kernel_str == "rbf"
            # RBF kernel: C and gamma
            SVMClassifier(
                kernel = LIBSVM.Kernel.RadialBasis,
                cost = Float64(C),
                gamma = Float64(gamma)
            )
        elseif kernel_str == "sigmoid"
            # Sigmoid kernel: C, gamma, and coef0
            SVMClassifier(
                kernel = LIBSVM.Kernel.Sigmoid,
                cost = Float64(C),
                gamma = Float64(gamma),
                coef0 = Float64(coef0)
            )
        elseif kernel_str == "poly" || kernel_str == "polynomial"
            # Polynomial kernel: C, degree, gamma, and coef0
            SVMClassifier(
                kernel = LIBSVM.Kernel.Polynomial,
                cost = Float64(C),
                gamma = Float64(gamma),
                degree = Int32(degree),
                coef0 = Float64(coef0)
            )
        else
            # Default to RBF
            SVMClassifier(
                kernel = LIBSVM.Kernel.RadialBasis,
                cost = Float64(C),
                gamma = Float64(gamma)
            )
        end
            
        elseif normalizedModelType == :DecisionTreeClassifier
            # Extract Decision Tree hyperparameters
            max_depth = getParam(modelHyperparameters, "max_depth", :max_depth, -1)
            
            # Create Decision Tree model with fixed random seed for reproducibility
            model = DTClassifier(
                max_depth = max_depth,
                rng = Random.MersenneTwister(1)
            )
            
        elseif normalizedModelType == :KNeighborsClassifier
            # Extract kNN hyperparameters
            K = getParam(modelHyperparameters, "K", :K, 3)
            n_neighbors = getParam(modelHyperparameters, "n_neighbors", :n_neighbors, K)
            
            # Create kNN model
            model = kNNClassifier(K = n_neighbors)
            
        else
            error("Unknown model type: $normalizedModelType")
        end
        
        # Create machine object with training data
        mach = MLJ.machine(model, MLJ.table(train_inputs), categorical(train_targets))
        
        # Train the model
        MLJ.fit!(mach, verbosity=0)
        
        # Make predictions on test set
        if normalizedModelType == :SVMClassifier
            # SVM returns CategoricalArray directly
            predictions = MLJ.predict(mach, MLJ.table(test_inputs))
            predictions = string.(predictions)  # Convert to strings for consistency
        else
            # Decision Tree and kNN return UnivariateFiniteArray
            # Need to use mode to get the most likely class
            predictions = mode.(MLJ.predict(mach, MLJ.table(test_inputs)))
            predictions = string.(predictions)  # Convert to strings for consistency
        end
        
        # Compute confusion matrix and metrics for this fold
        metrics = confusionMatrix(predictions, test_targets, classes)
        acc, err, sens, spec, ppv, npv, f1, confusion = metrics
        
        # Store metrics
        push!(accuracy_folds, acc)
        push!(error_folds, err)
        push!(sensitivity_folds, sens)
        push!(specificity_folds, spec)
        push!(ppv_folds, ppv)
        push!(npv_folds, npv)
        push!(f1_folds, f1)
        
        # Accumulate confusion matrix
        global_confusion .+= confusion
    end
    
    # Compute mean and standard deviation for each metric
    stats(vector) = (mean(vector), std(vector; corrected=false))
    
    # Return results in the same format as ANNCrossValidation
    return (stats(accuracy_folds),
            stats(error_folds),
            stats(sensitivity_folds),
            stats(specificity_folds),
            stats(ppv_folds),
            stats(npv_folds),
            stats(f1_folds),
            global_confusion)
end


# ============================================================================
# ENSEMBLE VOTING CLASSIFIER
# ============================================================================

"""
VotingClassifier: Ensemble model that combines predictions from multiple models.
"""
mutable struct VotingClassifier <: MLJ.Probabilistic
    models::Vector
    voting::Symbol
    weights::Union{Nothing, Vector{Float64}}
end

function VotingClassifier(; models=Probabilistic[], voting=:hard, weights=nothing)
    @assert voting in [:hard, :soft] "The only possible labels are :hard or :soft"
    
    normalized_weights = nothing
    if weights !== nothing
        @assert length(weights) == length(models) "Number of weights must match number of models"
        @assert all(w >= 0 for w in weights) "All weights must be non-negative"
        
        # Normalize weights to sum to 1.0
        normalized_weights = Float64.(weights) ./ sum(weights)
    end
    
    return VotingClassifier(models, voting, normalized_weights)
end

function MLJModelInterface.fit(model::VotingClassifier, verbosity::Int, X, y)
    # Train each base model
    machs = [begin
        mm = machine(deepcopy(m), X, y)
        MLJ.fit!(mm, verbosity=0)
        mm
    end for m in model.models]

    fitresults = (
        machines = machs,
        class_levels = collect(levels(y)),   # Kept the levels to use them in the same order
        class_pool = CategoricalArrays.pool(y)
    )
    
    # Save necessary information
    cache = nothing
    report = (n_models=length(model.models), voting=model.voting, weights=model.weights)
    
    return fitresults, cache, report
end

function MLJModelInterface.predict_mode(model::VotingClassifier, fitresult, Xnew)
    machines = fitresult.machines
    class_levels = fitresult.class_levels
    
    # Get predictions from all models
    predictions = [categorical(predict_mode(mach, Xnew), levels=class_levels) for mach in machines]
    
    # Get all unique classes
    n_samples = length(predictions[1])
    n_models = length(machines)
    
    # Determine weights (equal if not specified)
    weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights
    
    # Output Vector with the same type as pthe predictions
    ensemble_pred = similar(predictions[1])
    
    for i in 1:n_samples
        # Count weighted votes for each class
        vote_counts = Dict{eltype(predictions[1][1]), Float64}()
        
        for (j, prediction) in enumerate(predictions)
            vote_counts[prediction[i]] = get(vote_counts, prediction[i], 0.0) + weights[j]
        end
        
        # Have to change this for binary problems(sin usar argmax sobre Dict)
        best_label = nothing
        best_score = -Inf
        for (lbl, sc) in vote_counts
            if sc > best_score
                best_score = sc
                best_label = lbl
            end
        end

        ensemble_pred[i] = best_label
    end

    return ensemble_pred
end

function MLJModelInterface.predict(model::VotingClassifier, fitresult, Xnew)
    machines     = fitresult.machines
    class_levels = fitresult.class_levels
    class_pool   = fitresult.class_pool

    result = if model.voting == :hard
       # Hard voting, prediction based on majority class
        yhat = MLJModelInterface.predict_mode(model, fitresult, Xnew)
        yhat = categorical(yhat; levels=class_levels)  # asegura mismos niveles

        # Return as one-hot encoded predictions
        [MLJBase.UnivariateFinite(
                    class_levels,
                    [lvl == yhat[i] ? 1.0 : 0.0 for lvl in class_levels];
                    pool=class_pool
                ) for i in 1:length(yhat)]
    else
        # Soft voting averaging probabilities
        all_predictions = [predict(mach, Xnew) for mach in machines]

        n_samples = length(all_predictions[1])
        n_models  = length(machines)
        n_classes = length(class_levels)
        weights   = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights

        avg_probs = zeros(n_samples, n_classes)
        for (w, prediction) in zip(weights, all_predictions)
            for i in 1:n_samples
                p_i = prediction[i]
                if p_i isa MLJBase.UnivariateFinite
                    for (j, level) in enumerate(class_levels)
                        avg_probs[i, j] += w * pdf(p_i, level)
                    end
                else
                    # determinista -> one-hot
                    for (j, level) in enumerate(class_levels)
                        avg_probs[i, j] += w * (p_i == level ? 1.0 : 0.0)
                    end
                end
            end
        end

        # Normalize each probability to prevent numerical issues due to floating point arithmetic
        for i in 1:n_samples
            s = sum(@view avg_probs[i, :])
            if s > 0
                @. avg_probs[i, :] = avg_probs[i, :] / s
            end
        end

        # Uses the same pool for consistency than in training
        [MLJBase.UnivariateFinite(class_levels, @view avg_probs[i, :]; pool=class_pool)
         for i in 1:n_samples]
    end

    return result
end

MLJModelInterface.metadata_model(VotingClassifier,
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    load_path="VotingClassifier"
)


function trainClassEnsemble(
    estimator::Symbol,
    modelsHyperParameters::Dict,
    ensembleHyperParameters::Dict,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1}
)   
    SVMClassifier = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
    DTClassifier  = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels verbosity=0


    # 1. Unpack and Prepare Data
    X_mat, Y_bool = trainingDataset
    n_samples = size(X_mat, 1)

    # Convert Boolean/One-Hot to a single vector of Strings
    y_raw = Vector{String}(undef, n_samples)
    if size(Y_bool, 2) == 1
        y_raw .= string.(vec(Y_bool))
    else
        for i in 1:n_samples
            idx = findfirst(identity, @view(Y_bool[i, :]))
            if isnothing(idx)
                error("Sample $i has no true label.")
            end
            y_raw[i] = string(idx)
        end
    end

    # MLJ: Coerce to Categorical BEFORE splitting
    y_categorical = categorical(y_raw)
    classes = levels(y_categorical)

    # 2. Helper to extract parameters safely
    function get_p(dict, key_str, key_sym, default)
        val = get(dict, key_str, get(dict, key_sym, default))
        return val
    end

    # 3. Initialize Metrics
    metrics_store = Dict(
        :acc  => Float64[], :err  => Float64[],
        :sens => Float64[], :spec => Float64[],
        :ppv  => Float64[], :npv  => Float64[],
        :f1   => Float64[]
    )

    # Storage for internal validation
    val_metrics_store = Dict(
        :acc  => Float64[], :err  => Float64[],
        :sens => Float64[], :spec => Float64[],
        :ppv  => Float64[], :npv  => Float64[],
        :f1   => Float64[]
    )

    global_confusion = zeros(Int, length(classes), length(classes))

    numFolds = maximum(kFoldIndices)

    # 4. K-Fold Loop
    for fold in 1:numFolds
        # Stratified Split по заранее заданным индексам
        test_mask  = (kFoldIndices .== fold)
        train_mask = .!test_mask

        X_train = X_mat[train_mask, :]
        X_test  = X_mat[test_mask, :]
        y_train = y_categorical[train_mask]
        y_test  = y_categorical[test_mask]

        # 5. Base Model Generation
        base_model = nothing

        if estimator == :SVC || estimator == :SVMClassifier
            C     = get_p(modelsHyperParameters, "C",        :C,        1.0)
            kern  = get_p(modelsHyperParameters, "kernel",   :kernel,   "rbf")
            gamma = get_p(modelsHyperParameters, "gamma",    :gamma,    0.1)
            deg   = get_p(modelsHyperParameters, "degree",   :degree,   3)
            coef0 = get_p(modelsHyperParameters, "coef0",    :coef0,    0.0)

            k_sym = Symbol(lowercase(string(kern)))
            kernel_val = if k_sym == :linear
                LIBSVM.Kernel.Linear
            elseif k_sym == :sigmoid
                LIBSVM.Kernel.Sigmoid
            elseif k_sym in [:poly, :polynomial]
                LIBSVM.Kernel.Polynomial
            else
                LIBSVM.Kernel.RadialBasis
            end

            base_model = SVMClassifier(
                kernel = kernel_val,
                cost   = Float64(C),
                gamma  = Float64(gamma),
                degree = Int32(deg),
                coef0  = Float64(coef0)
            )

        elseif estimator == :DecisionTreeClassifier
            depth = get_p(modelsHyperParameters, "max_depth", :max_depth, -1)
            base_model = DTClassifier(max_depth = depth,
                                      rng       = Random.MersenneTwister(1))

        elseif estimator == :kNN || estimator == :KNN || estimator == :KNeighborsClassifier
            K_val = get_p(modelsHyperParameters, "n_neighbors", :n_neighbors,
                          get_p(modelsHyperParameters, "K", :K, 3))
            base_model = KNNClassifier(K = Int(K_val))
        else
            error("Unknown estimator: $estimator")
        end

        # 6. Build Homogeneous Ensemble (EnsembleModel)
        n_estimators = get_p(ensembleHyperParameters, "n", :n, 10)
        bag_frac     = get_p(ensembleHyperParameters, "bagging_fraction", :bagging_fraction, 1.0)
        rng_val      = get_p(ensembleHyperParameters, "rng", :rng, Random.MersenneTwister(1))

        rng_obj = isa(rng_val, Integer) ? Random.MersenneTwister(rng_val) : rng_val

        ensemble_model = EnsembleModel(
            model           = base_model,
            n               = n_estimators,
            bagging_fraction = bag_frac,
            rng             = rng_obj
        )

        # 7. Internal validation (hold-out на train)
        val_fraction = get_p(ensembleHyperParameters, "val_fraction", :val_fraction, 0.2)
        
        # Share for INTERNAL training:
        train_inner_fraction = 1.0 - val_fraction 

        train_inner_idx, val_inner_idx = MLJ.partition(
            eachindex(y_train),
            train_inner_fraction; 
            shuffle = true,
            rng    = rng_obj
        )
        X_train_inner = X_train[train_inner_idx, :]
        X_val         = X_train[val_inner_idx, :]
        y_train_inner = y_train[train_inner_idx]
        y_val         = y_train[val_inner_idx]

        # Teach the model on (train_inner) and calculate metrics on validation
        mach_val = machine(ensemble_model, MLJ.table(X_train_inner), y_train_inner)
        fit!(mach_val, verbosity=0)

        val_preds = predict(mach_val, MLJ.table(X_val))

        val_pred_labels = if val_preds[1] isa MLJ.UnivariateFinite
            string.(mode.(val_preds))
        else
            string.(val_preds)
        end

        y_val_str = string.(y_val)

        (v_acc, v_err, v_sens, v_spec, v_ppv, v_npv, v_f1, _) =
            UtilsML1.confusionMatrix(val_pred_labels, y_val_str, string.(classes))

        push!(val_metrics_store[:acc],  v_acc)
        push!(val_metrics_store[:err],  v_err)
        push!(val_metrics_store[:sens], v_sens)
        push!(val_metrics_store[:spec], v_spec)
        push!(val_metrics_store[:ppv],  v_ppv)
        push!(val_metrics_store[:npv],  v_npv)
        push!(val_metrics_store[:f1],   v_f1)

        # 8. Training на всём train для тестовой оценки
        mach = machine(ensemble_model, MLJ.table(X_train), y_train)
        fit!(mach, verbosity=0)

        # 9. Evaluation на тесте
        preds = predict(mach, MLJ.table(X_test))

        y_pred_labels = if preds[1] isa MLJ.UnivariateFinite
            string.(mode.(preds))
        else
            string.(preds)
        end

        y_test_str = string.(y_test)

        (acc, err, sens, spec, ppv, npv, f1, conf) =
            UtilsML1.confusionMatrix(y_pred_labels, y_test_str, string.(classes))

        push!(metrics_store[:acc],  acc)
        push!(metrics_store[:err],  err)
        push!(metrics_store[:sens], sens)
        push!(metrics_store[:spec], spec)
        push!(metrics_store[:ppv],  ppv)
        push!(metrics_store[:npv],  npv)
        push!(metrics_store[:f1],   f1)

        global_confusion .+= conf
    end

    # 10. Aggregation
    agg(v) = (mean(v), std(v; corrected = false))

    return (
        agg(metrics_store[:acc]),
        agg(metrics_store[:err]),
        agg(metrics_store[:sens]),
        agg(metrics_store[:spec]),
        agg(metrics_store[:ppv]),
        agg(metrics_store[:npv]),
        agg(metrics_store[:f1]),
        global_confusion
    )
end


function trainClassEnsemble(
    estimators::AbstractVector{Symbol}, 
    modelsHyperParameters, 
    ensembleHyperParameters::Dict,
    trainingDataset::Tuple{AbstractMatrix{<:Real}, AbstractArray{Bool,2}},
    kFoldIndices::Vector{Int}
)
    # 1. Initialization and data preprocessing
    inputs, outputs_bool = trainingDataset
    N = size(outputs_bool, 1)
    num_classes = size(outputs_bool, 2)
    # Convert boolean label matrix to 1D string labels
    labels = Vector{String}(undef, N)
    if num_classes == 1
        # Binary case: use "true"/"false" as class labels
        for i in 1:N
            labels[i] = outputs_bool[i, 1] ? "true" : "false"
        end
    else
        # Multi-class case: assign "Class1", "Class2", ... based on one-hot position
        for i in 1:N
            # find the index of the true label in one-hot row i
            local idx = findfirst(outputs_bool[i, :] .== true)
            @assert idx !== nothing "No class label found for sample $i"
            labels[i] = "Class$(idx)"
        end
    end
    # Unique class names for confusion matrix calculations
    classes = unique(labels)
    # Metric accumulators for each fold
    accuracy_folds = Float64[]; error_folds = Float64[]
    sensitivity_folds = Float64[]; specificity_folds = Float64[]
    ppv_folds = Float64[]; npv_folds = Float64[]; f1_folds = Float64[]
    # Global confusion matrix (initialized to all zeros)
    global_confusion = zeros(Int, length(classes), length(classes))
    # Determine number of folds from kFoldIndices
    numFolds = maximum(kFoldIndices)
    
    # Helper to retrieve hyperparameter values (accepts string or symbol keys)
    getParam = function(dict, key_str::AbstractString, key_sym::Symbol, default)
        if haskey(dict, key_str)
            return dict[key_str]
        elseif haskey(dict, key_sym)
            return dict[key_sym]
        else
            return default
        end
    end
    
    # Load model types (if not already loaded in the environment)
    ProbabilisticSVC = MLJ.@load ProbabilisticSVC pkg=LIBSVM verbosity=0
    DecisionTreeClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    KNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
    
    # 4. Cross-validation loop over each fold
    for fold in 1:numFolds
        # Split indices for this fold
        local test_mask = (kFoldIndices .== fold)
        local train_mask = (kFoldIndices .!= fold)
        # Partition the data into training and testing sets
        local train_inputs = inputs[train_mask, :]
        local test_inputs  = inputs[test_mask, :]
        local train_targets = labels[train_mask]   # string vector
        local test_targets  = labels[test_mask]    # string vector
        # 2. Instantiate each base model for this fold
        local base_models = Vector{Any}()  # will hold model instances
        for (est, hyp) in zip(estimators, modelsHyperParameters)
            local model = nothing
            # Determine which model to build based on `est` symbol
            if est == :SVC || est == :SVMClassifier || est == :ProbabilisticSVC
                # Support Vector Classifier (LIBSVM)
                local C      = getParam(hyp, "C", :C, 1.0)
                local kernel = getParam(hyp, "kernel", :kernel, "rbf")
                local gamma  = getParam(hyp, "gamma", :gamma, 0.1)
                local degree = getParam(hyp, "degree", :degree, 3)
                local coef0  = getParam(hyp, "coef0", :coef0, 0.0)
                # Choose kernel type
                local kernel_type = kernel == "linear"       ? LIBSVM.Kernel.Linear :
                                     kernel == "rbf"          ? LIBSVM.Kernel.RadialBasis :
                                     kernel == "sigmoid"      ? LIBSVM.Kernel.Sigmoid :
                                     (kernel == "poly" || kernel == "polynomial") ? LIBSVM.Kernel.Polynomial :
                                     LIBSVM.Kernel.RadialBasis  # default to RBF
                # Instantiate SVM model with appropriate parameters
                if kernel_type == LIBSVM.Kernel.Linear
                    model = ProbabilisticSVC(kernel=LIBSVM.Kernel.Linear, cost=Float64(C))
                elseif kernel_type == LIBSVM.Kernel.RadialBasis && kernel == "rbf"
                    model = ProbabilisticSVC(kernel=LIBSVM.Kernel.RadialBasis, cost=Float64(C), gamma=Float64(gamma))
                elseif kernel_type == LIBSVM.Kernel.Sigmoid
                    model = ProbabilisticSVC(kernel=LIBSVM.Kernel.Sigmoid, cost=Float64(C), gamma=Float64(gamma), coef0=Float64(coef0))
                elseif kernel_type == LIBSVM.Kernel.Polynomial
                    model = ProbabilisticSVC(kernel=LIBSVM.Kernel.Polynomial, cost=Float64(C), gamma=Float64(gamma),
                                            degree=Int32(degree), coef0=Float64(coef0))
                else
                    model = ProbabilisticSVC(kernel=LIBSVM.Kernel.RadialBasis, cost=Float64(C), gamma=Float64(gamma))
                end
            elseif est == :DecisionTree || est == :DecisionTreeClassifier
                # Decision Tree classifier
                local max_depth = getParam(hyp, "max_depth", :max_depth, -1)
                model = DecisionTreeClassifier(max_depth=max_depth, rng=Random.MersenneTwister(1))
            elseif est == :kNN || est == :KNNClassifier || est == :KNeighborsClassifier
                # k-Nearest Neighbors classifier
                local K_val = getParam(hyp, "K", :K, 3)
                local n_neighbors = getParam(hyp, "n_neighbors", :n_neighbors, K_val)
                model = KNNClassifier(K = n_neighbors)
            else
                error("Unsupported model type: $est")
            end
            push!(base_models, model)
        end
        
        # 3. Create the VotingClassifier ensemble for this fold
        local voting_strategy = getParam(ensembleHyperParameters, "voting", :voting, :hard)
        local weights = getParam(ensembleHyperParameters, "weights", :weights, nothing)
        # Initialize ensemble model with base models, voting type, and weights
        local ensemble_model = VotingClassifier(models=base_models, voting=voting_strategy, weights=weights)
        # Train the ensemble on this fold's training data
        local ensemble_machine = MLJ.machine(ensemble_model, MLJ.table(train_inputs), categorical(train_targets))
        MLJ.fit!(ensemble_machine, verbosity=0)
        
        # Evaluate on the test set
        # Get predicted class labels (use predict_mode for final ensemble prediction)
        local y_pred = MLJ.predict_mode(ensemble_machine, MLJ.table(test_inputs))
        y_pred = string.(y_pred)  # convert predictions to String vector
        # True labels for this test fold (already strings in test_targets)
        local y_true = test_targets
        # 5. Compute metrics using confusionMatrix from UtilsML1
        local (acc, err, sens, spec, ppv, npv, f1, confusion) = confusionMatrix(y_pred, y_true, classes)
        # Store metrics for this fold
        push!(accuracy_folds, acc)
        push!(error_folds, err)
        push!(sensitivity_folds, sens)
        push!(specificity_folds, spec)
        push!(ppv_folds, ppv)
        push!(npv_folds, npv)
        push!(f1_folds, f1)
        # Accumulate the confusion matrix
        global_confusion .+= confusion
    end
    
    # 6. Compute mean and standard deviation of metrics across folds
    stats(v) = (mean(v), std(v; corrected=false))
    return (
        stats(accuracy_folds),
        stats(error_folds),
        stats(sensitivity_folds),
        stats(specificity_folds),
        stats(ppv_folds),
        stats(npv_folds),
        stats(f1_folds),
        global_confusion
    )
end

end # module