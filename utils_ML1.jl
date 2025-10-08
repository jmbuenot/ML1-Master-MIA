module UtilsML1

using Flux
using Flux.Losses
using Statistics
using Random

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
       holdOut


function oneHotEncoding(feature::AbstractArray{<:Any,1},
                        classes::AbstractArray{<:Any,1})
    @assert all(in(value, classes) for value in feature) "We found values out of the set of classes"

    numClasses = length(classes)
    @assert numClasses > 1 "We need at least two classes"

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


function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims = 1), maximum(dataset, dims = 1)
end


function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims = 1), std(dataset, dims = 1)
end


function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
                          normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    dataset[:, vec(minValues .== maxValues)] .= 0
    return dataset
end


function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end


function normalizeMinMax(dataset::AbstractArray{<:Real,2},
                         normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeMinMax!(copy(dataset), normalizationParameters)
end


function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
end


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues, stdValues = normalizationParameters
    dataset .-= avgValues
    dataset ./= stdValues
    dataset[:, vec(stdValues .== 0)] .= 0
    return dataset
end


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end


function normalizeZeroMean(dataset::AbstractArray{<:Real,2},
                           normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeZeroMean!(copy(dataset), normalizationParameters)
end


function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))
end


function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real = 0.5)
    numOutputs = size(outputs, 2)
    @assert numOutputs >= 1 "Output matrix has to have at least one column"
    if numOutputs == 1
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims = 2)
        classified = falses(size(outputs))
        classified[indicesMaxEachInstance] .= true
        @assert all(sum(classified, dims = 2) .== 1) "Each pattern must be assigned to one class"
        return classified
    end
end


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert all(size(outputs) .== size(targets)) "Output and target matrix have to have same dimenssions"

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
    @assert all(size(outputs) .== size(targets)) "Output and target matrix have to have same dimenssions"

    if size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1]; threshold = threshold)
    else
        return accuracy(classifyOutputs(outputs; threshold = threshold), targets)
    end
end


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Outputs and targets have to have the same length"

    total = length(outputs)
    @assert total > 0 "We need at least one pattern for the confusion matrix"

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


function confusionMatrix(outputs::AbstractArray{<:Real,1},
                         targets::AbstractArray{Bool,1};
                         threshold::Real = 0.5)
    return confusionMatrix(outputs .>= threshold, targets)
end


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    accuracy_val,
    error_rate,
    sensitivity,
    specificity,
    positive_predictive_value,
    negative_predictive_value,
    fscore,
    confusion = confusionMatrix(outputs, targets)

    println("Accuracy: $accuracy_val")
    println("Error rate: $error_rate")
    println("Sensitivity (Recall): $sensitivity")
    println("Specificity: $specificity")
    println("Positive predictive value (Precision): $positive_predictive_value")
    println("Negative predictive value: $negative_predictive_value")
    println("F-score: $fscore")
    println("Confusion Matrix:")
    println(confusion)
    return nothing
end


function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
                              targets::AbstractArray{Bool,1};
                              threshold::Real = 0.5)
    return printConfusionMatrix(outputs .>= threshold, targets)
end


function buildClassANN(numInputs::Int,
                       topology::AbstractArray{<:Int,1},
                       numOutputs::Int;
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)))
    @assert length(transferFunctions) == length(topology) "It has to be one transfer function per hidden layer"

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
    @assert size(inputs, 1) == size(targets, 1) "Patterns must coincide"

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
            showText && println("Early stopping over $epochsWithoutImprovement epochs without improvement (best epoch: $bestEpoch)")
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



mse(outputs::Array{Float32,1}, targets::Array{Float32,1}) = mean((targets .- outputs) .^ 2)


mse(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{<:Real,1}) = mean((targets .- outputs) .^ 2)


function avgGreaterThan0(valores::AbstractArray{<:Real,1}; return_mask::Bool = false)
    positivos = valores .> 0
    promedio = mean(valores[positivos])
    return return_mask ? (positivos, promedio) : promedio
end


function rango_columnas(matriz::AbstractArray{<:Real,2})
    min_col = minimum(matriz, dims = 1)
    max_col = maximum(matriz, dims = 1)
    return max_col .- min_col
end


loss_regression(m, x, y) = Losses.mse(m(x), y)


loss_binary(m, x, y) = Losses.binarycrossentropy(m(x), y)


loss_multiclass(m, x, y) = Losses.crossentropy(m(x), y)


loss(m, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(m(x), y) : Losses.crossentropy(m(x), y)


function holdOut(N::Int, P::Real)
    @assert N >= 0 "Number of patterns must be positive"
    @assert 0 <= P <= 1 "Proportion must be in [0, 1]"

    num_test = round(Int, P * N)
    ordering = randperm(N)

    test_indices = num_test == 0 ? Int[] : collect(ordering[1:num_test])
    train_indices = num_test == N ? Int[] : collect(ordering[(num_test + 1):end])

    return train_indices, test_indices
end


function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert N >= 0 "Number of patterns must be positive"
    @assert 0 <= Pval <= 1 "Proportion for validation must be in [0, 1]"
    @assert 0 <= Ptest <= 1 "Proportion for test must be in [0, 1]"
    @assert Pval + Ptest <= 1 "Sum of proportions must be as much as 1"

    train_val_indices, test_indices = holdOut(N, Ptest)

    remaining = length(train_val_indices)
    num_val = round(Int, Pval * N)
    num_val = min(num_val, remaining)

    if remaining == 0
        @assert num_val == 0 "No patterns for validation"
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

end # module