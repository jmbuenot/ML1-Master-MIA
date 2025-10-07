module UtilsML1

using Flux
using Flux.Losses
using Statistics
using Random

export oneHotEncoding,
       calculateMinMaxNormalizationParameters,
       calculateZeroMeanNormalizationParameters,
       normalizeMinMax!,
       normalizeMinMax,
       normalizeZeroMean!,
       normalizeZeroMean,
       classifyOutputs,
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
    @assert all(in(value, classes) for value in feature) "Se han encontrado valores fuera del conjunto de clases"

    numClasses = length(classes)
    @assert numClasses > 1 "Se necesitan al menos dos clases para codificar"

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
    @assert numOutputs >= 1 "La matriz de salidas debe tener al menos una columna"

    if numOutputs == 1
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims = 2)
        classified = falses(size(outputs))
        classified[indicesMaxEachInstance] .= true
        @assert all(sum(classified, dims = 2) .== 1) "Cada patrón debe asignarse exactamente a una clase"
        return classified
    end
end


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert all(size(outputs) .== size(targets)) "Las matrices de salidas y objetivos deben tener las mismas dimensiones"

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
    @assert all(size(outputs) .== size(targets)) "Las matrices de salidas y objetivos deben tener las mismas dimensiones"

    if size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1]; threshold = threshold)
    else
        return accuracy(classifyOutputs(outputs; threshold = threshold), targets)
    end
end


function buildClassANN(numInputs::Int,
                       topology::AbstractArray{<:Int,1},
                       numOutputs::Int;
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)))
    @assert length(transferFunctions) == length(topology) "Debe haber una función de activación por capa oculta"

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
                       dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
                       maxEpochs::Int = 1000,
                       minLoss::Real = 0.0,
                       learningRate::Real = 0.01)
    inputs, targets = dataset
    @assert size(inputs, 1) == size(targets, 1) "El número de patrones debe coincidir"

    ann = buildClassANN(size(inputs, 2), topology, size(targets, 2);
                        transferFunctions = transferFunctions)

    function loss_fn(model, x, y)
        return (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    end

    inputs_t = permutedims(Float32.(inputs))
    targets_t = permutedims(Float32.(targets))

    training_batch = [(inputs_t, targets_t)]
    trainingLosses = Float32[]

    trainingLoss = loss_fn(ann, inputs_t, targets_t)
    push!(trainingLosses, trainingLoss)
    println("Epoch 0: loss: $trainingLoss")

    opt_state = Flux.setup(Adam(learningRate), ann)

    epoch = 0
    while (epoch < maxEpochs) && (trainingLoss > minLoss)
        Flux.train!(loss_fn, ann, training_batch, opt_state)
        epoch += 1
        trainingLoss = loss_fn(ann, inputs_t, targets_t)
        push!(trainingLosses, trainingLoss)
        println("Epoch $epoch: loss: $trainingLoss")
    end

    return ann, trainingLosses
end

function trainClassANN(topology::AbstractArray{<:Int,1},
                       dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
                       transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
                       maxEpochs::Int = 1000,
                       minLoss::Real = 0.0,
                       learningRate::Real = 0.01)
    inputs, targets = dataset
    targets_matrix = reshape(targets, length(targets), 1)
    return trainClassANN(topology, (inputs, targets_matrix);
                          transferFunctions = transferFunctions,
                          maxEpochs = maxEpochs,
                          minLoss = minLoss,
                          learningRate = learningRate)
end


add(x::Float32, y::Float32) = x + y

add(x::Real, y::Real) = x + y


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
    @assert N >= 0 "El número de patrones no puede ser negativo"
    @assert 0 <= P <= 1 "La proporción debe pertenecer al intervalo [0, 1]"

    num_test = round(Int, P * N)
    ordering = randperm(N)

    test_indices = num_test == 0 ? Int[] : collect(ordering[1:num_test])
    train_indices = num_test == N ? Int[] : collect(ordering[(num_test + 1):end])

    return train_indices, test_indices
end

end # module