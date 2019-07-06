"""
    ModelFactory

A collection of functions to generate the feedback networks and baseline models
used in evaluations and examples.
"""
module ModelFactory
using Flux
using ..Splitters
using ..Mergers
using ..FeedbackChains

export basenetparams, featnetparams, kernnetparams, slimnetparams, largenetparams
# export forwardconvnet, feedbackconvnet
export spoerer_model_fw, spoerer_model_b, spoerer_model_bf, spoerer_model_bk,
       spoerer_model_bl, spoerer_model_bt, spoerer_model_blt
################################################################################
# Parameter definitions
################################################################################
# baseline architectures from Spoerer et al. (2017)
basenetparams = (((3, 3), 1=>32, relu), ((3, 3), 32=>32, relu))
featnetparams = (((3, 3), 1=>45, relu), ((3, 3), 45=>45, relu))
kernnetparams = (((5, 5), 1=>32, relu), ((5, 5), 32=>32, relu))
# additional baseline architectures
slimnetparams = (((3, 3), 1=>17, relu), ((3, 3), 17=>17, relu), ((3, 3), 17=>17, relu),
                 ((3, 3), 17=>17, relu), ((3, 3), 17=>17, relu), ((3, 3), 17=>17, relu),
                 ((3, 3), 17=>17, relu), ((3, 3), 17=>17, relu))
largenetparams = (((3, 3), 1=>32, relu), ((3, 3), 32=>32, relu), ((3, 3), 32=>32, relu),
                  ((3, 3), 32=>32, relu), ((3, 3), 32=>32, relu), ((3, 3), 32=>32, relu),
                  ((3, 3), 32=>32, relu), ((3, 3), 32=>32, relu))

################################################################################
# Network generators
################################################################################
"""
    forwardconvnet(layerparams; classes=10, poolsize=(28, 28))

Construct a feedforward ConvNet like the ones used as baselines in Jarvers & Neumann (2019).
"""
function forwardconvnet(layerparams; classes=10, poolsize=(28, 28))
    # convolution layers according to parametrization
    layers = map(params -> Conv(params..., pad=params[1][1] .÷ 2), layerparams)
    # readout layer with global max-pooling and number of
    features = size(layers[end].weight)[4]
    poolglobal = MaxPool(poolsize)
    flatten(x) = reshape(x, :, size(x, 4))
    readout = Dense(features, classes)
    # build network
    return Chain(layers..., poolglobal, flatten, readout, softmax)
end # function forwardconvnet

"""
    feedbackconvnet(forwardparams, backwardparams, combineop; features=32,
        classes=10, poolsize=(28, 28), examplestate=zeros(Float32, 1,1,1,1))

Construct a ConvNet with feedback connections.
"""
function feedbackconvnet(forwardparams, backwardparams, combineop; features=32,
        classes=10, poolsize=(28, 28), examplestate=zeros(Float32, 1,1,1,1))
    layers = map(params -> Conv(params..., pad=params[1][1] .÷ 2), forwardparams)
    feedbacks = map(params -> Conv(params..., pad=params[1][1] .÷ 2), backwardparams)
    splits = map(i -> Splitter("split_$i"), 1:length(forwardparams))
    #merges = map((i, fb) -> )
    # TODO: complete this function
end # function feedbackconvnet

"""
    spoerer_model_fw(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)

Generate one of the forward models (B, B-K, B-F) from the paper:
  Spoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).
  Recurrent convolutional neural networks: a better model of biological object recognition.
  Frontiers in Psychology 8, 1551.
"""
function spoerer_model_fw(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)
    return Chain(
        Conv(kernel, channels=>features, relu, pad=map(x -> x ÷ 2, kernel)),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool((2,2), stride=(2,2)),
        Conv(kernel, features=>features, relu, pad=map(x -> x ÷ 2, kernel)),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool(map(x -> x ÷ 2, inputsize)),
        flatten,
        Dense(features, classes, σ)
    )
end # function spoerer_model_fw

spoerer_model_b(T; channels=1, inputsize=(28,28), classes=10) =
    spoerer_model_fw(T, channels=channels, inputsize=inputsize, classes=classes)

spoerer_model_bk(T; channels=1, inputsize=(28,28), classes=10) =
    spoerer_model_fw(T, channels=channels, inputsize=inputsize, classes=classes, kernel=(5,5))

spoerer_model_bf(T; channels=1, inputsize=(28,28), classes=10) =
    spoerer_model_fw(T, channels=channels, inputsize=inputsize, classes=classes, features=64)

# TODO: currently, the forward, lateral and backward convolutions all have their
#       own biases. This should not make the model more powerful, as they are
#       combined additively before the non-linearity, but it wastes some resources.
"""
    spoerer_model_bl(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)

Generate the convolutional neural network with lateral recurrence (BL) from:
  Spoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).
  Recurrent convolutional neural networks: a better model of biological object recognition.
  Frontiers in Psychology 8, 1551.
"""
function spoerer_model_bl(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)
    return FeedbackChain(
        Conv(kernel, channels=>features, pad=map(x -> x ÷ 2, kernel)),
        Merger("l1", ConvTranspose(kernel, features=>features, pad=map(x -> x ÷ 2, kernel)), +),
        x -> relu.(x),
        Splitter("l1"),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool((2,2), stride=(2,2)),
        Conv(kernel, features=>features, pad=map(x -> x ÷ 2, kernel)),
        Merger("l2", ConvTranspose(kernel, features=>features, pad=map(x -> x ÷ 2, kernel)), +),
        x -> relu.(x),
        Splitter("l2"),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool(map(x -> x ÷ 2, inputsize)),
        flatten,
        Dense(features, classes, σ)
    )
end # function spoerer_model_bl

"""
    spoerer_model_bt(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)

Generate the convolutional neural network with top-down recurrence (BT) from:
  Spoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).
  Recurrent convolutional neural networks: a better model of biological object recognition.
  Frontiers in Psychology 8, 1551.
"""
function spoerer_model_bt(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)
    return FeedbackChain(
        Conv(kernel, channels=>features, pad=map(x -> x ÷ 2, kernel)),
        Merger("l2",  ConvTranspose((2,2), features=>features, stride=2), +),
        x -> relu.(x),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool((2,2), stride=(2,2)),
        Conv(kernel, features=>features, relu, pad=map(x -> x ÷ 2, kernel)),
        Splitter("l2"),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool(map(x -> x ÷ 2, inputsize)),
        flatten,
        Dense(features, classes, σ)
    )
end # function spoerer_model_bt

"""
    spoerer_model_blt(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)

Generate the convolutional neural network with lateral and top-down recurrence (BLT) from:
  Spoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).
  Recurrent convolutional neural networks: a better model of biological object recognition.
  Frontiers in Psychology 8, 1551.
"""
function spoerer_model_blt(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)
    return FeedbackChain(
        Conv(kernel, channels=>features, pad=map(x -> x ÷ 2, kernel)),
        Merger("l1", ConvTranspose(kernel, features=>features, pad=map(x -> x ÷ 2, kernel)), +),
        Merger("l2", ConvTranspose((2,2), features=>features, stride=2), +),
        x -> relu.(x),
        Splitter("l1"),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool((2,2), stride=(2,2)),
        Conv(kernel, features=>features, pad=map(x -> x ÷ 2, kernel)),
        Merger("l2", ConvTranspose(kernel, features=>features, pad=map(x -> x ÷ 2, kernel)), +),
        x -> relu.(x),
        Splitter("l2"),
        LRN(T(1.0), T(0.0001), T(0.5), 5),
        MaxPool(map(x -> x ÷ 2, inputsize)),
        flatten,
        Dense(features, classes, σ)
    )
end # function spoerer_model_blt

################################################################################
# Helper functions
################################################################################
# helper function to turn Arrays and similar into tuples
tuplefy = (xs...) -> xs

# helper function to turn high-dimensional arrays into 2D (feature x batch)
flatten(x) = reshape(x, :, size(x, 4))

"""
    LRN{T,I}

Local response normalization layer. Input `i` is processed according to
out(x,y,f,b) = x * [b + α * sum( i(x, y, f-k÷2:f+k÷2, b)^2 )]^(-β)

Todo: β is currently ignored (always set to 0.5)
"""
struct LRN{T,I}
    b::T
    α::T
    β::T
    k::I
end

"""
    (l::LRN)(i)

Applies a local response normalization layer according to:
out(x,y,f,b) = x * [c + α * sum( i(x, y, f-k÷2:f+k÷2, b)^2 )]^(-β)

Todo: β is currently ignored (always set to 0.5)
"""
function (l::LRN)(x)
    ω = similar(x)
    fsize = size(x, 3)
    depth = l.k ÷ 2
    buffer = similar(x, size(x, 1), size(x, 2), 2*depth+1, size(x, 4))
    for i ∈ 1:fsize
        f_min = max(1, i - depth)
        f_max = min(fsize, i + depth)
        buffer = Flux.Tracker.data(x[:, :, f_min:f_max, :])
        ω[:, :, i, :] = sum(buffer.^2, dims=3)
    end
    return x ./ sqrt.(l.b .+ l.α .* ω)
end # function (l::LRN)

Flux.@treelike LRN

end # module ModelFactory
