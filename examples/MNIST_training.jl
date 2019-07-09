# This scripts shows how to train a network with feedback operations (as well as
# a comparable forward network) on MNIST.
usegpu = false

using Base.Iterators: partition
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy
using MLDatasets
using BSON: @save
using FeedbackNets
if usegpu
    using CuArrays
end

include("DataHelpers.jl")
using .DataHelpers

# load MNIST
@info("Preparing data ...")
batchsize=128
images, labels = MNIST.traindata()
images = DataHelpers.standardize(Float32.(images))
trainimgs, trainlbls = DataHelpers.makebatches(images[:, :, 1:55000], labels[1:55000], batchsize=batchsize)
valimgs, vallbls = DataHelpers.makebatches(images[:, :, 55001:end], labels[55001:end], batchsize=batchsize)
trainimgs = map(x -> reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)), trainimgs)
valimgs = map(x -> reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)), valimgs)
trainlbls = map(x -> onehotbatch(x, 0:9), trainlbls)
vallbls = map(x -> onehotbatch(x, 0:9), vallbls)

if usegpu
    trainimgs = gpu.(trainimgs)
    trainlbls = gpu.(trainlbls)
    valimgs = gpu.(valimgs)
    vallbls = gpu.(vallbls)
end

# construct models
@info("Building models ...")
# Forward network: this is essentially LeNet5, with the difference that ReLUs
# are used and that the first layer uses a padding of 2 in order to accomodate
# input images of size 28x28 instead of 32x32.
forwardnet = Chain(
    Conv((5,5), 1=>6, relu, pad=2),
    MaxPool((2,2), stride=(2,2)),
    Conv((5,5), 6=>16, relu),
    MaxPool((2,2), stride=(2,2)),
    Conv((5,5), 16=>120, relu),
    x -> reshape(x, 120, size(x, 4)),
    Dense(120, 84, relu),
    Dense(84, 10),
    softmax
)
# Feedback network: the same structure as the forward network, but with feedback
# from the output of the second convolution to before the first pooling, and
# from the output of the first fully connected layer (here implemented as a
# convolution) to before the second pooling layer.
feedbacknet = FeedbackChain(
    Conv((5,5), 1=>6, relu, pad=2),
    Merger("conv2", ConvTranspose((10,10), 16=>6, relu, stride=2), +),
    MaxPool((2,2), stride=(2,2)),
    Conv((5,5), 6=>16, relu),
    Splitter("conv2"),
    Merger("fc1", ConvTranspose((10,10), 120=>16, relu), +),
    MaxPool((2,2), stride=(2,2)),
    Conv((5,5), 16=>120, relu),
    Splitter("fc1"),
    x -> reshape(x, 120, size(x, 4)),
    Dense(120, 84, relu),
    Dense(84, 10),
    softmax
)
# generate initial state and wrap feedback net in Recur
h = Dict(
    "conv2" => zeros(Float32, 10, 10, 16, batchsize),
    "fc1" => zeros(Float32, 1, 1, 120, batchsize)
)
feedbacknet = Flux.Recur(feedbacknet, h)

function loss(x, y, model)
    error = crossentropy(model(x), y)
    Flux.reset!(model)
    return(error)
end

function accuracy(x, y, model)
    mean([x == y for (x, y) in zip(onecold(model(x)), onecold(y))])
end

function trainmodel(model, trainset, valset; epochs=1, opt=Nesterov(0.0001),
                    verbose=false, loss=loss, accuracy=accuracy)
    # training loop
    verbose ? @info("initial accuracy: $(accuracy(valset..., model))") : nothing
    for e in 1:epochs
        Flux.train!((x, y) -> loss(x, y, model), params(model), trainset, opt)
        verbose ? @info("accuracy after episode $e: $(accuracy(valset..., model))") : nothing
    end
end

# do the actual training
@info("Training forward network ...")
trainmodel(forwardnet, zip(trainimgs, trainlbls), zip(valimgs, vallbls))
@save "forward_MNIST.bson"

@info("Training feedback network ...")
trainmodel(feedbacknet, zip(trainimgs, trainlbls), zip(valimgs, vallbls))
@save "feedback_MNIST.bson"
