# This scripts trains several feedforward and feedback network architectures on MNIST.
#
# These architectures include the ones trained in:
# Spoerer, McClure & Kriegeskorte (2017). Recurrent Convolutional Neural Networks:
# A Better Model of Biological Object Recognition, Frontiers in Psychology 8, 1551.
#
# It also includes additional architectures with more complex feedback operations.
# For a list of all architectures, see ModelFactory.jl
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
images, labels = MNIST.traindata()
images = DataHelpers.standardize(Float32.(images))
trainimgs, trainlbls = DataHelpers.makebatches(images[:, :, 1:55000], labels[1:55000])
valimgs, vallbls = DataHelpers.makebatches(images[:, :, 55001:end], labels[55001:end])

if usegpu
    trainimgs = gpu.(trainimgs)
    trainlbls = gpu.(trainlbls)
    valimgs = gpu.(valimgs)
    vallbls = gpu.(vallbls)
end

# construct models
@info("Building models ...")
basenet = forwardconvnet(basenetparams)
featnet = forwardconvnet(featnetparams)
kernnet = forwardconvnet(kernnetparams)
slimnet = forwardconvnet(slimnetparams)
largenet = forwardconvnet(largenetparams)
models = basenet, featnet, kernnet, slimnet, largenet
if usegpu
    models = map(gpu, models)
end

function trainmodel(model, trainset, valset; epochs=100, opt=Nesterov(0.1), verbose=false)
    # helper functions
    loss(x, y) = crossentropy(model(x), y)
    accuracy(x, y) = mean([x == y for (x, y) in zip(onecold(model(x)), onecold(y))])
    # training loop
    verbose ? @info("initial accuracy: $(accuracy(valset...))") : nothing
    for e in 1:epochs
        Flux.train!(loss, params(model), trainset, opt)
        verbose ? @info("accuracy after episode $e: $(accuracy(valset...))") : nothing
    end
end

# do the actual training
for model in models
    trainmodel(model, zip(trainimgs, trainlbls), zip(valimgs, vallbls))
    @save "$(model)_MNIST.bson"
end
