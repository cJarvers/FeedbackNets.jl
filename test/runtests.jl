using Test
using Flux
using FeedbackNets

@testset "FeedbackNets" begin

include("splitter_tests.jl")
include("merger_tests.jl")
include("feedbackchain_tests.jl")
include("feedbacktree_tests.jl")

end # @testset "FeedbackNets"
