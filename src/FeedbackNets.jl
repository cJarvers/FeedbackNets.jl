"""
    FeedbackNets

Implements deep networks with feedback operations from higher to lower layers.
Uses Flux as a backend.
"""
module FeedbackNets
using Reexport

include("Splitters.jl")
include("Mergers.jl")
include("FeedbackChains.jl")
include("FeedbackTrees.jl")
include("ModelFactory.jl")
@reexport using .Splitters
@reexport using .Mergers
@reexport using .FeedbackChains
@reexport using .FeedbackTrees
@reexport using .ModelFactory

end # module FeedbackNets
