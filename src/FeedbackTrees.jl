module FeedbackTrees

using Flux
import Flux: children, mapchildren

using ..Splitters
using ..Mergers
export FeedbackTree

struct FeedbackTree{T<:Tuple}
    layers::T
    FeedbackTree(xs...) = new{typeof(xs)}(xs)
end

# empty feedback tree just returns the input
function (c::FeedbackTree{Tuple{}})(h, x)
    return h, x
end # function (c::FeedbackTree{Tuple{}})

"""
    (c::FeedbackTree)(h, x)

Apply a `FeedbackTree` to input `x` with hidden state `h`. `h` should take the
form of a dictionary mapping `Splitter` names to states.
"""
function (c::FeedbackTree)(h, x)
    newh = Dict{String, Any}()
    for layer ∈ c.layers
        if layer isa Splitter
            newh[layer.name] = x
            x = h[layer.name]
        elseif layer isa Merger
            x = layer(x, h[layer.splitname])
        else
            x = layer(x)
        end
    end
    return newh, x
end # function (c::FeedbackTree)

children(c::FeedbackTree) = c.layers
mapchildren(f, c::FeedbackTree) = FeedbackTree(f.(c.layers)...)

end # module FeedbackTrees
