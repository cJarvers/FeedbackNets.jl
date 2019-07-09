module FeedbackTrees

using Flux
import Flux: children, mapchildren
import Base: getindex
using MacroTools: @forward

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
            newh[splitname(layer)] = x
            x = h[splitname(layer)]
        elseif layer isa Merger
            x = layer(x, h)
        else
            x = layer(x)
        end
    end
    return newh, x
end # function (c::FeedbackTree)

# These overloads ensure that a FeedbackTree behaves as Flux expects, e.g.,
# when moving to gpu or collecting parameters.
children(c::FeedbackTree) = c.layers
mapchildren(f, c::FeedbackTree) = FeedbackTree(f.(c.layers)...)

# These overloads ensure that indexing / slicing etc. work with FeedbackTrees
@forward FeedbackTree.layers Base.getindex, Base.length, Base.first, Base.last,
         Base.iterate, Base.lastindex
getindex(c::FeedbackTree, i::AbstractArray) = FeedbackTree(c.layers[i]...)

end # module FeedbackTrees
