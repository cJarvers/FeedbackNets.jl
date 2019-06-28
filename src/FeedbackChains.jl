module FeedbackChains

using Flux
import Flux: children, mapchildren

using ..Splitters
using ..Mergers
export FeedbackChain

struct FeedbackChain{T<:Tuple}
    layers::T
    FeedbackChain(xs...) = new{typeof(xs)}(xs)
end

# empty feedback chain just returns the input
function (c::FeedbackChain{Tuple{}})(h, x)
    return h, x
end # function (c::FeedbackChain{Tuple{}})

"""
    (c::FeedbackChain)(h, x)

Apply a `FeedbackChain` to input `x` with hidden state `h`. `h` should take the
form of a dictionary mapping `Splitter` names to states.
"""
function (c::FeedbackChain)(h, x)
    newh = Dict{String, Any}()
    for layer ∈ c.layers
        if layer isa Splitter
            newh[layer.name] = x
        elseif layer isa Merger
            x = layer(x, h[layer.splitname])
        else
            x = layer(x)
        end
    end
    return newh, x
end # function (c::FeedbackChain)

children(c::FeedbackChain) = c.layers
mapchildren(f, c::FeedbackChain) = FeedbackChain(f.(c.layers)...)

end # module FeedbackChains
