module StateFulls

using Flux: @functor
import Flux: trainable, reset!

export StateFull

"""
    StateFull{C,S,I}

Wrapper that turns a recurrent cell into a stateful unit, similar to Flux.Recur.
"""
mutable struct StateFull{C,S}
    cell::C
    state::S
    init::S
end # mutable struct StateFull

# convenient constructors
StateFull(c, s, i) = StateFull{typeof(c), typeof(s), typeof(i)}(c, s, i)
StateFull(c, i) = StateFull(c, i, i)

# application: handle state internally
function (sf::StateFull)(x)
    sf.state, y = sf.cell(sf.state, x)
    return y
end

# declarations to turn it into a usable Flux layer
@functor StateFull
trainable(sf::StateFull) = (sf.cell,)
function reset!(sf::StateFull)
    sf.state = sf.init
end # reset

end # module StateFulls
