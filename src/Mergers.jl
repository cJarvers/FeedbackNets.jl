module Mergers

import Flux: children, mapchildren

export Merger

"""
    Merger{F,O}

An element in a `FeedbackChain` in which the forward stream and feedback stream
are combined according to an operation `op`.

# Fields
- `splitname::String`: name of the `Splitter` node from which the feedback is taken
- `fb::F`: feedback branch
- `op::O`: operation to combine forward and feedback branches

# Details
`fb` typically takes the form of a Flux operation or chain. When a `FeedbackChain`
encounters a `Merger`, it will look up the state `s` of the `Splitter` given by `forkname`
from the previous timestep, apply `fb` to it and combine it with the forward input `x`
according  to `op(x, fb(s))`
"""
struct Merger{F,O}
    splitname::String
    fb::F
    op::O
end # struct Merger

function (m::Merger)(x, y)
    m.op(x, m.fb(y))
end

children(m::Merger) = (m.fb, m.op)
mapchildren(f, m::Merger) = Merger(m.splitname, f(m.fb), m.op)

end # module Mergers
