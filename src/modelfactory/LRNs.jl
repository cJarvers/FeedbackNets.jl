"""
Implementation of local response normalization.
"""
module LRNs
using Flux
import Flux: params

export LRN

"""
    LRN{T,I}

Local response normalization layer. Input `i` is processed according to
out(x,y,f,b) = x / [c + α * sum( i(x-k÷2:x+k÷2, y-k÷2:y+k÷2, f-n÷2:f+n÷2, b)^2 )]^β

Todo: β is currently ignored (always set to 0.5)

# Arguments
- `c::T`: additive constant for normalization sum. Default 1.0
- `α::T`: multiplicative scaling constant for normalization sum. Default 1.0
- `β::T`: power scaling constant for normalization sum. Default 0.5
- `size::Tuple{I, I}`: size of normalization field in image space. Default (1, 1)
- `depth::I`: number of neighboring feature maps that go into normalization. Default 5.
"""
struct LRN{T,I,A}
    c::T
    α::T
    β::T
    size::Tuple{I, I}
    depth::I
    kernel::A
end

# convenience constructor that generates the appropriate summation kernel
function LRN(c::T, α::T, β::T, size::Tuple{I, I}, depth::I, features::I) where {T, I}
    # using convolution to do the sum, like https://github.com/FluxML/Flux.jl/pull/720
    # create kernel:
    kernel = zeros(T, size[1], size[2], features, features)
    for i in 1:features
        lower = max(i-depth÷2, 1)
        upper = min(i+depth÷2, features)
        kernel[:, :, lower:upper, i] .= 1
    end
    LRN{T, I, typeof(kernel)}(c, α, β, size, depth, kernel)
end

# convenience constructor with default arguments
function LRN(features; c=1.0, α=1.0, β=0.5, size=(1, 1), depth=5)
    return LRN(c, α, β, size, depth, features)
end # function LRN



"""
    (l::LRN)(i)

Applies a local response normalization layer according to:
out(x,y,f,b) = x ./ [c + α * sum( i(x, y, f-n÷2:f+n÷2, b)^2 )]^(β)

Todo: β is currently ignored (always set to 0.5)
"""
function (l::LRN)(x)
    norm = l.c .+ l.α .* conv(x .^ 2, l.kernel)
    norm = norm .^ l.β
    return x ./ norm
end # function (l::LRN)

Flux.params(l::LRN) = nothing
Flux.@functor LRN

end # module LRNs
