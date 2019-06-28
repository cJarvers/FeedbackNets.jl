# Getting Started

FeedbackNets is a Julia package based on Flux. If you are new to Julia, there
are great learning resources [here](https://julialang.org/learning/) and the
[documentation](https://docs.julialang.org/) is helpful too. In order to get to
know Flux, have a look at their [website](https://fluxml.ai/) and
[documentation](https://fluxml.ai/Flux.jl/stable/).

## Installation

The package can be installed using `Pkg.add()`

```julia
using Pkg
Pkg.add("https://github.com/cJarvers/FeedbackNets.jl.git")
```

or using the REPL shorthand

```julia
] add https://github.com/cJarvers/FeedbackNets.jl.git
```

The package depends on `Flux`. `CuArrays` is required for GPU support.
