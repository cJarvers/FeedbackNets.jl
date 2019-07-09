module AbstractFeedbackNets

using ..Splitters

export AbstractFeedbackNet, splitnames

"""
    AbstractFeedbackNet

Abstract base type for networks that include handling for feedback.

# Interface

Any subtype should support iteration (over its layers) in order for the generic
method of this type to work.
"""
abstract type AbstractFeedbackNet end

"""
    splitnames(net::AbstractFeedbackNet)

Return the names of all `Splitter`s in `net`.
"""
function splitnames(net::AbstractFeedbackNet)
    names = Vector{String}()
    for layer in net
        if layer isa Splitter
            push!(names, splitname(layer))
        end
    end
    return names
end # function splitnames

end # module AbstractFeedbackNets
