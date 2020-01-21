module AbstractMergers

export AbstractMerger

"""
    AbstractMerger

Abstract base type for mergers.

# Interface

Any subtype should support to combine a forward stream with other streams that
can be accessed through a state dictionary via their `Splitter` name.
"""
abstract type AbstractMerger end

end # module AbstractMergers
