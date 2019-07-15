using Documenter, FeedbackNets, Flux

makedocs(
    sitename = "FeedbackNets Documentation",
    modules = [FeedbackNets, Flux],
    pages = [
        "Home" => "index.md"
        "Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Chains vs Trees" => "guide/chains_vs_trees.md",
            "Working with Networks" => "guide/working_with_networks.md"
        ]
        "Reference" => "reference.md"
    ]
)

deploydocs(
    repo = "github.com/cJarvers/FeedbackNets.jl.git"
)
