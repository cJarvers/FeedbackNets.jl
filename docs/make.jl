using Documenter#, FeedbackConvNets

makedocs(
    sitename = "FeedbackConvNets Documentation",
    #modules = [FeedbackConvNets],
    pages = [
        "Home" => "index.md"
        "Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Chains vs Trees" => "guide/chains_vs_trees.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/cJarvers/FeedbackNets.jl.git"
)
