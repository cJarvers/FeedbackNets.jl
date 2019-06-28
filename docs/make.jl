using Documenter#, FeedbackConvNets

makedocs(
    sitename = "FeedbackConvNets Documentation",
    #modules = [FeedbackConvNets],
    pages = [
        "Home" => "index.md"
        "Guide" => [
            "Getting Started" => "guide/getting_started.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/cJarvers/FeedbackNets.jl.git"
)
