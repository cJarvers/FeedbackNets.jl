using Documenter#, FeedbackConvNets

makedocs(
    sitename = "FeedbackConvNets Documentation",
    #modules = [FeedbackConvNets],
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/cJarvers/FeedbackNets.jl.git"
)
