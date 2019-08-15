var documenterSearchIndex = {"docs":
[{"location":"#FeedbackConvNets.jl-Docs-1","page":"Home","title":"FeedbackConvNets.jl Docs","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Welcome to the documentation of FeedbackConvNets.jl.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"FeedbackConvNets is a Julia package based on Flux that implements recurrent connections between different layers of a deep network. This makes it possible to have feedback connections from higher/later layers in the networks to lower/earlier layers.","category":"page"},{"location":"#Contents-1","page":"Home","title":"Contents","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Pages = [\n    \"guide/getting_started.md\",\n    \"guide/chains_vs_trees.md\",\n    \"guide/working_with_networks.md\"\n]","category":"page"},{"location":"guide/getting_started/#Getting-Started-1","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"FeedbackNets is a Julia package based on Flux. If you are new to Julia, there are great learning resources here and the documentation is helpful too. In order to get to know Flux, have a look at their website and documentation.","category":"page"},{"location":"guide/getting_started/#Installation-1","page":"Getting Started","title":"Installation","text":"","category":"section"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"The package can be installed using Pkg.add()","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"using Pkg\nPkg.add(\"FeedbackNets\")","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"or using the REPL shorthand","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"] add FeedbackNets","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"The package depends on Flux. CuArrays is required for GPU support.","category":"page"},{"location":"guide/getting_started/#Basic-Usage-1","page":"Getting Started","title":"Basic Usage","text":"","category":"section"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"Once the package is installed, you can access it with Julia's package manager:","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"using FeedbackNets","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"Typically, you'll want to load Flux as well for its network layers:","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"using Flux","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"In Flux, you would build a (feedforward) deep network by concatenating layers in a Chain. For example, the following code generates a two-layer network that maps 10 input units on 20 hidden units (with ReLU-nonlinearity) and maps these to 2 output units:","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"net = Chain(\n    Dense(10, 20, relu),\n    Dense(20, 2)\n)","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"This network can be applied to an input like any function:","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"x = randn(10)\ny = net(x)","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"In order to construct a deep network with feedback, you can use a FeedbackChain, similar to the standard Flux Chain. The difference between a normal Chain and a FeedbackChain is that the latter knows how to treat two specific types of layers: Mergers and Splitters.","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"Imagine that in the network above, we wanted to provide a feedback signal from the two-unit output layer and change activations in the hidden layer based on it. This requires two steps: first we need to retain the value of that layer, second we need to project it back to the hidden layer (e.g., through another Dense layer) and add it to the activations there.","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"The first part is handled by a Splitter. Essentially, whenever the FeedbackChain encounters a Splitter, it saves the output of the previous layer to a dictionary. This way, it can be reused in the next timestep. The second part is handled by a Merger. This layer looks up the value that the Splitter saved to the dictionary, applies some operation to it (in our case, the Dense layer) and merges the result into the forward pass (in our case, by addition):","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"net = FeedbackChain(\n    Dense(10, 20, relu),\n    Merger(\"split1\", Dense(2, 20), +),\n    Dense(20, 2),\n    Splitter(\"split1\")\n)","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"Note that the name \"split1\" is used by both Merger and Splitter. This is how the Merger knows which value from the state dictionary to take. But what happens during the first feedforward pass? The network has not yet encountered the Splitter, so how does the Merger get its value? When a FeedbackChain is applied to an input, it expects to get a dictionary as well, which the user needs to generate for the first timestep. The FeedbackChain returns the updated dictionary as well as the output of the last layer.","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"state = Dict(\"split1\" => zeros(2))\nx = randn(10)\nstate, out = net(state, x)","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"If the user does not want to handle the state manually, they can wrap the net in a Flux Recur, essentially treating the whole network like on recurrent cell:","category":"page"},{"location":"guide/getting_started/#","page":"Getting Started","title":"Getting Started","text":"using Flux: Recur\nnet = Recur(net, state)\noutput = net.([x, x, x])","category":"page"},{"location":"guide/chains_vs_trees/#Controlling-Information-Flow:-Chains-vs-Trees-1","page":"Chains vs Trees","title":"Controlling Information Flow: Chains vs Trees","text":"","category":"section"},{"location":"guide/chains_vs_trees/#","page":"Chains vs Trees","title":"Chains vs Trees","text":"FeedbackNets.jl provides two types to implement deep networks with feedback: FeedbackChains and FeedbackTrees. Their interfaces are identical and they can be used interchangably. The difference between the two is how information flows through the network in the forward pass. Whereas a FeedbackChain propagates information from input to output in a single timestep, a FeedbackTree breaks this up over several timesteps.","category":"page"},{"location":"guide/chains_vs_trees/#FeedbackChains:-Fast-Forward-Passing-1","page":"Chains vs Trees","title":"FeedbackChains: Fast Forward Passing","text":"","category":"section"},{"location":"guide/chains_vs_trees/#","page":"Chains vs Trees","title":"Chains vs Trees","text":"FeedbackChains behave in a way that should be intuitive to users of pure feedforward networks: in each timestep, all layers are applied sequentially to transform input into output. There is feedback across timesteps via Splitters and Mergers, but this does not change the fact that the network can be conceptualized as a sequence of layers.","category":"page"},{"location":"guide/chains_vs_trees/#","page":"Chains vs Trees","title":"Chains vs Trees","text":"However, this means that there is a fundamental asymmetry between information passed in the forward and the backward direction. Imagine a model of ten layers, each of which provides feedback to the previous one. A change in the input will propagate forward to the final layer within one timestep. However, in order for feedback from the top layer to affect what happens in the lowest layer of the network, it has to propagate to layer 9 (which takes one timestep), then to layer 8 (another timestep) and so on. It will take 9 timesteps to reach the first layer.","category":"page"},{"location":"guide/chains_vs_trees/#","page":"Chains vs Trees","title":"Chains vs Trees","text":"This asymmetry is abolished in FeedbackTrees","category":"page"},{"location":"guide/chains_vs_trees/#FeedbackTrees:-Symmetric-Passing-1","page":"Chains vs Trees","title":"FeedbackTrees: Symmetric Passing","text":"","category":"section"},{"location":"guide/chains_vs_trees/#","page":"Chains vs Trees","title":"Chains vs Trees","text":"In a feedback tree, layers are applied to the input in sequence until the first Splitter is encountered. As in a FeedbackChain, the current value is saved to the state dictionary. However, the network then retrieves the value stored at the previous timestep and applies the next layers to that. In the ten-layer network scenario outlined above, this means that ten timesteps are necessary for a new input to affect the output layer. Information spreads with the same speed in the forward and backward direction.","category":"page"},{"location":"guide/working_with_networks/#Working-with-networks-1","page":"Working with Networks","title":"Working with networks","text":"","category":"section"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"There are several points to keep in mind while working with feedback networks.","category":"page"},{"location":"guide/working_with_networks/#Slicing-1","page":"Working with Networks","title":"Slicing","text":"","category":"section"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"Both FeedbackChains and FeedbackTrees support slicing like a normal Flux Chain in order to select a subset of operations in the network.","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"julia> net = FeedbackChain(\n           Merger(\"s1\", Dense(5,10), +),\n           Dense(10,5),\n           Splitter(\"s1\"),\n           Dense(5,1)\n       )\nFeedbackChain(Merger(\"s1\", Dense(5, 10), +), Dense(10, 5), Splitter(\"s1\"), Dense(5, 1))\n\njulia> net[1]\nMerger(\"s1\", Dense(5, 10), +)\n\njulia> net[1:2]\nFeedbackChain(Merger(\"s1\", Dense(5, 10), +), Dense(10, 5))","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"This is convenient to trace the information flow through the network by applying a subset of layers at a time. However, by doing this you run the risk of selecting some Mergers that get input from Splitters which are not in your selected slice. Accordingly, the states required to calculate the next timestep are not added to the dictionary any more. Slicing should therefore be used with care.","category":"page"},{"location":"guide/working_with_networks/#Validating-names-1","page":"Working with Networks","title":"Validating names","text":"","category":"section"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"In order to test whether all inputs required by Mergers in a network are actually provided by corresponding Splitters, you can use the function namesvalid.","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"If each Splitter has a unique name and each Merger name corresponds to a Splitter, validation will succeed.","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"using Flux, FeedbackNets # hide\nnamesvalid(FeedbackChain(\n    Merger(\"s1\", Dense(5,10), +),\n    Dense(10, 5),\n    Splitter(\"s1\")\n))","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"However, if one of these constraints is violated, validation fails.","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"using Flux, FeedbackNets # hide\nnamesvalid(FeedbackChain(\n    Merger(\"s1\", Dense(5,10), +),\n    Dense(10, 5),\n    Splitter(\"s2\")\n))","category":"page"},{"location":"guide/working_with_networks/#Moving-to-GPU-1","page":"Working with Networks","title":"Moving to GPU","text":"","category":"section"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"In order to perform computations on a GPU, the usual Flux syntax can be used to move the model:","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"julia> net = net |> gpu","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"However, this does not work natively for dictionaries and accordingly also not for feedback networks wrapped in a Flux.Recur where the state is encoded as a dictionary. In order to move a dictionary to the GPU, generate a new Dict with the same keys and values moved to GPU:","category":"page"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"julia> state = Dict(key => gpu(val) for (key, val) in pairs(state))","category":"page"},{"location":"guide/working_with_networks/#Reset-1","page":"Working with Networks","title":"Reset","text":"","category":"section"},{"location":"guide/working_with_networks/#","page":"Working with Networks","title":"Working with Networks","text":"A Flux.Recur will keep accumulating gradients via its internal state, also across sequences. In order to prevent this and start from a fresh state for each new sample, you should call Flux.reset!() on your model after each input sequence. Typically, you would do this whenever you calculate the loss or accuracy. See here for details.","category":"page"},{"location":"reference/reference/#Reference-1","page":"Overview","title":"Reference","text":"","category":"section"},{"location":"reference/reference/#","page":"Overview","title":"Overview","text":"FeedbackNets","category":"page"},{"location":"reference/reference/#FeedbackNets","page":"Overview","title":"FeedbackNets","text":"FeedbackNets\n\nImplements deep networks with feedback operations from higher to lower layers. Uses Flux as a backend.\n\n\n\n\n\n","category":"module"},{"location":"reference/reference/#","page":"Overview","title":"Overview","text":"The following reference lists the components of FeedbackNets.jl, including the abstract base types and interfaces, the new layer types, network wrappers, and preimplemented models.","category":"page"},{"location":"reference/reference/#","page":"Overview","title":"Overview","text":"Abstract Base Types\nLayer Types\nNetwork Wrappers\nPreimplemented Models","category":"page"},{"location":"reference/basetypes/#Abstract-Base-Types-1","page":"Base Types","title":"Abstract Base Types","text":"","category":"section"},{"location":"reference/basetypes/#AbstractFeedbackNet-1","page":"Base Types","title":"AbstractFeedbackNet","text":"","category":"section"},{"location":"reference/basetypes/#","page":"Base Types","title":"Base Types","text":"AbstractFeedbackNet is the base type for networks that can handle feedback connections.","category":"page"},{"location":"reference/basetypes/#","page":"Base Types","title":"Base Types","text":"AbstractFeedbackNet","category":"page"},{"location":"reference/basetypes/#FeedbackNets.AbstractFeedbackNets.AbstractFeedbackNet","page":"Base Types","title":"FeedbackNets.AbstractFeedbackNets.AbstractFeedbackNet","text":"AbstractFeedbackNet\n\nAbstract base type for networks that include handling for feedback.\n\nInterface\n\nAny subtype should support iteration (over its layers) in order for the generic method of this type to work.\n\n\n\n\n\n","category":"type"},{"location":"reference/basetypes/#","page":"Base Types","title":"Base Types","text":"Every type that inherits from AbstractFeedbackNet should support iteration over its layers. This is used to implement the splitnames and namesvalid functions in a generic manner.","category":"page"},{"location":"reference/basetypes/#","page":"Base Types","title":"Base Types","text":"splitnames(net::AbstractFeedbackNet)","category":"page"},{"location":"reference/basetypes/#FeedbackNets.AbstractFeedbackNets.splitnames-Tuple{AbstractFeedbackNet}","page":"Base Types","title":"FeedbackNets.AbstractFeedbackNets.splitnames","text":"splitnames(net::AbstractFeedbackNet)\n\nReturn the names of all Splitters in net.\n\n\n\n\n\n","category":"method"},{"location":"reference/basetypes/#","page":"Base Types","title":"Base Types","text":"namesvalid(net::AbstractFeedbackNet)","category":"page"},{"location":"reference/basetypes/#FeedbackNets.AbstractFeedbackNets.namesvalid-Tuple{AbstractFeedbackNet}","page":"Base Types","title":"FeedbackNets.AbstractFeedbackNets.namesvalid","text":"namesvalid(net::AbstractFeedbackNet)\n\nCheck if the input names of all Mergers in net have a corresponding Splitter and that no two Splitters have the same name.\n\n\n\n\n\n","category":"method"},{"location":"reference/basetypes/#AbstractMerger-1","page":"Base Types","title":"AbstractMerger","text":"","category":"section"},{"location":"reference/basetypes/#","page":"Base Types","title":"Base Types","text":"AbstractMerger is the base type for layers that merge several streams (e.g., feedforward and feedback). Any type that inherits from it should implement a function to apply an instance of it to an input and a state dictionary from which to get the feedback streams.","category":"page"},{"location":"reference/basetypes/#","page":"Base Types","title":"Base Types","text":"AbstractMerger","category":"page"},{"location":"reference/basetypes/#FeedbackNets.AbstractMergers.AbstractMerger","page":"Base Types","title":"FeedbackNets.AbstractMergers.AbstractMerger","text":"AbstractMerger\n\nAbstract base type for mergers.\n\nInterface\n\nAny subtype should support to combine a forward stream with other streams that can be accessed through a state dictionary via their Splitter name.\n\n\n\n\n\n","category":"type"},{"location":"reference/layers/#Layer-Types-1","page":"Layer Types","title":"Layer Types","text":"","category":"section"},{"location":"reference/layers/#","page":"Layer Types","title":"Layer Types","text":"Currently, a basic Merger layer and a Splitter layer are implemented. In addition, there are several convenience layers for the preimplemented models.","category":"page"},{"location":"reference/layers/#Mergers-1","page":"Layer Types","title":"Mergers","text":"","category":"section"},{"location":"reference/layers/#","page":"Layer Types","title":"Layer Types","text":"Modules = [Mergers]","category":"page"},{"location":"reference/layers/#FeedbackNets.Mergers.Merger","page":"Layer Types","title":"FeedbackNets.Mergers.Merger","text":"Merger{F,O}\n\nAn element in a FeedbackChain in which the forward stream and feedback stream are combined according to an operation op.\n\nFields\n\nsplitname::String: name of the Splitter node from which the feedback is taken\nfb::F: feedback branch\nop::O: operation to combine forward and feedback branches\n\nDetails\n\nfb typically takes the form of a Flux operation or chain. When a FeedbackChain encounters a Merger, it will look up the state s of the Splitter given by forkname from the previous timestep, apply fb to it and combine it with the forward input x according  to op(x, fb(s))\n\n\n\n\n\n","category":"type"},{"location":"reference/layers/#FeedbackNets.Mergers.inputname-Tuple{Merger}","page":"Layer Types","title":"FeedbackNets.Mergers.inputname","text":"inputname(m::Merger)\n\nReturn the name of the Splitter from which m gets its input.\n\n\n\n\n\n","category":"method"},{"location":"reference/layers/#Splitters-1","page":"Layer Types","title":"Splitters","text":"","category":"section"},{"location":"reference/layers/#","page":"Layer Types","title":"Layer Types","text":"Modules = [Splitters]","category":"page"},{"location":"reference/layers/#FeedbackNets.Splitters.Splitter","page":"Layer Types","title":"FeedbackNets.Splitters.Splitter","text":"Splitter\n\nAn element in a FeedbackChain that marks locations where a feedback branch forks off from the forward branch.\n\nFields\n\nname::String: unique name used to identify the fork for the backward pass.\n\nDetails\n\nIn the forward stream, a is essentially an identity operation. It only alerts the FeedbackChain to add the current Array to the chain's state and mark it with the Splitters name so that Mergers can access it for feedback to the next timestep.\n\n\n\n\n\n","category":"type"},{"location":"reference/layers/#FeedbackNets.Splitters.splitname-Tuple{Splitter}","page":"Layer Types","title":"FeedbackNets.Splitters.splitname","text":"splitname(s::Splitter)\n\nReturn name of s.\n\n\n\n\n\n","category":"method"},{"location":"reference/layers/#Other-layers-1","page":"Layer Types","title":"Other layers","text":"","category":"section"},{"location":"reference/layers/#","page":"Layer Types","title":"Layer Types","text":"The preimplemented models use a flattening layer and local response normalization.","category":"page"},{"location":"reference/layers/#","page":"Layer Types","title":"Layer Types","text":"flatten(x)","category":"page"},{"location":"reference/layers/#FeedbackNets.ModelFactory.Flatten.flatten-Tuple{Any}","page":"Layer Types","title":"FeedbackNets.ModelFactory.Flatten.flatten","text":"flatten(x)\n\nTurns a high-dimensional array (e.g., a batch of feature maps) into a 2-d array, linearizing all except the last (batch) dimension.\n\n\n\n\n\n","category":"method"},{"location":"reference/layers/#","page":"Layer Types","title":"Layer Types","text":"Modules = [LRNs]","category":"page"},{"location":"reference/layers/#FeedbackNets.ModelFactory.LRNs","page":"Layer Types","title":"FeedbackNets.ModelFactory.LRNs","text":"Implementation of local response normalization.\n\n\n\n\n\n","category":"module"},{"location":"reference/layers/#FeedbackNets.ModelFactory.LRNs.LRN","page":"Layer Types","title":"FeedbackNets.ModelFactory.LRNs.LRN","text":"LRN{T,I}\n\nLocal response normalization layer. Input i is processed according to out(x,y,f,b) = x * [b + α * sum( i(x, y, f-k÷2:f+k÷2, b)^2 )]^(-β)\n\nTodo: β is currently ignored (always set to 0.5)\n\n\n\n\n\n","category":"type"},{"location":"reference/layers/#FeedbackNets.ModelFactory.LRNs.LRN-Tuple{Any}","page":"Layer Types","title":"FeedbackNets.ModelFactory.LRNs.LRN","text":"(l::LRN)(i)\n\nApplies a local response normalization layer according to: out(x,y,f,b) = x * [c + α * sum( i(x, y, f-k÷2:f+k÷2, b)^2 )]^(-β)\n\nTodo: β is currently ignored (always set to 0.5)\n\n\n\n\n\n","category":"method"},{"location":"reference/networks/#Network-Wrappers-1","page":"Network Types","title":"Network Wrappers","text":"","category":"section"},{"location":"reference/networks/#","page":"Network Types","title":"Network Types","text":"Two types of networks are currently implemented: FeedbackChains and FeedbackTrees. For a comparison, see Controlling Information Flow: Chains vs Trees","category":"page"},{"location":"reference/networks/#FeedbackChains-1","page":"Network Types","title":"FeedbackChains","text":"","category":"section"},{"location":"reference/networks/#","page":"Network Types","title":"Network Types","text":"Modules = [FeedbackChains]","category":"page"},{"location":"reference/networks/#FeedbackNets.FeedbackChains.FeedbackChain","page":"Network Types","title":"FeedbackNets.FeedbackChains.FeedbackChain","text":"FeedbackChain{T<:Tuple}\n\nTuple-like structure similar to a Flux.Chain with support for Splitters and Mergers.\n\n\n\n\n\n","category":"type"},{"location":"reference/networks/#FeedbackNets.FeedbackChains.FeedbackChain-Tuple{Any,Any}","page":"Network Types","title":"FeedbackNets.FeedbackChains.FeedbackChain","text":"(c::FeedbackChain)(h, x)\n\nApply a FeedbackChain to input x with hidden state h. h should take the form of a dictionary mapping Splitter names to states.\n\n\n\n\n\n","category":"method"},{"location":"reference/networks/#FeedbackTrees-1","page":"Network Types","title":"FeedbackTrees","text":"","category":"section"},{"location":"reference/networks/#","page":"Network Types","title":"Network Types","text":"Modules = [FeedbackTrees]","category":"page"},{"location":"reference/networks/#FeedbackNets.FeedbackTrees.FeedbackTree-Tuple{Any,Any}","page":"Network Types","title":"FeedbackNets.FeedbackTrees.FeedbackTree","text":"(c::FeedbackTree)(h, x)\n\nApply a FeedbackTree to input x with hidden state h. h should take the form of a dictionary mapping Splitter names to states.\n\n\n\n\n\n","category":"method"},{"location":"reference/preimplemented/#Preimplemented-Models-1","page":"Preimplemented Models","title":"Preimplemented Models","text":"","category":"section"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"ModelFactory","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory","text":"ModelFactory\n\nA collection of functions to generate the feedback networks and baseline models used in evaluations and examples.\n\n\n\n\n\n","category":"module"},{"location":"reference/preimplemented/#LeNet5-1","page":"Preimplemented Models","title":"LeNet5","text":"","category":"section"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"LeNet5","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.LeNet5","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.LeNet5","text":"Reimplementation of the LeNet5 architecture from\n\nLeCun, Bottou, Bengio & Haffner (1998), Gradient-based learning applied to document recognition. Procedings of the IEEE 86(11), 2278-2324.\n\nand a version of LeNet5 with feedback connections.\n\n\n\n\n\n","category":"module"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"ModelFactory.jl contains a modified version of the LeNet5 architecture from","category":"page"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"LeCun, Bottou, Bengio & Haffner (1998),\nGradient-based learning applied to document recognition.\nProcedings of the IEEE 86(11), 2278-2324.","category":"page"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"as well as a version with feedback connections.","category":"page"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"lenet5","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.LeNet5.lenet5","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.LeNet5.lenet5","text":"lenet5(; σ=tanh, pad=2)\n\nGenerate the LeNet5 architecture from LeCun et al. (1998) with small modifications.\n\nDetails\n\nThe implementation differs from the original LeNet5, as the output layer does not compute radial basis functions, but is a normal Dense layer with a softmax. The input image is padded. The network assumes a 32x32 input, so for MNIST digits a pad of 2 is appropriate. The non-linearity can be customized via the σ argument. The standard is tanh, whereas the original LeNet5 used x -> 1.7159 .* tanh(x).\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"lenet5_fb","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.LeNet5.lenet5_fb","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.LeNet5.lenet5_fb","text":"lenet5_fb(; σ=tanh, pad=2)\n\nGenerate the LeNet5 architecture from LeCun et al. (1998) with feedback connections and small modifications.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"In addition, there is a wrapper to more easily generate a Flux.Recur for the feedback model.","category":"page"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"wrapfb_lenet5","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.LeNet5.wrapfb_lenet5","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.LeNet5.wrapfb_lenet5","text":"wrapfb_lenet5(net, batchsize; generator=zeros)\n\nWrap a letnet5_fb network in a Flux.Recur, assuming that batches are of size batchsize and using the given generator to initialize the state.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#Networks-by-Spoerer-et-al.-1","page":"Preimplemented Models","title":"Networks by Spoerer et al.","text":"","category":"section"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"Spoerer2017","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017","text":"This module reimplements models from the paper:\n\nSpoerer, C.J., McClure, P. and Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"module"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"The paper contains six network architectures:","category":"page"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"spoerer_model_b","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_b","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_b","text":"spoerer_model_b(T; channels=1, inputsize=(28, 28), classes=10)\n\nGenerate the bottom-up (B) convolutional neural network from:\n\nSpoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"spoerer_model_bk","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bk","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bk","text":"spoerer_model_bk(T; channels=1, inputsize=(28, 28), classes=10)\n\nGenerate the convolutional neural network with increased kernel size (BK) from:\n\nSpoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"spoerer_model_bf","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bf","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bf","text":"spoerer_model_bf(T; channels=1, inputsize=(28, 28), classes=10)\n\nGenerate the convolutional neural network with additional feature maps (BF) from:\n\nSpoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"spoerer_model_bl","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bl","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bl","text":"spoerer_model_bl(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)\n\nGenerate the convolutional neural network with lateral recurrence (BL) from:\n\nSpoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"spoerer_model_bt","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bt","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_bt","text":"spoerer_model_bt(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)\n\nGenerate the convolutional neural network with top-down recurrence (BT) from:\n\nSpoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"spoerer_model_blt","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_blt","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_blt","text":"spoerer_model_blt(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)\n\nGenerate the convolutional neural network with lateral and top-down recurrence (BLT) from:\n\nSpoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"function"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"The first three architectures (B, BK, BF) are feedforward and are internally implemented with one function:","category":"page"},{"location":"reference/preimplemented/#","page":"Preimplemented Models","title":"Preimplemented Models","text":"ModelFactory.Spoerer2017.spoerer_model_fw","category":"page"},{"location":"reference/preimplemented/#FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_fw","page":"Preimplemented Models","title":"FeedbackNets.ModelFactory.Spoerer2017.spoerer_model_fw","text":"spoerer_model_fw(T; channels=1, inputsize=(28, 28), kernel=(3,3), features=32, classes=10)\n\nGenerate one of the forward models (B, B-K, B-F) from the paper:\n\nSpoerer, C.J., McClure, P. & Kriegeskorte, N. (2017).\nRecurrent convolutional neural networks: a better model of biological object recognition.\nFrontiers in Psychology 8, 1551.\n\n\n\n\n\n","category":"function"}]
}
