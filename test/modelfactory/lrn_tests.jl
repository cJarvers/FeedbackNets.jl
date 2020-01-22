@testset "LRNs" begin
    @testset "constructor" begin
        # default arguments
        l = LRN(1)
        @test l.c == 1.0
        @test l.α == 1.0
        @test l.β == 0.5
        @test l.size == (1, 1)
        @test l.depth == 5
        # custom arguments
        l = LRN(1, c=2.0, α=1.5, β=0.7, size=(5, 5), depth=3)
        @test l.c == 2.0
        @test l.α == 1.5
        @test l.β == 0.7
        @test l.size == (5, 5)
        @test l.depth == 3
        # custom arguments, standard constructor
        l = LRN(2.0, 1.5, 0.7, (5,5), 3, 1)
        @test l.c == 2.0
        @test l.α == 1.5
        @test l.size == (5, 5)
        @test l.depth == 3
    end # @testset "constructor"

    @testset "apply" begin
        # the first two dimensions are treated as image dims, the third is the
        # feature dimension and the last is batch.
        #
        # For the first few tests, we keep the third dimension at 1, so we can
        # ignore k.
        # We tests on zeros and ones with different biases and alphas:
        arr0 = zeros(5, 5, 1, 4)
        arr1 = ones(5, 5, 1, 4)
        l = LRN(1)
        @test l(arr0) == arr0 # multiplicative scaling does not affect zeros
        @test l(arr1) ≈ arr1 ./ sqrt(2.0)
        scaling = rand()
        c = rand()
        l = LRN(1, c=c)
        @test l(arr0) == arr0
        @test l(arr1) ≈ arr1 ./ sqrt(1.0 + c)
        @test l(scaling .* arr1) ≈ scaling .* arr1 ./ sqrt(scaling^2 + c)
        α = rand()
        l = LRN(1, α=α)
        @test l(arr0) == arr0
        @test l(arr1) ≈ arr1 ./ sqrt(1.0 + α)
        @test l(scaling .* arr1) ≈ scaling .* arr1 ./ sqrt(1.0 + α * scaling^2)
        # Next, test that the summation over the feature dimension happens
        # correctly:
        arr1 = ones(1, 1, 10, 2)
        l = LRN(10, depth=5)
        summed = arr1 .* 5.0
        summed[:, :, 1, :] .= 3.0
        summed[:, :, 2, :] .= 4.0
        summed[:, :, 9, :] .= 4.0
        summed[:, :, 10, :] .= 3.0
        @test l(arr1) ≈ arr1 ./ sqrt.(1.0 .+ summed)
    end # @testset "apply"
end # @testset "LRNs"
