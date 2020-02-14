@testset "NALU" begin
    nalu = NALU(2, 3)
    x = rand(2, 10) .+ 1
    z = nalu(x)

    @test size(z) == (3,10)
    
    ps = params(nalu)
    @test length(ps) == 4

    loss() = Flux.mse(nalu(x), reshape(repeat(x[1,:] .* x[2,:], inner=(3,1)),3,10))
    opt = ADAM(0.5)
    train_data = Iterators.repeated((), 1000)
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(weights(nalu.nac), [1.0 1.0; 1.0 1.0; 1.0 1.0], atol=1e-2))
    @test all(isapprox.(gate(nalu, x[:,1]), [0.0, 0.0, 0.0], atol=1e-3))
end
