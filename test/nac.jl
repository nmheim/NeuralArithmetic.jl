@testset "NAC" begin
    nac = NAC(2, 3)
    x = rand(2, 10)
    z = nac(x)

    @test size(z) == (3,10)
    
    ps = params(nac)
    @test length(ps) == 2

    loss() = Flux.mse(nac(x), reshape(repeat(x[1,:], inner=(3,1)),3,10))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(weights(nac), [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-2))
end
