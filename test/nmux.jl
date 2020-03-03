@testset "NMUX" begin

    nmu = NMUX(2,3)
    x = rand(Float32,2,10)
    z = nmu(x)

    @test size(z) == (3,10)
    ps = params(nmu)
    @test length(ps) == 2

    loss() = Flux.mse(nmu(x), reshape(repeat(x[1,:], inner=(3,1)),3,10))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(nmu.Re, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-2))

end
