@testset "NMU" begin
    nmu = NMU(2,3)
    x = rand(2,10)
    z = nmu(x)

    @test size(z) == (3,10)
    ps = params(nmu)
    @test length(ps) == 1

    loss() = Flux.mse(nmu(x), reshape(repeat(x[1,:], inner=(3,1)),3,10))
    opt = ADAM()
    train_data = Iterators.repeated((), 5000)
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(nmu.W, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-4))
end
