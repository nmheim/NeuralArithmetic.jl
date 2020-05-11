@testset "NPU" begin

    npu = NPU(2,3)
    x = rand(Float32,2,10)
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 1

    loss() = Flux.mse(npu(x), reshape(repeat(x[1,:], inner=(3,1)),3,10))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(npu.W, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-2))

end

@testset "NPUX" begin

    npu = NPUX(2,3)
    x = rand(Float32,2,10)
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 2

    loss() = Flux.mse(npu(x), reshape(repeat(x[1,:], inner=(3,1)),3,10))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(npu.Re, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-2))

end

@testset "GatedNPU" begin

    npu = GatedNPU(2,3)
    x = rand(Float32,2,10)
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 2

    loss() = Flux.mse(npu(x), reshape(repeat(x[1,:], inner=(3,1)),3,10))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(npu.W[:,1], ones(3), atol=1e-2))
    @test all(isapprox.(npu.g, [1.0, 0.0], atol=1e-2))

end
