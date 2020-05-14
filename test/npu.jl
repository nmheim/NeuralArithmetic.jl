@testset "NPU" begin

    npu = NPU(2,3) |> gpu
    x = rand(Float32,2,10) |> gpu
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 1

    loss() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    npu = npu |> cpu
    @test all(isapprox.(npu.W, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-2))

end

@testset "NPUX" begin

    npu = NPUX(2,3) |> gpu
    x = rand(Float32,2,10) |> gpu
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 2

    loss() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    npu = npu |> cpu
    @test all(isapprox.(npu.Re, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-2))

end

@testset "GatedNPU" begin

    npu = GatedNPU(2,3) |> gpu
    x = rand(Float32,2,10) |> gpu
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 2

    loss() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    npu = npu |> cpu
    @test all(isapprox.(npu.W[:,1], ones(3), atol=1e-2))
    @test all(isapprox.(npu.g, [1.0, 0.0], atol=1e-2))

end

@testset "GatedNPUX" begin

    npu = GatedNPUX(2,3) |> gpu
    x = rand(Float32,2,10) |> gpu
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 3

    loss() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    opt = RMSProp()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    npu = npu |> cpu
    @test all(isapprox.(npu.Re[:,1], ones(3), atol=1e-2))
    @test all(isapprox.(npu.g, [1.0, 0.0], atol=1e-2))

end
