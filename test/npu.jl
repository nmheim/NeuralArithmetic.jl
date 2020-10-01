@testset "NPU" begin

    npu = NPU(2,3) |> gpu
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
    @test all(isapprox.(npu.Re[:,1], ones(3), atol=1e-3))
    @test all(isapprox.(NeuralArithmetic.gateclip(npu.g), [1.0, 0.0], atol=1e-3))

end

@testset "RealNPU" begin

    npu = RealNPU(2,3) |> gpu
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
    @test all(isapprox.(npu.Re[:,1], ones(3), atol=1e-3))
    @test all(isapprox.(NeuralArithmetic.gateclip(npu.g), [1.0, 0.0], atol=1e-3))

end

@testset "NaiveNPU" begin
    Re, Rē  = randn(3,2), randn(3,2)
    Im, Im̄  = randn(3,2), randn(3,2)
    x , x̄   = randn(2),   randn(2)
    rrule_test(NeuralArithmetic.mult, randn(3), (Re,Rē), (Im,Im̄), (x,x̄), rtol=1e-3, atol=1e-3)

    npu = NaiveNPU(2,3) |> gpu
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
    @test all(isapprox.(npu.Re, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-3))

end

@testset "RealNaiveNPU" begin

    npu = RealNaiveNPU(2,3) |> gpu
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
    @test all(isapprox.(npu.Re, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-3))

end

@testset "gateclip" begin
    g = [1.5, 0.5, -0.5]
    @test isapprox(NeuralArithmetic.gateclip(g), [1.,0.5,0.])
    f(g) = sum(NeuralArithmetic.gateclip(g))
    dg = Flux.gradient(f, g)[1]
    @test isapprox(dg, ones(3))
end
