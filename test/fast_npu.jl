@testset "FastNPU" begin

    npu = NPU(2,3)
    x = rand(Float32,2,10)
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 3

    loss1() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss1, ps)
    gs = vcat([vec(gs[p]) for p in ps]...)

    fastnpu = FastNPU(2,3)
    loss2(p) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))
    @test DiffEqFlux.paramlength(fastnpu) == 14
    @test length(DiffEqFlux.initial_params(fastnpu)) == 14

    p = Flux.destructure(npu)[1]
    gf = Flux.gradient(loss2, p)[1]
    @test all(gs .== gf)

    loss3(x) = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss3, x)

    loss4(x) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))
    gf = Flux.gradient(loss4, x)

    @test all(gs .== gf)
end

@testset "FastRealNPU" begin

    npu = RealNPU(2,3)
    x = rand(Float32,2,10)
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 2

    loss1() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss1, ps)
    gs = vcat([vec(gs[p]) for p in ps]...)

    fastnpu = FastRealNPU(2,3)
    loss2(p) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))
    @test DiffEqFlux.paramlength(fastnpu) == 8
    @test length(DiffEqFlux.initial_params(fastnpu)) == 8

    p = Flux.destructure(npu)[1]
    gf = Flux.gradient(loss2, p)[1]
    @test all(gs .== gf)

    loss3(x) = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss3, x)

    loss4(x) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))
    gf = Flux.gradient(loss4, x)

    @test all(gs .== gf)
end
