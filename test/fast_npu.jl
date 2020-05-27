@testset "FastGatedNPUX" begin

    npu = GatedNPUX(2,3)
    x = rand(Float32,2,10)
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 3

    loss() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss, ps)
    gs = vcat([vec(gs[p]) for p in ps]...)

    fastnpu = FastGatedNPUX(2,3)
    loss(p) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))

    p = Flux.destructure(npu)[1]
    gf = Flux.gradient(loss, p)[1]
    @test all(gs .== gf)

    loss(x) = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss, x)

    loss(x) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))
    gf = Flux.gradient(loss, x)

    @test all(gs .== gf)
end

@testset "FastGatedNPU" begin

    npu = GatedNPU(2,3)
    x = rand(Float32,2,10)
    z = npu(x)

    @test size(z) == (3,10)
    ps = params(npu)
    @test length(ps) == 2

    loss() = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss, ps)
    gs = vcat([vec(gs[p]) for p in ps]...)

    fastnpu = FastGatedNPU(2,3)
    loss(p) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))

    p = Flux.destructure(npu)[1]
    gf = Flux.gradient(loss, p)[1]
    @test all(gs .== gf)

    loss(x) = sum(abs2, npu(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss, x)

    loss(x) = sum(abs2, fastnpu(x,p) .- reshape(x[1,:],1,:))
    gf = Flux.gradient(loss, x)

    @test all(gs .== gf)
end
