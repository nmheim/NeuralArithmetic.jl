@testset "FastNAU" begin
    nau = NAU(2,3)
    x = rand(Float32,2,10)
    z = nau(x)
    ps = params(nau)

    loss() = sum(abs2, nau(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss, ps)
    gs = vcat([vec(gs[p]) for p in ps]...)

    fastnau = FastNAU(2,3)
    loss1(p) = sum(abs2, fastnau(x,p) .- reshape(x[1,:],1,:))
    @test DiffEqFlux.paramlength(fastnau) == 6
    @test length(DiffEqFlux.initial_params(fastnau)) == 6

    p = vec(nau.W)
    gf = Flux.gradient(loss1, p)[1]
    @test all(gs .== gf)

    loss2(x) = sum(abs2, nau(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss2, x)

    loss3(x) = sum(abs2, fastnau(x,p) .- reshape(x[1,:],1,:))
    gf = Flux.gradient(loss3, x)

    @test all(gs .== gf)
end
