@testset "FastNAU" begin
    nau = NAU(2,3)
    x = rand(Float32,2,10)
    z = nau(x)
    ps = params(nau)

    loss() = sum(abs2, nau(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss, ps)
    gs = vcat([vec(gs[p]) for p in ps]...)

    fastnau = FastNAU(2,3)
    loss(p) = sum(abs2, fastnau(x,p) .- reshape(x[1,:],1,:))

    p = vec(nau.W)
    gf = Flux.gradient(loss, p)[1]
    @test all(gs .== gf)

    loss(x) = sum(abs2, nau(x) .- reshape(x[1,:],1,:))
    gs = Flux.gradient(loss, x)

    loss(x) = sum(abs2, fastnau(x,p) .- reshape(x[1,:],1,:))
    gf = Flux.gradient(loss, x)

    @test all(gs .== gf)
end
