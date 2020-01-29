@testset "NAC" begin
    nac = NAC(2, 3)
    x = rand(2, 10)
    z = nac(x)

    @test size(z) == (3,10)
    
    ps = params(nac)
    @test length(ps) == 2

    loss() = sum(abs, nac(x))
    gs = Flux.gradient(loss, ps)
    for p in ps
        @test size(gs[p]) == size(p)
    end
end
