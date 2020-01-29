@testset "NALU" begin
    nalu = NALU(2, 3)
    x = rand(2, 10)
    z = nalu(x)

    @test size(z) == (3,10)
    
    ps = params(nalu)
    @test length(ps) == 4

    loss() = sum(abs, nalu(x))
    gs = Flux.gradient(loss, ps)
    for p in ps
        @test size(gs[p]) == size(p)
    end
end
