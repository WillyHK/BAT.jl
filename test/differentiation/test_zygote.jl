# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, StatsBase, ValueShapes
using LinearAlgebra
using ForwardDiff, Zygote

@testset "zygote" begin
    prior = NamedTupleDist(
        a = Exponential(),
        b = [4.2, 3.3],
        c = Normal(1, 3),
        d = [Weibull(), Weibull()],
        e = Beta(),
        f = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    )

    likelihood = (;logdensity = v -> sqrt(sum(map(x -> norm(x)^2, values(v)))))

    # Causes Zygote to fail with "Can't differentiate foreigncall expression" exception:
    # likelihood = v -> (;logval = sqrt(sum(map(x -> norm(x)^2, values(v)))))

    pstr = PosteriorDensity(likelihood, prior)

    v = bat_initval(pstr).result

    x = unshaped(v)
    f = logdensityof(unshaped(pstr))
    @test @inferred(f(x)) isa Real

    grad_fw = ForwardDiff.gradient(f, x)
    @test @inferred(Zygote.gradient(f, x)[1]) isa AbstractVector{<:Real}
    @test Zygote.pullback(f, x)[1] == f(x)
    grad_zg = Zygote.gradient(f, x)[1]
    @test grad_fw ≈ grad_zg

    tr_pstr, trafo = bat_transform(PriorToGaussian(), pstr, PriorSubstitution())
    v = bat_initval(tr_pstr).result
    f = logdensityof(tr_pstr)
    @test @inferred(f(x)) isa Real

    grad_fw = ForwardDiff.gradient(f, x)
    # Doesn't work yet:
    #=
    @test @inferred(Zygote.gradient(f, x)[1]) isa AbstractVector{<:Real}
    @test Zygote.pullback(f, x)[1] == f(x)
    grad_zg = Zygote.gradient(f, x)[1]
    @test grad_fw ≈ grad_zg
    =#
end

#=

# Scratch: ==================================================

using BAT: apply_dist_trafo, fwddiff_trafo_src_v
dst_v = Normal(1.5, 0.5)
src_v = Weibull(2, 3)
f = let dst_v = dst_v, src_v = src_v; x -> apply_dist_trafo(dst_v, src_v, x, 0) end
fwddiff_trafo_src_v(dst_v, src_v, 0.5, 0)


# Scratch: ==================================================

=#
