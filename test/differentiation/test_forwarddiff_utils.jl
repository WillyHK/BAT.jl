# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, StaticArrays
using ForwardDiff, ChainRulesCore, Zygote

@testset "forwarddiff_utils" begin
    f = (x...) -> (sum(map(x -> norm(x)^2, x)), sum(map(x -> norm(x)^3, x)), 42)
    xs = (2f0, (3, 4f0), SVector(5f0, 6f0, 7f0))
    ΔΩ = (10, 20, 30)

    fv = (x...) -> SVector(sum(map(x -> norm(x)^2, x)), sum(map(x -> norm(x)^3, x)), 42)
    ΔΩv = SVector(10, 20, 30)

    BAT.partial_forwarddiff_fwd_back(f, xs, Val(1), ΔΩ)
    BAT.partial_forwarddiff_fwd_back(f, xs, Val(2), ΔΩ)
    BAT.partial_forwarddiff_fwd_back(f, xs, Val(3), ΔΩ)

    for i in eachindex(xs)
        y, back = ChainRulesCore.rrule(BAT.WithForwardDiff(f), xs...)
        map(ChainRulesCore.unthunk, back(ΔΩ))
    end


    y, back = Zygote.pullback(BAT.WithForwardDiff(f), xs...)
    back(ΔΩ)


    function f_loss_1(xs...)
        r = BAT.WithForwardDiff(f)(xs...)
        @assert sum(r) < 10000
        sum(r[1])
    end
    Zygote.gradient(f_loss_1, 1,2,3)

    function f_loss_3(xs...)
        r = BAT.WithForwardDiff(f)(xs...)
        @assert sum(r) < 10000
        sum(r[3])
    end
    Zygote.gradient(f_loss_3, 1,2,3)

    function f_loss_3z(xs...)
        r = f(xs...)
        @assert sum(r) < 10000
        sum(r[3])
    end
    Zygote.gradient(f_loss_3z, 1,2,3)


    # Old:

    DTf32 = ForwardDiff.Dual{ForwardDiff.Tag{typeof(f),Float32}}
    @test @inferred(BAT.forwarddiff_eval(f, 4f0)) == (DTf32(16.0,8.0), DTf32(64.0,48.0), 42)
    rdf = (DTf32(26.0,6.0,2.0,8.0), DTf32(92.0,27.0,3.0,48.0), 42)
    @test @inferred(BAT.forwarddiff_eval(f, 3, true, 4f0)) == rdf
    @test @inferred(BAT.forwarddiff_eval(fv, 3, true, 4f0)) == SVector(rdf)

    x = SVector(3, true, 4f0)
    r_ref = (ntuple(i -> DTf32(f(x)[i], ForwardDiff.jacobian(x -> SVector(f(x)[i]), SVector(3, true, 4f0))...), Val(2))..., 42)
    @test @inferred(BAT.forwarddiff_eval(f, (x...,))) == r_ref
    @test @inferred(BAT.forwarddiff_eval(f, x)) == r_ref

    @test @inferred(BAT.forwarddiff_value(BAT.forwarddiff_eval(sin, 0.5))) == sin(0.5)
    @test @inferred(ForwardDiff.partials(BAT.forwarddiff_eval(sin, 0.5))[1]) == cos(0.5)
    @test length(ForwardDiff.partials(BAT.forwarddiff_eval(sin, 0.5))) == 1

    @test @inferred(BAT.forwarddiff_vjp(0.7, BAT.forwarddiff_eval(sin, 0.5))) == (0.7 * ForwardDiff.derivative(sin, 0.5),)

    x = SVector(3, true, 4f0)
    @test @inferred(BAT.forwarddiff_vjp((ΔΩv...,), BAT.forwarddiff_eval(f, x))) == (ForwardDiff.jacobian(x -> SVector(f(x)), x)' * ΔΩv...,)
    @test @inferred(BAT.forwarddiff_vjp(ΔΩv, BAT.forwarddiff_eval(fv, x))) == (ForwardDiff.jacobian(fv, x)' * ΔΩv...,)

    x = (3, true, 4f0, 7)
    @test all(map(isapprox, Base.tail(@inferred ChainRulesCore.rrule(BAT.WithForwardDiff(f), x)[2](ΔΩ))[1], Zygote.pullback(f, x)[2](ΔΩ)[1]))
    @test all(map(isapprox, Base.tail(@inferred ChainRulesCore.rrule(BAT.WithForwardDiff(f), x...)[2](ΔΩ)), Zygote.pullback(f, x...)[2](ΔΩ)))


    #@test @inferred((x -> BAT.forwarddiff_pullback(f, x)[1])(x)) == f(x)
    #@test @inferred(BAT.forwarddiff_pullback(f, x)[2](ΔΩ)) == ForwardDiff.jacobian(f, x)' * ΔΩ

    #X = [SVector(0.1i, 0.2i, 0.3i) for i in 1:7]
    #ΔΩs = [SVector(10i, 20i) for i in 1:7]
    #@test @inferred((X -> BAT.forwarddiff_broadcast_pullback(Ref(f), X)[1])(X)) == broadcast(f, X)
    #@test @inferred(BAT.forwarddiff_broadcast_pullback(f, X)[2](ΔΩs)) == broadcast((f, x, ΔΩ) -> ForwardDiff.jacobian(f, x)' * ΔΩ, Ref(f), X, ΔΩs)
end
