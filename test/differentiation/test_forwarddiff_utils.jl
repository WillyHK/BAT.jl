# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using StaticArrays, ForwardDiff

@testset "forwarddiff_utils" begin
    f = (x...) -> (sum(map(x -> norm(x)^2, x)), sum(map(x -> norm(x)^3, x)))
    fv = (x...) -> SVector(sum(map(x -> norm(x)^2, x)), sum(map(x -> norm(x)^3, x)))

    DTf32 = ForwardDiff.Dual{ForwardDiff.Tag{typeof(f),Float32}}
    @test @inferred(BAT.forwarddiff_eval(f, 4f0)) == (DTf32(16.0,8.0), DTf32(64.0,48.0))
    rdf = (DTf32(26.0,6.0,2.0,8.0), DTf32(92.0,27.0,3.0,48.0))
    @test @inferred(BAT.forwarddiff_eval(f, 3.0, true, 4f0)) == (DTf32(26.0,6.0,2.0,8.0), DTf32(92.0,27.0,3.0,48.0))
    @test @inferred(BAT.forwarddiff_eval(fv, 3.0, true, 4f0)) == SVector(DTf32(26.0,6.0,2.0,8.0), DTf32(92.0,27.0,3.0,48.0))

    x = SVector(3.0, true, 4f0)
    r_ref = ntuple(i -> DTf32(f(x)[i], ForwardDiff.jacobian(x -> SVector(f(x)[i]), SVector(3.0, true, 4f0))...), Val(2))
    @test @inferred(BAT.forwarddiff_eval(f, (x...,))) == r_ref
    @test @inferred(BAT.forwarddiff_eval(f, x)) == r_ref

    @test @inferred(BAT.forwarddiff_value(BAT.forwarddiff_eval(sin, 0.5))) == sin(0.5)
    @test @inferred(ForwardDiff.partials(BAT.forwarddiff_eval(sin, 0.5))[1]) == cos(0.5)
    @test length(ForwardDiff.partials(BAT.forwarddiff_eval(sin, 0.5))) == 1

    @test @inferred(BAT.forwarddiff_vjp(0.7, BAT.forwarddiff_eval(sin, 0.5))) == (0.7 * ForwardDiff.derivative(sin, 0.5),)

    ΔΩ = SVector(10, 20)
    @test @inferred(BAT.forwarddiff_vjp((ΔΩ...,), BAT.forwarddiff_eval(f, x))) == (ForwardDiff.jacobian(x -> SVector(f(x)), x)' * ΔΩ...,)
    @test @inferred(BAT.forwarddiff_vjp(ΔΩ, BAT.forwarddiff_eval(fv, x))) == (ForwardDiff.jacobian(fv, x)' * ΔΩ...,)

    @test @inferred((x -> BAT.forwarddiff_pullback(f, x)[1])(x)) == f(x)
    @test @inferred(BAT.forwarddiff_pullback(f, x)[2](ΔΩ)) == ForwardDiff.jacobian(f, x)' * ΔΩ

    X = [SVector(0.1i, 0.2i, 0.3i) for i in 1:7]
    ΔΩs = [SVector(10i, 20i) for i in 1:7]
    @test @inferred((X -> BAT.forwarddiff_broadcast_pullback(Ref(f), X)[1])(X)) == broadcast(f, X)
    @test @inferred(BAT.forwarddiff_broadcast_pullback(f, X)[2](ΔΩs)) == broadcast((f, x, ΔΩ) -> ForwardDiff.jacobian(f, x)' * ΔΩ, Ref(f), X, ΔΩs)
end
