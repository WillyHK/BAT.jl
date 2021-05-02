# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, StaticArrays
using ForwardDiff, ChainRulesCore, Zygote

@testset "forwarddiff_utils" begin
    function crc_fwd_and_back(f, xs, ΔΩ)
        y, back = ChainRulesCore.rrule(f, xs...)
        back_thunks = back(ΔΩ)
        Δx = map(unthunk, back_thunks)
        y, Δx
    end

    function zg_fwd_and_back(f, xs, ΔΩ)
        y, back = Zygote.pullback(f, xs...)
        Δx = back(ΔΩ)
        y, Δx
    end


    let f = x -> map(a -> a^2, x), x = SVector(3,4,5,6), ΔΩ = SVector(10,20,30,40)
        @test @inferred(BAT.forwarddiff_back(SVector, ΔΩ, @inferred BAT.forwarddiff_fwd(f, (x,), Val(1)))) == ForwardDiff.jacobian(f, x)' * ΔΩ
        @test @inferred(BAT.forwarddiff_fwd_back(f, (x,), Val(1), ΔΩ)) == ForwardDiff.jacobian(f, x)' * ΔΩ
    end


    sum_pow2(x) = sum(map(x -> x^2, x))
    sum_pow3(x) = sum(map(x -> x^3, x))
    f = (xs...) -> (sum(map(sum_pow2, xs)), sum(map(sum_pow3, xs)), 42)
    xs = (2f0, (3, 4f0), SVector(5f0, 6f0, 7f0))
    ΔΩ = (10, 20, 30)

    fv = let f = f; (xs...) -> SVector(f(xs...)); end
    ΔΩv = SVector(ΔΩ)


    @test @inferred(BAT.forwarddiff_fwd_back(f, xs, Val(1), ΔΩ)) == 280
    @test @inferred(BAT.forwarddiff_fwd_back(f, xs, Val(2), ΔΩ)) == (600, 1040)
    @test @inferred(BAT.forwarddiff_fwd_back(f, xs, Val(3), ΔΩ)) == SVector(1600, 2280, 3080)

    @test @inferred(ChainRulesCore.rrule(fwddiff(f), xs...)) isa Tuple{Tuple, Function}
    @test @inferred((ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ)) isa Tuple{ChainRulesCore.Zero, BAT.FwdDiffPullbackThunk, BAT.FwdDiffPullbackThunk,BAT.FwdDiffPullbackThunk}
    @test @inferred(map(unthunk, (ChainRulesCore.rrule(fwddiff(f), xs...)[2])(ΔΩ))) == (Zero(), 280, (600, 1040), SVector(1600, 2280, 3080))

    @test @inferred(crc_fwd_and_back(fwddiff(f), xs, ΔΩ)) isa Tuple{Tuple{Float32, Float32, Int64}, Tuple{Zero, Float32, Tuple{Float32, Float32}, SVector{3, Float32}}}
    @test @inferred(zg_fwd_and_back(fwddiff(f), xs, ΔΩ)) isa Tuple{Tuple{Float32, Float32, Int64}, Tuple{Float32, Tuple{Float32, Float32}, SVector{3, Float32}}}

    @test crc_fwd_and_back(fwddiff(f), xs, ΔΩ) == ((139, 783, 42), (Zero(), 280, (600, 1040), SVector(1600, 2280, 3080)))
    @test zg_fwd_and_back(fwddiff(f), xs, ΔΩ) == ((139, 783, 42), (280, (600, 1040), SVector(1600, 2280, 3080))) # == zg_fwd_and_back(f, xs, ΔΩ)
    

    function f_loss_1(xs...)
        r = fwddiff(f)(xs...)
        @assert sum(r) < 10000
        sum(r[1])
    end
    # @inferred(Zygote.gradient(f_loss_1, xs...))
    Zygote.gradient(f_loss_1, xs...)

    function f_loss_3(xs...)
        r = fwddiff(f)(xs...)
        @assert sum(r) < 10000
        sum(r[3])
    end
    Zygote.gradient(f_loss_3, xs...)

    function f_loss_3z(xs...)
        r = f(xs...)
        @assert sum(r) < 10000
        sum(r[3])
    end
    Zygote.gradient(f_loss_3z, 1,2,3)
end
