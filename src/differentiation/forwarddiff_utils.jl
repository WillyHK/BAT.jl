# This file is a part of BAT.jl, licensed under the MIT License (MIT).


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = ForwardDiff.Dual{TagType}(x, one(x))

function forwarddiff_dualized(::Type{TagType}, x::Vararg{Real,N}) where {TagType,N}
    ntuple(j -> ForwardDiff.Dual{TagType}(x[j], ntuple(i -> i == j, Val(N))), Val(N))
end

forwarddiff_dualized(::Type{TagType}, x::Tuple) where {TagType} = forwarddiff_dualized(TagType, x...)

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::SVector) where {TagType} = SVector(forwarddiff_dualized(TagType, x...))


@inline forwarddiff_tagtype(f::Base.Callable, xs...) = typeof(ForwardDiff.Tag(f, promote_type(map(eltype, xs)...)))


@inline forwarddiff_eval(f::Base.Callable, xs::Real...) = 
    f(forwarddiff_dualized(forwarddiff_tagtype(f, xs), xs...)...)

@inline forwarddiff_eval(f::Base.Callable, x::NTuple{N,Real}) where N =
    f(forwarddiff_dualized(forwarddiff_tagtype(f, x), x))

# Equivalent to ForwardDiff internal function static_dual_eval(TagType, f, x) (for SVector):
@inline forwarddiff_eval(f::Base.Callable, x::SVector{N,<:Real}) where N =
    f(forwarddiff_dualized(forwarddiff_tagtype(f, x), x))


@inline forwarddiff_value(y_dual::Real) = ForwardDiff.value(y_dual)
@inline forwarddiff_value(y_dual::NTuple{N,Real}) where N = map(ForwardDiff.value, y_dual)
@inline forwarddiff_value(y_dual::SVector{N,<:Real}) where N = SVector(map(ForwardDiff.value, y_dual))


@inline forwarddiff_vjp(ΔΩ::Real, y_dual::Real) = (ΔΩ * ForwardDiff.partials(y_dual)...,)

function forwarddiff_vjp(ΔΩ::NTuple{N,Real}, y_dual::NTuple{N,Real}) where N
    (sum(map((ΔΩ_i, y_dual_i) -> ForwardDiff.partials(y_dual_i) * ΔΩ_i, ΔΩ, y_dual))...,)
end

@inline function forwarddiff_vjp(ΔΩ::ChainRulesCore.Composite{<:Any,<:NTuple{N,Real}}, y_dual::NTuple{N,Real}) where N
    forwarddiff_vjp((ΔΩ...,), y_dual)
end

# BAT.forwarddiff_vjp(ΔΩ, BAT.forwarddiff_eval(f, x)) == (ForwardDiff.jacobian(f, x)' * ΔΩ (for SVector)...,):
@inline function forwarddiff_vjp(ΔΩ::SVector{N,<:Real}, y_dual::SVector{N,<:Real}) where N
    forwarddiff_vjp((ΔΩ...,), (y_dual...,))
end

@inline forwarddiff_vjp(::Type{<:Real}, ΔΩ, y_dual) = forwarddiff_vjp(ΔΩ, y_dual)
@inline forwarddiff_vjp(::Type{<:Tuple}, ΔΩ, y_dual) = (forwarddiff_vjp(ΔΩ, y_dual),)
@inline forwarddiff_vjp(::Type{<:SVector}, ΔΩ, y_dual) = (SVector(forwarddiff_vjp(ΔΩ, y_dual)),)


struct FwdDiffDualBack{TX,TY} <: Function
    y_dual::TY
end

function (back::FwdDiffDualBack{TX})(ΔΩ) where TX
    # @info "RUN BACK" typeof(ΔΩ)
    (ChainRulesCore.NO_FIELDS, forwarddiff_vjp(TX, ΔΩ, back.y_dual)...)
end

function forwarddiff_pullback(f::Base.Callable, xs...)
    # @info "RUN forwarddiff_pullback"
    y_dual = forwarddiff_eval(f, xs...)
    y = forwarddiff_value(y_dual)
    y, FwdDiffDualBack{eltype(xs), typeof(y_dual)}(y_dual)
end


struct WithForwardDiff{F} <: Function
    f::F
end

@inline (wrapped_f::WithForwardDiff)(xs...) = wrapped_f.f(xs...)

@inline ChainRulesCore.rrule(wrapped_f::WithForwardDiff{F}, xs...) where F = forwarddiff_pullback(wrapped_f.f, xs...)



struct DualizedFunction{F} <: Function
    f::F
end

@inline (dualized_f::DualizedFunction)(x...) = forwarddiff_eval(dualized_f.f, x...)


# ToDo: ChainRulesCore.rrule(::typeof(Base.broadcasted), wrapped_f::WithForwardDiff, xs...)
