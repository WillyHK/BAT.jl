# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const RealOrZero = Union{Real, ChainRulesCore.AbstractZero}


function _fu_replace_nth(f::Base.Callable, x::Tuple, ::Val{i}) where i
    ntuple(j -> i == j ? f(x[j]) : x[j], Val(length(x)))
end


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = ForwardDiff.Dual{TagType}(x, true)

function forwarddiff_dualized(::Type{TagType}, x::NTuple{N,Real}) where {TagType,N}
    ntuple(j -> ForwardDiff.Dual{TagType}(x[j], ntuple(i -> i == j, Val(N))), Val(N))
end

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::SVector) where {TagType} = SVector(forwarddiff_dualized(TagType, (x...,)))


@inline function forwarddiff_fwd(f::Base.Callable, xs::Tuple, ::Val{i}) where i
    # TagType = ... # Not type stable if TagType declared outside of _fu_replace_nth:
    dualized_args = _fu_replace_nth(x_i -> forwarddiff_dualized(typeof(ForwardDiff.Tag((f,Val(i)), eltype(xs[i]))), x_i), xs, Val(i))
    f(dualized_args...)
end


@inline forwarddiff_value(y_dual::Real) = ForwardDiff.value(y_dual)
@inline forwarddiff_value(y_dual::NTuple{N,Real}) where N = map(ForwardDiff.value, y_dual)
@inline forwarddiff_value(y_dual::SVector{N,<:Real}) where N = SVector(map(ForwardDiff.value, y_dual))


@inline forwarddiff_back_unshaped(ΔΩ::RealOrZero, y_dual::Real) = (ΔΩ * ForwardDiff.partials(y_dual)...,)

function forwarddiff_back_unshaped(ΔΩ::NTuple{N,RealOrZero}, y_dual::NTuple{N,Real}) where N
    (sum(map((ΔΩ_i, y_dual_i) -> ForwardDiff.partials(y_dual_i) * ΔΩ_i, ΔΩ, y_dual))...,)
end

@inline function forwarddiff_back_unshaped(ΔΩ::ChainRulesCore.Composite{<:Any,<:NTuple{N,RealOrZero}}, y_dual::NTuple{N,Real}) where N
    forwarddiff_back_unshaped((ΔΩ...,), y_dual)
end

@inline function forwarddiff_back_unshaped(ΔΩ::SVector{N,<:Real}, y_dual::SVector{N,<:Real}) where N
    forwarddiff_back_unshaped((ΔΩ...,), (y_dual...,))
end


@inline shape_forwarddiff_gradient(::Type{<:Real}, Δx::Tuple{}) = nothing
@inline shape_forwarddiff_gradient(::Type{<:Real}, Δx::Tuple{Real}) = Δx[1]
@inline shape_forwarddiff_gradient(::Type{<:Tuple}, Δx::NTuple{N,Real}) where N = Δx
@inline shape_forwarddiff_gradient(::Type{<:SVector}, Δx::Tuple{}) = nothing
@inline shape_forwarddiff_gradient(::Type{<:SVector}, Δx::NTuple{N,Real}) where N = SVector(Δx)


# For `x::SVector`, `BAT.forwarddiff_back(SVector, ΔΩ, BAT.forwarddiff_fwd(f, (x,), Val(1))) == ForwardDiff.jacobian(f, x)' * ΔΩ`:
@inline forwarddiff_back(::Type{T}, ΔΩ, y_dual) where T = shape_forwarddiff_gradient(T, forwarddiff_back_unshaped(ΔΩ, y_dual))

# For `x::SVector`, `forwarddiff_fwd_back(f, (x,), Val(1), ΔΩ) == ForwardDiff.jacobian(f, x)' * ΔΩ`:
@inline function forwarddiff_fwd_back(f::Base.Callable, xs::Tuple, ::Val{i}, ΔΩ) where i
    # @info "RUN forwarddiff_fwd_back(f, xs, Val($i), ΔΩ)"
    x_i = xs[i]
    y_dual = forwarddiff_fwd(f, xs, Val(i))
    forwarddiff_back(typeof(x_i), ΔΩ, y_dual)
end


struct FwdDiffDualBack{TX,TY} <: Function
    y_dual::TY
end

function (back::FwdDiffDualBack{TX})(ΔΩ) where TX
    # @info "RUN BACK (back::FwdDiffDualBack{TX})(ΔΩ)"
    (ChainRulesCore.NO_FIELDS, forwarddiff_back(TX, ΔΩ, back.y_dual)...)
end


function forwarddiff_pullback(f::Base.Callable, xs::Vararg{Any,N}) where N

    y_dual = forwarddiff_fwd(f, xs...)
    y = forwarddiff_value(y_dual)
    y, FwdDiffDualBack{eltype(xs), typeof(y_dual)}(y_dual)
end


struct WithForwardDiff{F} <: Function
    f::F
end

@inline (wrapped_f::WithForwardDiff)(xs...) = wrapped_f.f(xs...)



struct FwdDiffPullbackThunk{F<:Base.Callable,T<:Tuple,i,U<:Any} <: ChainRulesCore.AbstractThunk
    f::F
    xs::T
    ΔΩ::U
end

function FwdDiffPullbackThunk(f::F, xs::T, ::Val{i}, ΔΩ::U) where {F<:Base.Callable,T<:Tuple,i,U<:Any}
    FwdDiffPullbackThunk{F,T,i,U}(f, xs, ΔΩ)
end

@inline ChainRulesCore.unthunk(tnk::FwdDiffPullbackThunk{F,T,i,U}) where {F,T,i,U} = forwarddiff_fwd_back(tnk.f, tnk.xs, Val(i), tnk.ΔΩ)

(tnk::FwdDiffPullbackThunk)() = ChainRulesCore.unthunk(tnk)


Base.@generated function _forwarddiff_pullback_thunks(f::Base.Callable, xs::NTuple{N,Any}, ΔΩ::Any) where N
    Expr(:tuple, ChainRulesCore.NO_FIELDS, (:(BAT.FwdDiffPullbackThunk(f, xs, Val($i), ΔΩ)) for i in 1:N)...)
end

@inline function ChainRulesCore.rrule(wrapped_f::WithForwardDiff, xs::Vararg{Any,N}) where N
    # @info "RUN ChainRulesCore.rrule(wrapped_f::WithForwardDiff, xs) with N = $N"
    f = wrapped_f.f
    y = f(xs...)
    with_fwddiff_pullback(ΔΩ) = _forwarddiff_pullback_thunks(f, xs, ΔΩ)
    return y, with_fwddiff_pullback
end


#=
struct DualizedFunction{F<:Base.Callable,i} <: Function
    f::F
end

DualizedFunction(f::F, ::Val{i}) where {F<:Base.Callable,i} = DualizedFunction(F,i)(f)

@inline (dualized_f::DualizedFunction{F,i})(x...) where {F,i} = forwarddiff_fwd(dualized_f.f, xs, Val(i))
=#


# ToDo: ChainRulesCore.rrule(::typeof(Base.broadcasted), wrapped_f::WithForwardDiff, xs...)


#=

function rrule(
    ::typeof(*),
    A::AbstractVecOrMat{<:CommutativeMulNumber},
    B::AbstractVecOrMat{<:CommutativeMulNumber},
)
    function times_pullback(Ȳ)
        return (
            NO_FIELDS,
            InplaceableThunk(
                @thunk(Ȳ * B'),
                X̄ -> mul!(X̄, Ȳ, B', true, true)
            ),
            InplaceableThunk(
                @thunk(A' * Ȳ),
                X̄ -> mul!(X̄, A', Ȳ, true, true)
            )
        )
    end
    return A * B, times_pullback
end


function rrule(
    ::typeof(*), A::CommutativeMulNumber, B::AbstractArray{<:CommutativeMulNumber}
 )
     function times_pullback(Ȳ)
         return (
             NO_FIELDS,
             @thunk(dot(Ȳ, B)'),
             InplaceableThunk(
                 @thunk(A' * Ȳ),
                 X̄ -> mul!(X̄, conj(A), Ȳ, true, true)
             )
         )
     end
     return A * B, times_pullback
 end
 
=#
