# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _fu_replace_nth(f::Base.Callable, x::Tuple, ::Val{i}) where i
    ntuple(j -> i == j ? f(x[j]) : x[j], Val(length(x)))
end


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = ForwardDiff.Dual{TagType}(x, true)

function forwarddiff_dualized(::Type{TagType}, x::NTuple{N,Real}) where {TagType,N}
    ntuple(j -> ForwardDiff.Dual{TagType}(x[j], ntuple(i -> i == j, Val(N))), Val(N))
end

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::SVector) where {TagType} = SVector(forwarddiff_dualized(TagType, (x...,)))


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

_fd_scalar(x::Tuple{<:Real}) = x[1]

@inline forwarddiff_vjp(::Type{<:Real}, ΔΩ, y_dual) = _fd_scalar(forwarddiff_vjp(ΔΩ, y_dual))
@inline forwarddiff_vjp(::Type{<:Tuple}, ΔΩ, y_dual) = forwarddiff_vjp(ΔΩ, y_dual)
@inline forwarddiff_vjp(::Type{<:SVector}, ΔΩ, y_dual) = SVector(forwarddiff_vjp(ΔΩ, y_dual))


@inline function partial_forwarddiff_eval(f::Base.Callable, xs::Tuple, ::Val{i}) where i
    TagType = typeof(ForwardDiff.Tag((f,Val(i)), eltype(xs[i])))
    dualized_args = _fu_replace_nth(x_i -> forwarddiff_dualized(TagType, x_i), xs, Val(i))
    f(dualized_args...)
end

@inline function partial_forwarddiff_fwd_back(f::Base.Callable, xs::Tuple, ::Val{i}, ΔΩ) where i
    @info "RUN partial_forwarddiff_fwd_back(f, xs, $i, ΔΩ)"
    x_i = xs[i]
    y_dual = partial_forwarddiff_eval(f, xs, Val(i))
    forwarddiff_vjp(typeof(x_i), ΔΩ, y_dual)
end


struct FwdDiffDualBack{TX,TY} <: Function
    y_dual::TY
end

function (back::FwdDiffDualBack{TX})(ΔΩ) where TX
    # @info "RUN BACK (back::FwdDiffDualBack{TX})(ΔΩ)"
    (ChainRulesCore.NO_FIELDS, forwarddiff_vjp(TX, ΔΩ, back.y_dual)...)
end


function forwarddiff_pullback(f::Base.Callable, xs::Vararg{Any,N}) where N

    y_dual = forwarddiff_eval(f, xs...)
    y = forwarddiff_value(y_dual)
    y, FwdDiffDualBack{eltype(xs), typeof(y_dual)}(y_dual)
end


struct WithForwardDiff{F} <: Function
    f::F
end

@inline (wrapped_f::WithForwardDiff)(xs...) = wrapped_f.f(xs...)

@inline function ChainRulesCore.rrule(wrapped_f::WithForwardDiff, xs::Vararg{Any,N}) where N
    @info "RUN ChainRulesCore.rrule(wrapped_f::WithForwardDiff, xs)"
    f = wrapped_f.f
    y = f(xs...)
    function with_fwddiff_pullback(ΔΩ)
        return (
            NO_FIELDS,
            ntuple(i -> ChainRulesCore.@thunk(partial_forwarddiff_fwd_back(f, xs, Val(i), ΔΩ)), Val(N))...
        )
    end
end


#=
struct DualizedFunction{F<:Base.Callable,i} <: Function
    f::F
end

DualizedFunction(f::F, ::Val{i}) where {F<:Base.Callable,i} = DualizedFunction(F,i)(f)

@inline (dualized_f::DualizedFunction{F,i})(x...) where {F,i} = partial_forwarddiff_eval(dualized_f.f, xs, Val(i))
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
