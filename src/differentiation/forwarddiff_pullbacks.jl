# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function dual_number end
function dual_value end
function dual_partials end
function dual_tagtype end

@inline dual_number(::Type{TagType}, x::Real, p::NTuple{N,Real}) where {TagType,N} = ForwardDiff.Dual{TagType}( x, p...)
@inline dual_value(x::Real) = ForwardDiff.value(x)
@inline dual_partials(x::Real) = ForwardDiff.partials(x)
@inline dual_tagtype(f::Any, ::Type{T}) where T = typeof(ForwardDiff.Tag(f, T))


const RealOrZero = Union{Real, ChainRulesCore.AbstractZero}


function _fu_replace_nth(f::Base.Callable, x::Tuple, ::Val{i}) where i
    ntuple(j -> i == j ? f(x[j]) : x[j], Val(length(x)))
end


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = dual_number(TagType, x, (true,))

function forwarddiff_dualized(::Type{TagType}, x::NTuple{N,Real}) where {TagType,N}
    ntuple(j -> dual_number(TagType, x[j], ntuple(i -> i == j, Val(N))), Val(N))
end

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::SVector) where {TagType} = SVector(forwarddiff_dualized(TagType, (x...,)))


_fieldvals(x) = ntuple(i -> getfield(x, i), Val(fieldcount(typeof(x))))

@generated function _strip_type_parameters(tp::Type{T}) where T
    nm = T.name
    :($(nm.module).$(nm.name))
end

function forwarddiff_dualized(::Type{TagType}, x::T) where {TagType,T}
    tp = _strip_type_parameters(T)
    fieldvals = _fieldvals(x)
    dual_fieldvals = forwarddiff_dualized(TagType, fieldvals)
    tp(dual_fieldvals...)
end


@inline function forwarddiff_fwd(f::Base.Callable, xs::Tuple, ::Val{i}) where i
    # TagType = ... # Not type stable if TagType declared outside of _fu_replace_nth:
    dualized_args = _fu_replace_nth(x_i -> forwarddiff_dualized(dual_tagtype((f,Val(i)), eltype(xs[i])), x_i), xs, Val(i))
    f(dualized_args...)
end


@inline forwarddiff_value(y_dual::Real) = dual_value(y_dual)
@inline forwarddiff_value(y_dual::NTuple{N,Real}) where N = map(dual_value, y_dual)
@inline forwarddiff_value(y_dual::SVector{N,<:Real}) where N = SVector(map(dual_value, y_dual))


@inline forwarddiff_back_unshaped(ΔΩ::RealOrZero, y_dual::Real) = (ΔΩ * dual_partials(y_dual)...,)

function forwarddiff_back_unshaped(ΔΩ::NTuple{N,RealOrZero}, y_dual::NTuple{N,Real}) where N
    (sum(map((ΔΩ_i, y_dual_i) -> dual_partials(y_dual_i) * ΔΩ_i, ΔΩ, y_dual))...,)
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

@inline shape_forwarddiff_gradient(::Type{T}, Δx::Tuple{}) where T = nothing
@inline @generated function shape_forwarddiff_gradient(::Type{T}, Δx::Tuple) where T
    :(NamedTuple{$(fieldnames(T))}(Δx))
end


# For `x::SVector`, `BAT.forwarddiff_back(SVector, ΔΩ, BAT.forwarddiff_fwd(f, (x,), Val(1))) == ForwardDiff.jacobian(f, x)' * ΔΩ`:
@inline forwarddiff_back(::Type{T}, ΔΩ, y_dual) where T = shape_forwarddiff_gradient(T, forwarddiff_back_unshaped(ΔΩ, y_dual))

# For `x::SVector`, `forwarddiff_fwd_back(f, (x,), Val(1), ΔΩ) == ForwardDiff.jacobian(f, x)' * ΔΩ`:
@inline function forwarddiff_fwd_back(f::Base.Callable, xs::Tuple, ::Val{i}, ΔΩ) where i
    # @info "RUN forwarddiff_fwd_back(f, xs, Val($i), ΔΩ)"
    x_i = xs[i]
    y_dual = forwarddiff_fwd(f, xs, Val(i))
    forwarddiff_back(typeof(x_i), ΔΩ, y_dual)
end



struct FwddiffFwd{F<:Base.Callable,i} <: Function
    f::F
end
FwddiffFwd(f::F, ::Val{i}) where {F<:Base.Callable,i} = FwddiffFwd{F,i}(f)

(fwd::FwddiffFwd{F,i})(xs...) where {F,i} = forwarddiff_fwd(fwd.f, xs, Val(i))


struct FwddiffBack{TX<:Any} <: Function end

(bck::FwddiffBack{TX})(ΔΩ, y_dual) where TX = forwarddiff_back(TX, ΔΩ, y_dual)


function forwarddiff_bc_fwd_back(f::Base.Callable, Xs::Tuple, ::Val{i}, ΔΩA::Any) where i
    # @info "RUN forwarddiff_bc_fwd_back(f, Xs, Val($i), ΔΩA)"
    fwd = FwddiffFwd(f, Val(i))
    TX = eltype(Xs[i])
    bck = FwddiffBack{TX}()
    bck.(ΔΩA, fwd.(Xs...))
end



struct WithForwardDiff{F} <: Function
    f::F
end

@inline (wrapped_f::WithForwardDiff)(xs...) = wrapped_f.f(xs...)

# Desireable for consistent behavior?
# Base.broadcasted(wrapped_f::WithForwardDiff, xs...) = broadcast(wrapped_f.f, xs...)

"""
    fwddiff(f::Base.Callable)::Function

Use `ForwardDiff` in `ChainRulesCore` pullback For

* `fwddiff(f)(args...)
* `fwddiff(f).(args...)
"""
fwddiff(f::Base.Callable) = WithForwardDiff(f)
export fwddiff



struct FwdDiffPullbackThunk{F<:Base.Callable,T<:Tuple,i,U<:Any} <: ChainRulesCore.AbstractThunk
    f::F
    xs::T
    ΔΩ::U
end

function FwdDiffPullbackThunk(f::F, xs::T, ::Val{i}, ΔΩ::U) where {F<:Base.Callable,T<:Tuple,i,U<:Any}
    FwdDiffPullbackThunk{F,T,i,U}(f, xs, ΔΩ)
end

@inline function ChainRulesCore.unthunk(tnk::FwdDiffPullbackThunk{F,T,i,U}) where {F,T,i,U}
    forwarddiff_fwd_back(tnk.f, tnk.xs, Val(i), tnk.ΔΩ)
end

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



struct FwdDiffBCPullbackThunk{F<:Base.Callable,T<:Tuple,i,U} <: ChainRulesCore.AbstractThunk
    f::F
    Xs::T
    ΔΩA::U
end

function FwdDiffBCPullbackThunk(f::F, Xs::T, ::Val{i}, ΔΩA::U) where {F<:Base.Callable,T<:Tuple,i,U}
    FwdDiffBCPullbackThunk{F,T,i,U}(f, Xs, ΔΩA)
end

@inline function ChainRulesCore.unthunk(tnk::FwdDiffBCPullbackThunk{F,T,i,U}) where {F,T,i,U}
    forwarddiff_bc_fwd_back(tnk.f, tnk.Xs, Val(i), tnk.ΔΩA)
end

(tnk::FwdDiffBCPullbackThunk)() = ChainRulesCore.unthunk(tnk)



Base.@generated function _forwarddiff_bc_pullback_thunks(f::Base.Callable, Xs::NTuple{N,Any}, ΔΩA::Any) where N
    Expr(:tuple, ChainRulesCore.NO_FIELDS, (:(BAT.FwdDiffBCPullbackThunk(f, Xs, Val($i), ΔΩA)) for i in 1:N)...)
end

function ChainRulesCore.rrule(::typeof(Base.broadcasted), wrapped_f::WithForwardDiff, Xs::Vararg{Any,N}) where N
    # @info "RUN ChainRulesCore.rrule(Base.broadcasted, wrapped_f::WithForwardDiff, Xs) with N = $N"
    f = wrapped_f.f
    y = broadcast(f, Xs...)
    bc_with_fwddiff_pullback(ΔΩA) = _forwarddiff_bc_pullback_thunks(f, Xs, ΔΩA)
    return y, bc_with_fwddiff_pullback
end
