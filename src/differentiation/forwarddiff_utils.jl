# This file is a part of BAT.jl, licensed under the MIT License (MIT).


forwarddiff_dualized(::Type{TagType}, x::Real) where TagType = ForwardDiff.Dual{TagType}(x, one(x))

function forwarddiff_dualized(::Type{TagType}, x::Vararg{<:Real,N}) where {TagType,N}
    ntuple(j -> ForwardDiff.Dual{TagType}(x[j], ntuple(i -> i == j, Val(N))), Val(N))
end

forwarddiff_dualized(::Type{TagType}, x::Tuple) where {TagType} = forwarddiff_dualized(TagType, x...)

# Equivalent to ForwardDiff internal function dualize(T, x):
forwarddiff_dualized(::Type{TagType}, x::SVector) where {TagType} = SVector(forwarddiff_dualized(TagType, x...))



# Equivalent to ForwardDiff internal function static_dual_eval(TagType, f, x) (for SVector):
#function forwarddiff_eval(f::Base.Callable, x::Union{T, NTuple{N,T}, SVector{N,T}}) where {N,T<:Real}
#    TagType = typeof(ForwardDiff.Tag(f, T))
#    x_dual = forwarddiff_dualized(T, x)
#    y_dual = f(x_dual)
#end


function forwarddiff_eval(f::Base.Callable, x...)
    T = promote_type(map(eltype, x)...)
    TagType = typeof(ForwardDiff.Tag(f, T))
    f(forwarddiff_dualized(TagType, x...)...)
end



forwarddiff_vjp(ΔΩ::Real, y_dual::ForwardDiff.Dual{TagType,T,1}) where {TagType,T<:Real} = ΔΩ * first(ForwardDiff.partials(y_dual))

function forwarddiff_vjp(ΔΩ::Union{NTuple{N,T},SVector{N,T}}, y_dual::NTuple{N,<:ForwardDiff.Dual}) where {N,T<:Real}
    (sum(map((ΔΩ_i, y_dual_i) -> ForwardDiff.partials(y_dual_i) * ΔΩ_i, ΔΩ, y_dual))...,)
end

# BAT.forwarddiff_vjp(ΔΩ, BAT.forwarddiff_eval(f, x)) == ForwardDiff.jacobian(f, x)' * ΔΩ (for SVector):
forwarddiff_vjp(ΔΩ::Union{NTuple{N,T},SVector{N,T}}, y_dual::SVector{N,<:ForwardDiff.Dual}) where {N,T<:Real} = SVector(forwarddiff_vjp((ΔΩ...,), (y_dual...,)))


# Return type of fwddiff_back (`SVector` or `NTuple`) currently depeds on type of `f(x)`, not type of `x`:
function forwarddiff_pullback(f::Base.Callable, x::Union{NTuple{N,T}, SVector{N,T}}) where {N,T<:Real}
    # Seems faster this way, according to benchmarking (benchmark artifact?):
    TagType = typeof(ForwardDiff.Tag(f, T))
    x_dual = forwarddiff_dualized(TagType, x)
    y_dual = f(x_dual)

    # Seems slower this way, for some reason:
    # y_dual = forwarddiff_eval(f, x)

    y = map(ForwardDiff.value, y_dual)
    fwddiff_back(ΔΩ) = forwarddiff_vjp(ΔΩ, y_dual)
    y, fwddiff_back
end


function forwarddiff_broadcast_pullback(fs, X::AbstractArray{<:Union{NTuple{N,T},SVector{N,T}}}) where {N,T<:Real}
    Y_dual = broadcast(forwarddiff_eval, fs, X)
    Y = broadcast(y_dual -> map(ForwardDiff.value, y_dual), Y_dual)
    fwddiff_back(ΔΩs) = broadcast(forwarddiff_vjp, ΔΩs, Y_dual)
    Y, fwddiff_back
end

#=
# May be faster for cheap `fs`, according to benchmarking, in spite of double evalutation (why?). Limited to SVector elements:
function forwarddiff_broadcast_pullback(fs, X::AbstractArray{<:SVector{N,T}}) where {N,T<:Real}
    Y = broadcast(fs, X)
    fwddiff_back(ΔΩs) = broadcast((f, x, ΔΩ) -> ForwardDiff.jacobian(f, x)' * ΔΩ, fs, X, ΔΩs)
    Y, fwddiff_back
end
=#

#=

# Modified version of `dual_function` in Zygote.jl:
function dual_function(f::F) where F
    function dualized_f(args::Vararg{Any,N}) where N
        ds = map(args, ntuple(identity,Val(N))) do x, i
            dual(x, ntuple(j -> i==j, Val(N)))
        end
        return f(ds...)
    end
    return dualized_f
end


fwddiff_broadcast(f, args::Vararg{Union{AbstractArray,Number},N}) where N = broadcast(f, args...)


@inline function broadcast_forward(f, args::Vararg{Union{AbstractArray,Number},N}) where N
    T = Broadcast.combine_eltypes(f, args)
    out = dual_function(f).(args...)
    eltype(out) <: Dual || return (out, _ -> nothing)
    y = map(x -> x.value, out)
    _back(ȳ, i) = unbroadcast(args[i], ((a, b) -> a*b.partials[i]).(ȳ, out))
    back(ȳ) = ntuple(i -> _back(ȳ, i), N)
    return y, back
end
=#
