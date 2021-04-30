# This file is a part of BAT.jl, licensed under the MIT License (MIT).


#=
logval, back = Zygote.pullback(f_logval, v_unshaped)
unshaped_gradient = first(back(Zygote.sensitivity(res)))
=#
