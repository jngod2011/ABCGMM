include("parameters.jl")
truep = lb_param_ub[:,2]
truep = truep[[4,5,7]]
prettyprint(truep)
data = readdlm("ABCGMMresults")
for param = 1:3
    est = data[:,param*4-4+1]
    lower = data[:,param*4-2+1]
    upper = data[:,param*4-1+1]
    inci = (lower .< truep[param]) .& (upper .> truep[param])
    println("ci coverage, parameter ", param, ": ", mean(inci))
end    
