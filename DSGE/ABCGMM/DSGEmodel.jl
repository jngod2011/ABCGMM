# the moments
function aux_stat(theta::Array{Float64,1}, data::Array{Float64,2})
    ms = DSGEmoments(theta, data)
    n = size(data,1)
    sigma = NeweyWest(ms)
    m = mean(ms,1)
    m += randn(size(m))*chol(sigma)/sqrt(n)
    siginv = inv(sigma)
    return m, (m*siginv*m')[1,1]
end

# this function generates a draw from the prior
function sample_from_prior()
    include("parameters.jl")
    lb = lb_param_ub[:,1]
    ub = lb_param_ub[:,3]
	theta = rand(size(lb))
    theta = (ub-lb).*theta + lb
end

function check_in_support(theta::Array{Float64,1})
    include("parameters.jl")
    lb = lb_param_ub[:,1]
    ub = lb_param_ub[:,3]
    ok = all((theta .>= lb) .& (theta .<= ub))
    return ok, lb, ub
end

