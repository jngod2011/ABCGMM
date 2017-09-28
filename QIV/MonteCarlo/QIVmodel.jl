using Distributions

# the dgp. tau quantile of epsilon is 0,
# so tau quantile of y is x*beta
function makeQIVdata(beta::Array{Float64,1}, tau::Float64, n::Int64)
    alpha = [0.0, 0.5, 0.5, -1.0]
    z = [ones(n,1) randn(n,3)]
    P = chol([1.0 0.5; 0.5 1.0])
    ev = randn(n,2)*P
    epsilon = ev[:,1]
    V = ev[:,2]
    D = z*alpha + V
    x = [ones(n,1) D]
    y = x*beta + epsilon
    sig = (tau*(1.0 -tau)*(z'z/n))
    siginv = inv(sig)
    cholsig = chol(sig)
    xhat = z*(z\x)
    yhat = z*(z\y)
    betahatIV = inv(x'*xhat)*x'*yhat
    return y,x,z,cholsig,siginv,betahatIV
end

# the moments
function aux_stat(beta::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,2},
    z::Array{Float64,2}, tau::Float64, cholsig::Array{Float64,2})
    n,k = size(x)
    m = mean(z.*(tau .- map(Float64, y .<= x*beta)),1)
    m += randn(size(m))*cholsig/sqrt(n)
    return m
end

# this function generates a draw from the prior
function sample_from_prior()
	theta = rand(2)
    lb = [-6.0, 0.0]
    ub = [6.0, 4.0]
    theta = (ub-lb).*theta + lb
end

function check_in_support(theta::Array{Float64,1})
    lb = [-6.0, 0.0]
    ub = [6.0, 4.0]
    ok = all((theta .>= lb) .& (theta .<= ub))
    return ok, lb, ub
end

