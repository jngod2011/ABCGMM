using Econometrics

#=
 variables in card.data are
 log_wage
 nearc4
 educ
 exper
 black
 smsa
 south
=# 
function getdata()
    data = readdlm("card.data")
    # dep var
    LNW = data[:,1]
    EDUC = data[:,2]
    EXPER = data[:,3] 
    BLACK = data[:,4]
    SMSA = data[:,5]
    SOUTH = data[:,6]
    NEARC2 = data[:,7]
    NEARC4 = data[:,8]
    EXPSQ = (EXPER.^2.0)/100.0
    constant = ones(size(LNW))
    Z = [constant EXPER EXPSQ BLACK SMSA SOUTH NEARC4]
    EHAT = Z*(Z\EDUC)
    Z = [constant EHAT EXPER EXPSQ BLACK SMSA SOUTH]
    X = [constant EDUC EXPER EXPSQ BLACK SMSA SOUTH]
    return LNW, X, Z
end

# the moments
function aux_stat(beta::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,2},
    z::Array{Float64,2}, tau::Float64, cholsig::Array{Float64,2})
    n,k = size(x)
    m = mean(z.*(tau .- map(Float64, y .<= x*beta)),1)
    m += randn(size(m))*cholsig/sqrt(n)
    return m
end

# this function generates a draw from the prior, based on OLS estimates 
function sample_from_prior()
    lb = [2.0; 0.0; -0.5*ones(5)]
    ub = [7.0; 0.5;  0.5*ones(5)]
theta = (ub-lb).*rand(size(lb)) + lb
end

function check_in_support(theta::Array{Float64,1})
    lb = [2.0; 0.0; -0.5*ones(5)]
    ub = [7.0; 0.5;  0.5*ones(5)]
    ok = all((theta .>= lb) .& (theta .<= ub))
    return ok, lb, ub
end

