# the moments
function aux_stat(theta::Array{Float64,1}, data::Array{Float64,2})
    ms = DSGEmoments(theta, data)
    n = size(data,1)
    sigma = cov(ms)
    siginv = inv(sigma)
    m = mean(ms,1)
    distance = n*(m*siginv*m')[1,1]
    m += randn(size(m))*chol(sigma)/sqrt(n)
    return m, distance
end

# this function generates a draw from the prior
function sample_from_prior()
lb_param_ub = [
0.20   	0.33   	0.4;	# alpha
0.95    0.99   	0.9999; 	    # beta
0.01    0.025   0.1;    # delta
0.0	    2      	5;		# gam
0    	0.9   	0.9999;	    # rho1
0.0001       0.02 	0.1;    # sigma1
0    	0.7     0.9999;      # rho2
0.0001	    0.01  	0.1;    # sigma2
6/24    8/24	9/24	# nss
]
    lb = lb_param_ub[:,1]
    ub = lb_param_ub[:,3]
	theta = rand(size(lb))
    theta = (ub-lb).*theta + lb
end

function check_in_support(theta::Array{Float64,1})
lb_param_ub = [
0.20   	0.33   	0.4;	# alpha
0.95    0.99   	0.9999; 	    # beta
0.01    0.025   0.1;    # delta
0.0	    2      	5;		# gam
0    	0.9   	0.9999;	    # rho1
0.0001       0.02 	0.1;    # sigma1
0    	0.7     0.9999;      # rho2
0.0001	    0.01  	0.1;    # sigma2
6/24    8/24	9/24	# nss
]    
    lb = lb_param_ub[:,1]
    ub = lb_param_ub[:,3]
    ok = all((theta .>= lb) .& (theta .<= ub))
    return ok, lb, ub
end

