# uniform random walk, with bounds check
function proposal(current, tuning)
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
    trial = copy(current)
    i = rand(1:size(current,1))
    tt = 0.0
    #for i = 1:size(current,1)
        ok = false
        while ok != true
            tt = current[i] + tuning[i].*randn()
            ok = (tt > lb[i]) && (tt < ub[i])
        end
        trial[i] = tt
    #end   
    return trial
end

function likelihood(theta, data)
    gs = DSGEmoments(theta, data)
    n = size(data,1)
    sigma = cov(gs)
    #siginv = inv(diagm(diag(sigma)))
    siginv = inv(sigma)
    ghat = mean(gs,1)
    distance = n*(ghat*siginv*ghat')[1,1]
    lnL = -0.5*distance
    return lnL
end

function makechain(theta, reps, burnin, tuning, data)
    Lcurrent = likelihood(theta, data)
    chain = zeros(size(theta,1) + 1,reps)
    for rep = 1:reps+burnin
        trial = proposal(theta, tuning)
        Ltrial = likelihood(trial, data)
        accept = 0
        # MH accept/reject (simple with uniform prior and symmetric proposal)
        if rand() < exp(Ltrial-Lcurrent) 
            theta = trial
            Lcurrent = Ltrial
            accept = 1
        end
        if rep > burnin
            chain[:, rep-burnin] = [theta; accept]
        end    
    end
    return chain
end

function summarize(chain)
    dstats(chain')
    # report posterior mean, 5% and 95% quantiles 
end


