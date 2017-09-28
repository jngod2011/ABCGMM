# uniform random walk, with bounds check
function proposal(current, tuning)
    include("parameters.jl")
    lb = lb_param_ub[:,1]
    ub = lb_param_ub[:,3]
    trial = similar(current)
    for i = 1:size(current,1)
        tt = 0.0
        ok = false
        while ok != true
            tt = current[i] + tuning[i].*randn()
            ok = (tt > lb[i]) && (tt < ub[i])
        end
        trial[i] = tt
    end   
    return trial
end

function likelihood(theta, data)
    g = DSGEmoments(theta, data)
    #W = inv(NeweyWest(g))
    #W = inv(diagm(diag(NeweyWest(g))))
    W = inv(diagm(diag(cov(g))))
    #W = inv(cov(g))
    #W = eye(size(g,2))
    ghat = mean(g,1)
    n = size(data,1)
    lnL = (-0.5*n*ghat*W*ghat')[1,1]
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


