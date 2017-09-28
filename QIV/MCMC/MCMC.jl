# uniform random walk, with bounds check
function proposal(current, tuning)
    ok = false
    trial = similar(current)
    while ok != true
        trial = current + tuning.*randn(2)
        ok, junk, junk = check_in_support(trial)
    end    
    return trial
end

function likelihood(theta, y, x, z, tau, cholsig, W)
    ghat = aux_stat(theta, y, x, z, tau)
    n = size(x,1)
    lnL = (-0.5*n*ghat*W*ghat')[1,1]
    ghat += randn(size(ghat))*cholsig/sqrt(n) # used for nonparametric fitting
    L = exp(lnL)
    return L, ghat
end

function makechain(reps, burnin, tuning, y, x, z, tau, cholsig, W)
    theta = [quantile(Normal(), tau), 2.0] # start chain at true posterior mean, so long burnin should not bee needed
    Lcurrent, ghat = likelihood(theta, y, x, z, tau, cholsig, W)
    chain = zeros(size(theta,1)+ 5,reps)
    for rep = 1:reps+burnin
        trial = proposal(theta, tuning)
        Ltrial, ghat_trial = likelihood(trial, y, x, z, tau, cholsig, W)
        accept = 0
        # MH accept/reject (simple with uniform prior and symmetric proposal)
        if rand() < Ltrial/Lcurrent 
            theta = trial
            Lcurrent = Ltrial
            accept = 1
            ghat = ghat_trial
        end
        if rep > burnin
            chain[:, rep-burnin] = [theta; ghat'; accept]
        end    
    end
    return chain
end

function summarize(chain)
    dstats(chain')
    println("90% CI, param 1: [", round(quantile(chain[1,:],0.05),4)," , ", round(quantile(chain[1,:],0.95),4), "]")
    println("90% CI, param 2: [", round(quantile(chain[2,:],0.05),4)," , ", round(quantile(chain[2,:],0.95),4), "]")
    # report posterior mean, 5% and 95% quantiles 
end


