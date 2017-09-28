# this does plan MCMC of the same design as is used for ABCGMM
# do export JULIA_NUM_THREADS=X from shell before running this

using Distributions
include("QIVmodel.jl")
include("MCMC.jl")

function main()
    #pyplot()
    beta = [0.0, 2.0]
    mcmcreps = 10000
    reps = 1000
    burnin = 1 # not needed, as we start from true posterior mean
    chain = 0.0 # initialize outside loop
    for n in [100, 400, 1600]
        for tau in [0.1, 0.5, 0.9]
            # the following keep acceptance rate around 0.4
            if tau == 0.1 && n == 100 tuning = 0.4 end
            if tau == 0.1 && n == 400 tuning = 0.1 end
            if tau == 0.1 && n == 1600 tuning = 0.05 end
            if tau == 0.5 && n == 100 tuning = 0.2 end
            if tau == 0.5 && n == 400 tuning = 0.08 end
            if tau == 0.5 && n == 1600 tuning = 0.04 end
            if tau == 0.9 && n == 100 tuning = 0.4 end
            if tau == 0.9 && n == 400 tuning = 0.10 end
            if tau == 0.9 && n == 1600 tuning = 0.05 end
            true1 = quantile(Normal(),tau)
            true2 = 2.0
            theta = [true1; true2]
            results = zeros(1000,4)
            Threads.@threads for rep = 1:1000
                y,x,z,cholsig,W = makeQIVdata(beta, tau, n)
                chain = makechain(mcmcreps, burnin, tuning, y, x, z, tau, cholsig, W)
                # plain MCMC fit
                posmean = mean(chain[1:2,:],2)
                lower = quantile(chain[1,:],0.05)
                upper = quantile(chain[1,:],0.95)
                inci1 = true1 >= lower && true1 <= upper
                lower = quantile(chain[2,:],0.05)
                upper = quantile(chain[2,:],0.95)
                inci2 = true2 >= lower && true2 <= upper
                results[rep,:] = [posmean; inci1; inci2]
            end
            # print results
            println()
            println("n: ", n, "   tau: ", tau)
            m = vec(mean(results[:,1:2],1))
            b = vec(m) - theta
            s = vec(std(results[:,1:2],1)) 
            mse = s.^2 + b.^2
            rmse = sqrt.(mse)
            println("MCMC results")
            output = [m b s rmse]
            prettyprint(output, ["mean", "bias", "st. err.", "rmse"], ["constant", "slope"])
            println()
            println("CI coverage")
            println("beta1 CI coverage: ", mean(results[:,3]))
            println("beta2 CI coverage: ", mean(results[:,4]))
        end    
    end
end
main();
