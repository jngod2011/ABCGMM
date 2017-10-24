# do export JULIA_NUM_THREADS=X from shell before running this
include("MCMC.jl")
include("DSGEmoments.jl")
function main()
    dsgedata = readdlm("simdata.design") # the data for the 1000 reps
    ChainLength = 50000
    reps = 100
    burnin = 1500000
    chain = 0.0 # initialize outside loop
    tuning = [0.004, 0.001, 0.001, 0.15, 0.08, 0.002, 0.1, 0.002, 0.005] # fix this somehow
    results = zeros(reps,18) # nparams X 2: pos mean and inci, for each param
    include("parameters.jl")
    lb = lb_param_ub[:,1]
    ub = lb_param_ub[:,3]
    truetheta = lb_param_ub[:,2]
    results = zeros(reps,18)
    Threads.@threads for rep = 1:reps
    #for rep = 1:reps
        # get the data for the rep (drawn from design at true param. values)
        data = dsgedata[rep,:]
        data = reshape(data, 160, 5)
        # fix this next for this DGP
        initialTheta = (ub + lb) / 2.0 # prior mean to start
        #initialTheta = truetheta
        chain = makechain(initialTheta, ChainLength, burnin, tuning, data)
        # plain MCMC fit
        posmean = mean(chain[1:9,:],2)
        inci = zeros(9)
        for i = 1:9
            lower = quantile(chain[i,:],0.05)
            upper = quantile(chain[i,:],0.95)
            inci[i] = truetheta[i] >= lower && truetheta[i] <= upper
        end
        results[rep,1:9] = posmean
        results[rep,10:18] = inci
        #println(results[rep,1:9])
        #println()
        # prettyprint(reshape(mean(results[1:rep,:],1),9,2))
    end    
    #writedlm("MCMCresults.out", results)
end
main();
