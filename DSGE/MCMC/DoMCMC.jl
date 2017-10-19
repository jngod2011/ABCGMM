# do export JULIA_NUM_THREADS=X from shell before running this
include("MCMC.jl")
include("DSGEmoments.jl")
function main()
    dsgedata = readdlm("simdata.design") # the data for the 1000 reps
    ChainLength = 50000
    reps = 1000
    reps = 1
    burnin = 1000000
    chain = 0.0 # initialize outside loop
    tuning = [0.002, 0.002, 0.001, 0.2, 0.05, 0.005, 0.1, 0.005, 0.02] # fix this somehow
    results = zeros(reps,18) # nparams X 2: pos mean and inci, for each param
    include("parameters.jl")
    lb = lb_param_ub[:,1]
    ub = lb_param_ub[:,3]
    truetheta = lb_param_ub[:,2]
    #Threads.@threads for rep = 1:reps
    for rep = 1:reps
        # get the data for the rep (drawn from design at true param. values)
        data = dsgedata[2,:]
        data = reshape(data, 160, 5)
        # fix this next for this DGP
        initialTheta = (ub + lb) / 2.0 # prior mean to start
        #initialTheta = truetheta
        chain = makechain(initialTheta, ChainLength, burnin, tuning, data)
        
        # plain MCMC fit
        posmean = mean(chain[1:9,:],2)
        #=
        # do this next in a loop over params
        lower = quantile(chain[1,:],0.05)
        upper = quantile(chain[1,:],0.95)
        inci1 = true1 >= lower && true1 <= upper
        lower = quantile(chain[2,:],0.05)
        upper = quantile(chain[2,:],0.95)
        inci2 = true2 >= lower && true2 <= upper
        results[rep,:] = [posmean; inci1; inci2]
        =#
    end
    # print results
    #=
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
    =#
    return chain'
end
main();
