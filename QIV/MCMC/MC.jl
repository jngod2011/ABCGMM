using Distributions
include("QIVmodel.jl") # load auction model code
include("MCMC.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")
include npreg.jl

global const tau = 0.5

function DoFit(chain)
    thetas = chain(1:2,:)'
    Zs = chain(3:6,:)'
    params = size(thetas,2)
    Zs = Zs./std(Zs,1)
    ll_fit = -999. *ones(1,4*params)
    for i = 1:params
        y = thetas[:,i]
        lb = 0.05
        ub = 1.5
        nb = 15
        cvscores = LeaveOneOutCV(y, Zs, nb, lb, ub, 500)
        bwLL = cvscores[indmin(cvscores[:,3]),1]
        #println("parameter ", i, " bwLL: ", bwLL)
        #println("bwLL95: ",bwLL95)
        Zn = zeros(1,size(Zs,2))
        ll_fit[:,(i*4-4+1):i*4] = LocalPolynomial(thetas[:,i], Zs, Zn, bwLL, 1, true, true)
    end
    return ll_fit
end    

function QIVWrapper()
    # features of model
    n = 100  # sample size
    beta = [0.0, 2.0] # true parameters
    # features of estimation procedure
    reps = 10000 # particles to keep per iter
    burnin = 1 # not needed, starting at true posterior mean
    # the following keep acceptance rate around 0.4
    if tau == 0.1 && n == 100  tuning = 0.4  end
    if tau == 0.1 && n == 400  tuning = 0.1  end
    if tau == 0.1 && n == 1600 tuning = 0.05 end
    if tau == 0.5 && n == 100  tuning = 0.2  end
    if tau == 0.5 && n == 400  tuning = 0.08 end
    if tau == 0.5 && n == 1600 tuning = 0.04 end
    if tau == 0.9 && n == 100  tuning = 0.4  end
    if tau == 0.9 && n == 400  tuning = 0.10 end
    if tau == 0.9 && n == 1600 tuning = 0.05 end
    # generate data
    y,x,z,cholsig,W = makeQIVdata(beta, tau, n) # draw the data
    chain = makechain(reps, burnin, tuning, y, x, z, tau, cholsig, W)
    contrib = DoFit(chain)
end

# the monitoring function
function QIVMonitor(sofar, results)
    if mod(sofar,1) == 0
        offset = [quantile(Normal(),tau) 0.0]
        theta = [0.0 2.0] + offset
        # local constant
        m = mean(results[1:sofar,[1;5]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[1;5]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt.(mse)
        println()
        # local linear fit
        m = mean(results[1:sofar,[1;5]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[1;5]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt.(mse)
        println()
        println("local linear results")
        println("reps so far: ", sofar)
        println("true: ", theta)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)
        println()
        println("CI coverage")
        inci = (results[1:sofar,3] .< theta[1]) .& (results[1:sofar,4] .>= theta[1])
        println("beta1 CI coverage: ", mean(inci))
        inci = (results[1:sofar,7] .< theta[2]) .& (results[1:sofar,8] .>= theta[2])
        println("beta2 CI coverage: ", mean(inci))
    end

    # save the LL results for local bandwidth tuning
    if sofar == size(results,1)
#        writedlm("first_round_estimates.out", results[:,[9;13]])
    end
end

function main()
    MPI.Init()
    reps = 1000   # desired number of MC reps
    n_returns = 8
    pooled = 1  # do this many reps before reporting
    montecarlo(QIVWrapper, QIVMonitor, MPI.COMM_WORLD, reps, n_returns, pooled)
    MPI.Finalize()
end

main()
