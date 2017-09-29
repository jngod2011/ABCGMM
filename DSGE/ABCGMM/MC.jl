include("AISGMM.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")
include("DSGEmodel.jl")
using Distributions

function wrapper()
    dsgedata = readdlm("simdata.design") # the data for the 1000 reps
    data = dsgedata[rep,:]
    data = reshape(data, 160, 5)
    # features of estimation procedure
    nParticles = 10000 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept, e.g., 5 means keep 20%
    # generate data
    contrib = AIS_fit(nParticles, multiples, data, false)
    return contrib
end

# the monitoring function
function QIVMonitor(sofar, results)
    if mod(sofar,1) == 0
        # local constant
        m = mean(results[1:sofar,[1;5]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[1;5]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt.(mse)
        println()
        println("local constant results")
        println("reps so far: ", sofar)
        println("true: ", theta)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)
        # local linear
        m = mean(results[1:sofar,[9;13]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[9;13]],1) 
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
        # IV  
        m = mean(results[1:sofar,[17;18]],1)
        er = m - theta
        b = mean(er,1)
        s = std(results[1:sofar,[1;2]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt.(mse)
        println()
        println("IV results")
        println("reps so far: ", sofar)
        println("true: ", theta)
        println("mean: ", m)
        println("bias: ", b)
        println("st. dev.: ", s)
        println("mse.: ",mse)
        println("rmse.: ",rmse)
        println() 
        println("local constant CI coverage")
        inci = (results[1:sofar,3] .< theta[1]) .& (results[1:sofar,4] .>= theta[1])
        println("beta1 CI coverage: ", mean(inci))
        inci = (results[1:sofar,7] .< theta[2]) .& (results[1:sofar,8] .>= theta[2])
        println("beta2 CI coverage: ", mean(inci))
        println()
        println("local linear CI coverage")
        inci = (results[1:sofar,11] .< theta[1]) .& (results[1:sofar,12] .>= theta[1])
        println("beta1 CI coverage: ", mean(inci))
        inci = (results[1:sofar,15] .< theta[2]) .& (results[1:sofar,16] .>= theta[2])
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
    n_returns = 18
    pooled = 1  # do this many reps before reporting
    montecarlo(QIVWrapper, QIVMonitor, MPI.COMM_WORLD, reps, n_returns, pooled)
    MPI.Finalize()
end


main()
