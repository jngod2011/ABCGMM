include("AISGMM.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")
include("DSGEmoments.jl")
include("DSGEmodel.jl")
using Distributions

function wrapper(rep)
    dsgedata = readdlm("simdata.design") # the data for the 1000 reps
    data = dsgedata[rep,:]
    data = reshape(data, 160, 5)
    # features of estimation procedure
    nParticles = 10000 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept, e.g., 5 means keep 20%
    # generate data
    verbose = false
    contrib = AIS_fit(nParticles, multiples, data, verbose)
    return contrib
end
results = 0
for rep = 1:100
    contrib = wrapper(rep)
    if rep == 1
        results = zeros(100,size(contrib,2))
    end
    results[rep,:] = contrib
    println("rep: ", rep)
    dstats(results[1:rep,:])
end
writedlm("ABCGMMresults", results)
