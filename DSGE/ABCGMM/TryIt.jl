include("AISGMM.jl") # the adaptive importance sampling algorithm
include(Pkg.dir()"/MPI/examples/montecarlo.jl")
include("DSGEmoments.jl")
include("DSGEmodel.jl")
using Distributions

function wrapper()
    dsgedata = readdlm("simdata.design") # the data for the 1000 reps
    data = dsgedata[2,:]
    data = reshape(data, 160, 5)
    # features of estimation procedure
    nParticles = 10000 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept, e.g., 5 means keep 20%
    # generate data
    contrib = AIS_fit(nParticles, multiples, data, true)
    return contrib
end
contrib = wrapper()
prettyprint(contrib')

