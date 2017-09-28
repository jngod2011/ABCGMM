#= note: to run this in parallel, from the system
prompt, and before starting julia, do
export JULIA_NUM_THREADS=X
where X is the desired number of threads. On
my 32 core system, 10 is good (cache contention issues)
Remember to re-set threads to 1 before using MPI
=#
using Econometrics
include("AISGMM.jl")
include("QIVmodel.jl")
LNW, X, Z = getdata()
println(size(LNW))
n = size(LNW,1)
Zn = zeros(1, size(Z,2))
for i = 1:9
    tau = round(0.1*i,2)
    sig = tau*(1.0-tau)*(Z'Z/n)
    cholsig = Array(chol(sig))
    siginv = inv(sig)
    otherargs = LNW, X, Z, tau, cholsig, siginv
    # features of estimation procedure
    nParticles = 50000 # particles to keep per iter
    multiples = 5  # particles tried is this multiple of particle kept
    # the following does ordinary QR (Z = X)
    # thetas, Zs = AIS_algorithm(nParticles, multiples, Zn, LNW, X, X, tau, cholsig, siginv, true)
    # the following does QIV (instruments instead of regressors)
    thetas, Zs = AIS_algorithm(nParticles, multiples, Zn, LNW, X, Z, tau, cholsig, siginv, true)
    println(tau)
    dstats(thetas)
    dstats(Zs)
    writedlm("thetaZ."sprint(show,tau), [thetas Zs])
end

