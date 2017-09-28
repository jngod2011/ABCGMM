#= AIS.jl
These are the generic functions for BII or ABC-GMM using adaptive
importance sampling. To use this, you must also supply
the model-specific code, which must define the functions

Z = aux_stat(theta, otherargs): computes aux. stats, given parameter
d = prior(theta): computes prior density at parameter value
theta = sample_from_prior(): generate a draw from prior
ok = check_in_supporttheta): bool checking if param in prior's support
=#

using Distributions, Distances, Econometrics
include("npreg.jl")
# sample from particles: normal perturbation of a 
# single parameter, with bounds enforcement by rejection
# sampling
function sample_from_particles(particles::Array{Float64,2}, delta::Array{Float64,1})
	n, k = size(particles)
    i = rand(1:n)
	j = rand(1:k)
    ok = false
    theta_s = similar(particles[i,:])
    @inbounds while ok != true
        theta_s = particles[i,:]
        theta_s[j:j] += delta[j]*randn(1)
        ok, junk, junk = check_in_support(theta_s)
    end    
    return theta_s
end

# sample from AIS density: choose random particle,
# add a smallish MVN perturbation
function sample_from_AIS(S::Int64, particles::Array{Float64,2})
    delta = std(particles,1)
	inds = rand(1:size(particles,1),S)
    theta_s = particles[inds,:] .+ delta.*randn(S, size(particles,2))
    return theta_s
end

# the importance sampling density: mixture of normals
function AIS_density(theta::Array{Float64,2}, particles::Array{Float64,2})
    # what scaling to use here?
    delta  = std(particles,1)
    sig = diagm(vec(delta.^2))
    nthetas = size(theta,1)
    nparticles = size(particles,1)
    dens = zeros(nthetas,1)
    @inbounds for i = 1:nparticles
        @inbounds for j = 1:nthetas
            thetaj = theta[j,:]
            mu = particles[i,:]
            d = MvNormal(mu, sig)
            dens[j,1] += pdf(d, thetaj)
        end
    end
    dens = dens/nparticles
    return dens
end    

function AIS_sample_mix(particles, AISdraws, mix, otherargs)
    S = Int64(round(mix*AISdraws))
    thetas1 = sample_from_AIS(S, particles)
    S2 = AISdraws - S
    thetas2 = zeros(S2, size(particles,2))
    for i = 1:S2
        thetas2[i,:] = sample_from_prior()
    end
    thetas = [thetas1; thetas2]
    junk = aux_stat(thetas[1,:], otherargs)
    Zs = zeros(AISdraws, size(junk,2))
    for i = 1:AISdraws
        Zs[i,:] = aux_stat(thetas[i,:], otherargs)
    end
    return thetas, Zs
end    
 

# the fitting routine: nonparametric regression
# possibly weighted by importance sampling
# weights from the AIS density

function AIS_fit(Zn::Array{Float64,2}, nParticles::Int64, multiples::Int64, iters::Int64, AISdraws::Int64, mix::Float64, otherargs, bwLC=-999.0, bwLL=-999.0)
    # do AIS to get particles
    thetas, Zs = AIS_algorithm(nParticles, multiples, iters, Zn, otherargs)
    # sample from AIS particles
    thetas, Zs = AIS_sample_mix(thetas, AISdraws, mix, otherargs)
    params = size(thetas,2)
    Zs = Zs./std(Zs,1)
    lc_fit = -999. *ones(1,4*params)
    ll_fit = -999. *ones(1,4*params)
    if bwLC == -999.0
        do_cv = true
    else
        do_cv = false
    end    
    for i = 1:params
        # do cross validation?
        if do_cv
            y = thetas[:,i]
            lb = 0.05
            ub = 1.5
            nb = 15
            cvscores = LeaveOneOutCV(y, Zs, nb, lb, ub, 250)
            #println("parameter: ", i)
            #prettyprint(cvscores)
            bwLC = cvscores[indmin(cvscores[:,2]),1]
            #bwLC05 = cvscores[indmin(cvscores[:,3]),1]
            #bwLC95 = cvscores[indmin(cvscores[:,4]),1]
            bwLL = cvscores[indmin(cvscores[:,3]),1]
            #bwLL05 = cvscores[indmin(cvscores[:,6]),1]
            #bwLL95 = cvscores[indmin(cvscores[:,7]),1]
        end
        #println("parameter ", i, " bwLC: ", bwLC, " bwLL: ", bwLL)
        #println("bwLC05: ",bwLC05)
        #println("bwLC95: ",bwLC95)
        #println("bwLL05: ",bwLL05)
        #println("bwLL95: ",bwLL95)
        lc_fit[:,(i*4-4+1):i*4] = LocalPolynomial(thetas[:,i], Zs, Zn, bwLC, 0, true, true)
        #a = LocalPolynomial(thetas[:,i], Zs, Zn, bwLC, 0, false, false)
        #b = LocalPolynomial(thetas[:,i], Zs, Zn, 3.0*bwLC, 0, true, true)
        #lc_fit[:,(i*4-4+1):i*4] = [a[1,1]; a[1,1]; b[1,3:4]]
        ll_fit[:,(i*4-4+1):i*4] = LocalPolynomial(thetas[:,i], Zs, Zn, bwLL, 1, true, true)
        #a = LocalPolynomial(thetas[:,i], Zs, Zn, bwLL, 1, false, false)
        #b = LocalPolynomial(thetas[:,i], Zs, Zn, 3.0*bwLL, 1, true, true)
        #ll_fit[:,(i*4-4+1):i*4] = [a[1,1]; a[1,1]; b[1,3:4]]
    end
    return [lc_fit ll_fit]
end    

# Draw particles from prior
function GetInitialParticles(initialParticles::Int64, otherargs)
    particle = sample_from_prior()
    Z = aux_stat(particle, otherargs)
    Zs = zeros(initialParticles, size(Z,2))
    particles = zeros(initialParticles, size(particle,1))
    particles[1,:] = particle
    Zs[1,:] = Z
    Threads.@threads for i = 2 :initialParticles
        particles[i,:] = sample_from_prior()
        Zs[i,:] = aux_stat(particles[i,:], otherargs)
    end
    return particles, Zs
end

# Sample from current particles
function GetNewParticles(particles::Array{Float64,2}, draws::Int64, otherargs)
    dimTheta = size(particles,2)
    newparticles = zeros(draws,dimTheta)
    particle = sample_from_prior()
    Z = aux_stat(particle, otherargs)
    dimZ = size(Z,2)
    newZs = zeros(draws, dimZ)
    delta = vec(std(particles,1))
    Threads.@threads for i = 1:draws
        newparticles[i,:] = sample_from_particles(particles, delta)
        newZs[i,:] = aux_stat(newparticles[i,:], otherargs)
    end
    return newparticles, newZs
end

# Select the best particles from current and new
function Select(nParticles, distances::Array{Float64,1}, particles::Array{Float64,2}, Zs::Array{Float64,2})
    ind = sortperm(distances) # indices of distances
    ind = ind[1:nParticles] # indices of best
    # keep the best distances, particles, Zs
    distances = distances[ind]
    particles = particles[ind,:]
    Zs = Zs[ind,:]
    #println()
    #dstats(particles)
    #println()
    #dstats(Zs) # add a verbosity option
    return particles, Zs, distances
end    

function AIS_algorithm(nParticles::Int64, multiple::Int64, iters::Int64, Zn, otherargs)
    # the initial particles
    particles, Zs = GetInitialParticles(multiple*nParticles, otherargs)
    # do bounding to compute scale the first time
    # in the loop, selection will remove outliers in subsequent interations
    dimZ = size(Zs,2)
    siginv = otherargs[6]::Array{Float64,2}
    distances = vec(pairwise(SqMahalanobis(siginv),Zs', Zn')) # get all distances
    particles, Zs, distances =  Select(nParticles, distances, particles, Zs)
    @inbounds for iter = 2:iters
        # generate new particles
        newparticles, newZs = GetNewParticles(particles, nParticles*multiple, otherargs)
        newdistances = vec(pairwise(SqMahalanobis(siginv),newZs', Zn')) # get all distances
        particles = [particles; newparticles]
        Zs = [Zs; newZs]
        distances = [distances; newdistances]
        particles, Zs, distances =  Select(nParticles, distances, particles, Zs)
    end
    return particles, Zs
end


