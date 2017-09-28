#= AISGMM.jl
These are the generic functions ABC-GMM using adaptive
importance sampling. To use this, you must also supply
the model-specific code, which must define the functions

Z = aux_stat(theta, otherargs): computes aux. stats, given parameter
theta = sample_from_prior(): generate a draw from prior
ok = check_in_supporttheta): bool checking if param in prior's support
=#

using Distributions, Distances, Econometrics
include("npreg.jl")
# sample from particles: normal perturbation of a single parameter
# with bounds enforcement by rejection sampling
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

# the fitting routine: nonparametric regression
function AIS_fit(Zn::Array{Float64,2}, nParticles::Int64, multiples::Int64, y, x, z, tau, cholsig, siginv, verbose, bwLC=-999.0, bwLL=-999.0)
    # do AIS to get particles
    thetas, Zs = AIS_algorithm(nParticles, multiples, Zn, y, x, z, tau, cholsig, siginv, verbose)
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
            cvscores = LeaveOneOutCV(y, Zs, nb, lb, ub, 500)
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
function GetInitialParticles(initialParticles::Int64, y, x, z, tau, cholsig)
    particle = sample_from_prior()
    Z = aux_stat(particle, y, x ,z, tau, cholsig)
    Zs = zeros(initialParticles, size(Z,2))
    particles = zeros(initialParticles, size(particle,1))
    particles[1,:] = particle
    Zs[1,:] = Z
    Threads.@threads for i = 2 :initialParticles
        particles[i,:] = sample_from_prior()
        Zs[i,:] = aux_stat(particles[i,:], y, x, z, tau, cholsig) # initial drawn using regressors, not instruments
    end
    return particles, Zs
end

# Sample from current particles
function GetNewParticles(particles::Array{Float64,2}, draws::Int64, y, x, z, tau, cholsig)
    dimTheta = size(particles,2)
    newparticles = zeros(draws,dimTheta)
    particle = sample_from_prior()
    Z = aux_stat(particle, y, x, z, tau, cholsig)
    dimZ = size(Z,2)
    newZs = zeros(draws, dimZ)
    delta = vec(std(particles,1))
    Threads.@threads for i = 1:draws
        newparticles[i,:] = sample_from_particles(particles, delta)
        newZs[i,:] = aux_stat(newparticles[i,:], y, x ,z, tau, cholsig)
    end
    return newparticles, newZs
end

# Select the best particles from current and new
function Select(nParticles, distances::Array{Float64,1}, particles::Array{Float64,2}, Zs::Array{Float64,2}, verbose)
    ind = sortperm(distances) # indices of distances
    ind = ind[1:nParticles] # indices of best
    # keep the best distances, particles, Zs
    distances = distances[ind]
    particles = particles[ind,:]
    Zs = Zs[ind,:]
    if verbose
        println("dstats particles")
        dstats(particles)
        println("dstats statistics")
        dstats(Zs) # add a verbosity option
    end    
    return particles, Zs, distances
end    

function AIS_algorithm(nParticles::Int64, multiple::Int64, Zn, y, x, z, tau, cholsig, siginv, verbose=false)
    # organization
    dimZ = size(Zn,2)
    n = size(y,1)
    sig = sqrt.(diag(inv(siginv)))
    # the initial particles
    particles, Zs = GetInitialParticles(multiple*nParticles, y, x ,z, tau, cholsig)
    distances = vec(pairwise(SqMahalanobis(siginv),Zs', Zn')) # get all distances
    particles, Zs, distances =  Select(nParticles, distances, particles, Zs, verbose)
    iterate = true
    iters = 0
    # keep selecting until distances approx. chi-square
    tolerance = 0.2
    maxiters = 20
    while iterate
        # check if we should iterate
        continue1 = any(mean((abs.(sqrt(n)*Zs./sig') .> 1.65),1) .> tolerance) 
        continue2 = mean(ccdf(Chisq(dimZ),n*distances) .< 0.1) > tolerance
        continue3 = (iters < maxiters)
        iterate =  (continue1 || continue2) & continue3 
        if iterate # generate new particles
            newparticles, newZs = GetNewParticles(particles, nParticles*multiple, y, x, z, tau, cholsig)
            newdistances = vec(pairwise(SqMahalanobis(siginv),newZs', Zn')) # get all distances
            particles = [particles; newparticles]
            Zs = [Zs; newZs]
            distances = [distances; newdistances]
            particles, Zs, distances =  Select(nParticles, distances, particles, Zs, verbose)
        end
        if !continue3
            println("hit iter limit")
        end
        iters += 1
    end
    return particles, Zs
end

