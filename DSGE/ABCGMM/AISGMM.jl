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
function AIS_fit(nParticles::Int64, multiples::Int64, data::Array{Float64,2}, verbose)
    # do AIS to get particles
    thetas, Zs = AIS_algorithm(nParticles, multiples, data, verbose)
    params = size(thetas,2)
    Zs = Zs./std(Zs,1)
    ll_fit = -999. *ones(1,4*params)
    for i = 1:params
        # cross validation
        y = thetas[:,i]
        lb = 0.05
        ub = 1.5
        nb = 15
        cvscores = LeaveOneOutCV(y, Zs, nb, lb, ub, 500)
        #println("parameter: ", i)
        #prettyprint(cvscores)
        bwLL = cvscores[indmin(cvscores[:,3]),1]
        ll_fit[:,(i*4-4+1):i*4] = LocalPolynomial(thetas[:,i], Zs, Zn, bwLL, 1, true, true)
    end
    return ll_fit
end    

# Draw particles from prior
function GetInitialParticles(initialParticles::Int64, data::Array{Float64,2})
    particle = sample_from_prior()
    Z, distance = aux_stat(particle, data)
    Zs = zeros(initialParticles, size(Z,2))
    distances = zeros(initialParticles)
    particles = zeros(initialParticles, size(particle,1))
    particles[1,:] = particle
    distances[1] = distance
    Zs[1,:] = Z
    Threads.@threads for i = 2 :initialParticles
        particles[i,:] = sample_from_prior()
        Zs[i,:], distances[i] = aux_stat(particles[i,:], data)
    end
    return particles, Zs, distances
end

# Sample from current particles
function GetNewParticles(particles::Array{Float64,2}, draws::Int64, data)
    dimTheta = size(particles,2)
    newparticles = zeros(draws,dimTheta)
    particle = sample_from_prior()
    Z, distance = aux_stat(particle, data)
    dimZ = size(Z,2)
    newZs = zeros(draws, dimZ)
    newdistances = zeros(draws)
    delta = vec(std(particles,1))
    Threads.@threads for i = 1:draws
        newparticles[i,:] = sample_from_particles(particles, delta)
        newZs[i,:], newdistances[i] = aux_stat(newparticles[i,:], data)
    end
    return newparticles, newZs, newdistances
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

function AIS_algorithm(nParticles::Int64, multiple::Int64, data, verbose=false)
    particles, Zs, distances = GetInitialParticles(multiple*nParticles, data)
    particles, Zs, distances = Select(nParticles, distances, particles, Zs, verbose)
    iterate = true
    iters = 0
    # keep selecting until distances approx. chi-square
    tolerance = 0.2
    maxiters = 20
    n = size(data,1)
    dimZ = size(Zs, 2)
    while iterate
        # check if we should iterate
        #continue1 = any(mean((abs.(sqrt(n)*Zs./sig') .> 1.65),1) .> tolerance) 
        continue1 = true
        continue2 = mean(ccdf(Chisq(dimZ),n*distances) .< 0.1) > tolerance
        continue3 = (iters < maxiters)
        iterate =  (continue1 || continue2) & continue3 
        if iterate # generate new particles
            newparticles, newZs, newdistances = GetNewParticles(particles, nParticles*multiple, data)
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

