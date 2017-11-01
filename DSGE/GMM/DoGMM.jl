# this computes the GMM estimator by SA minimization, for 
# each of the 1000 data sets. Can be parallelized using 
# threads.
include("DSGEmoments.jl")  # computes errors
include("parameters.jl") # load true parameter values
dsgedata = readdlm("simdata.design")
# estimate by simulated annealing
lb = lb_param_ub[:,1]
ub = lb_param_ub[:,3]
results = zeros(100,20)
Threads.@threads for i = 1:100
#for i = 1:100
    data = dsgedata[i,:]
    data = reshape(data, 160, 5)
    # define GMM criterion
    moments = theta -> DSGEmoments(theta, data)
    m = theta -> vec(mean(moments(theta),1)) # 1Xg
    momentcontrib = theta -> moments(theta) # nXg
    weight = theta -> inv(NeweyWest(momentcontrib(theta)))
    obj = theta -> m(theta)'*weight(theta)*m(theta)
    thetastart = (ub+lb)/2.0 # prior mean as start
    # gradient based (doesn't work)
    #thetahat, objvalue, converged = fmincon(obj, thetastart, [], [], lb, ub; iterlim=10000)
    #details = 1.0
    # simulated annealing
    thetahat, objvalue, converged, details = samin(obj, thetastart, lb, ub; ns = 20, verbosity = 1, rt = 0.9)
    D = (Calculus.jacobian(m, vec(thetahat), :central))
    W = weight(thetahat)
    V = inv(D'*W*D)/(size(data,1)-2.0)
    se = sqrt.(diag(V))
    println("results iter ", i)
    prettyprint([thetahat se])
    results[i,:] = [thetahat; se; details[end,1]; objvalue]
    #if i > 1
    #    dstats(results[1:i,:])
    #end    
end
writedlm("junk", results)

