# this computes the GMM estimator by SA minimization, for 
# each of the 1000 data sets. Can be parallelized using 
# threads.
include("DSGEmoments.jl")  # computes errors
include("parameters.jl") # load true parameter values
dsgedata = readdlm("simdata.design")
# estimate by simulated annealing
lb = lb_param_ub[:,1]
ub = lb_param_ub[:,3]
results = zeros(1000,21)
#Threads.@threads for i = 1:1000
for i = 1:1000
    data = dsgedata[i,:]
    data = reshape(data, 160, 5)
    # define GMM criterion
    moments = theta -> DSGEmoments(theta, data)
    m = theta -> vec(mean(moments(theta),1)) # 1Xg
    momentcontrib = theta -> moments(theta) # nXg
    weight = theta -> inv(cov(momentcontrib(theta)))
    obj = theta -> m(theta)'*weight(theta)*m(theta)
    thetastart = (ub+lb)/2.0 # prior mean as start
    # simulated annealing
    thetahat, objvalue, converged, details = samin(obj, thetastart, lb, ub; ns = 20, verbosity = 1, rt = 0.9)
    #ms = moments(thetahat)
    #dstats(ms)
    #prettyprint(cor(ms))
    # CUE
    thetahat2, objvalue2 = fmincon(obj, thetahat, [], [], lb, ub);
    results[i,:] = [thetahat; objvalue; details[end,1]; thetahat2; objvalue2]
    dstats(results[1,i,:])
end
writedlm("SA_results.out", results)

