#=
This script allows you to play around with GMM
esimation of the simple DSGE model. You can
experiment with different moments and different
start values. This should convince you that 
simply getting the GMM estimates can be quite 
difficult.

SA plus start at true values give optimized
function = 0, when exactly identified, so
there is a solution
=#

dsgedata = readdlm("simdata.design")
i = rand(1:1000)
dsgedata = dsgedata[i,:]
dsgedata = reshape(dsgedata, 160, 5)
dstats(dsgedata)
include("DSGEmoments.jl")  # computes errors
include("parameters.jl") # load true parameter values
theta0 = lb_param_ub[:,2] 

# define GMM criterion
moments = theta -> DSGEmoments(theta, dsgedata)
# average moments
m = theta -> vec(mean(moments(theta),1)) # 1Xg
# moment contributions
momentcontrib = theta -> moments(theta) # nXg
# weight
weight = theta -> inv(NeweyWest(momentcontrib(theta)))
# objective: CUE GMM criterion
obj = theta -> m(theta)'*weight(theta)*m(theta)
# estimate by simulated annealing
lb = lb_param_ub[:,1]
ub = lb_param_ub[:,3]
#thetastart = theta0    # true values as start
thetastart = (ub+lb)/2.0 # prior mean as start
# simulated annealing
thetahat, objvalue, converged, details = samin(obj, thetastart, lb, ub; ns = 20, verbosity = 2, rt = 0.75)
# attempt box constrained interior point method to refine
#thetahat, objvalue, converged = fmincon(obj, thetahat, [],[], lb, ub)
#println(thetahat)
println("the average moments at the estimate (should be zeros when exactly identified):")
dstats(moments(thetahat));

