# local constant or local linear regression and quantile regression
# this computes cond. mean and median, and 0.05 and 0.95 quantiles
using Econometrics
include(Pkg.dir()*"/QuantileRegression.jl/src/InteriorPoint.jl")

function LeaveOneOutCV(y::Array{Float64,1}, X::Array{Float64,2}, nb::Int64, lb::Float64, ub::Float64, ntrials::Int64)
    n = size(X,1)
    errorsLCmean = zeros(ntrials,nb)
    errorsLC05 = zeros(ntrials,nb)
    errorsLC95 = zeros(ntrials,nb)
    errorsLLmean = zeros(ntrials,nb)
    errorsLL05 = zeros(ntrials,nb)
    errorsLL95 = zeros(ntrials,nb)
    delta = (ub-lb)/(nb-1)
    bws = zeros(nb,1)
    Threads.@threads for i = 1:ntrials
        if ntrials == size(X,1) # just loop over sample if trials equal to sample size
            ii = i
        else
            ii = rand(1:n) # otherwise, take ntrials random draws
        end
        XX = X[1:n .!=ii, :]
        yy = y[1:n .!=ii]
        x0 = X[[ii], :]
        for j = 0:nb-1
            bw = lb + delta*j
            bws[j+1] = bw
            m =LocalPolynomial(yy, XX, x0, bw, 0, false, false)
            #m, junk, q05, q95 =LocalPolynomial(yy, XX, x0, bw, 0, true, true)
            errorsLCmean[i,j+1] = y[ii] - m
            #errorsLC05[i,j+1] = 0.05 - (y[ii] < q05)
            #errorsLC95[i,j+1] = 0.95 - (y[ii] < q95)
            #m, junk, q05, q95 = LocalPolynomial(yy, XX, x0, bw, 1, true, true)
            m = LocalPolynomial(yy, XX, x0, bw, 1, false, false)
            errorsLLmean[i,j+1] = y[ii] - m
            #errorsLL05[i,j+1] = 0.05 - (y[ii] < q05)
            #errorsLL95[i,j+1] = 0.95 - (y[ii] < q95)
         end
    end
    cvsLCmean = vec(sqrt.(mean(errorsLCmean.^2.0,1)))
    #cvsLC05 = vec(sqrt(mean((errorsLC05).^2.0,1)))
    #cvsLC95 = vec(sqrt(mean((errorsLC95).^2.0,1)))
    cvsLLmean = vec(sqrt.(mean(errorsLLmean.^2.0,1)))
    #cvsLL05 = vec(sqrt(mean((errorsLL05).^2.0,1)))
    #cvsLL95 = vec(sqrt(mean((errorsLL95).^2.0,1)))
    #return [bws cvsLCmean cvsLC05 cvsLC9 cvsLLmean cvsLL05 cvsLL95]
    #return [bws cvsLCmean cvsLC05 cvsLC95 cvsLLmean]
    return [bws cvsLCmean cvsLLmean]
end    

function LocalPolynomial(y::Array{Float64,1}, X::Array{Float64,2}, x0::Array{Float64,2}, bandwidth, order::Int64=1, do_median::Bool=false, do_ci::Bool=false)
    ymean = -999
    y50 = -999
    y05 = -999
    y95 = -999
    # compute kernel weights
    X = (X .- x0) /bandwidth
    weights = prod(exp.(-(X.^2.0)/2.0),2)
    weights = weights ./sum(weights,1)
    weights = weights[:,1]
    # drop those with no weight
    test = (weights .> 0.0)[:,1]
    if sum(test) > 0
        y = y[test]
        X = X[test,:]
        weights = sqrt.(weights[test])
        if order == 0
            n = size(y,1)
            X = reshape(weights, n, 1)
            y .*= weights
            ymean = sum(weights.*y)
        end    
        if order == 1
            weights = sqrt.(weights)
            X = [ones(size(X,1),1) X]
            X .*= weights
            y .*= weights
            a = X\y
            ymean = (X\y)[1]
        end    
        if order == 2
            weights = sqrt.(weights)
            X = [ones(size(X,1),1) X X.^2.0]
            X .*= weights
            y .*= weights
            ymean = (X\y)[1]
        end  
        if do_median
            y50 = qreg_coef(vec(y), X, 0.5)[1,:]
        end
        if do_ci
            y05 = qreg_coef(vec(y), X, 0.05)[1,:]
            y95 = qreg_coef(vec(y), X, 0.95)[1,:]
        end
    end
    if ~do_median && ~do_ci
        return ymean
    elseif do_median && ~do_ci
        return [ymean y50]
    else    
        return [ymean y50 y05 y95]
    end    

end
