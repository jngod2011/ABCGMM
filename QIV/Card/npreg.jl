# local constant or local linear regression and quantile regression
# this computes cond. mean and median, and 0.05 and 0.95 quantiles
using Econometrics
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
    for i = 1:ntrials
        if ntrials == n # just loop over sample if trials equal to sample size
            ii = i
        else
            ii = rand(1:n) # otherwise, take ntrials random draws
        end
        XX = X[setdiff(1:n,ii), :]
        yy = y[setdiff(1:n,ii)]
        x0 = X[[ii], :]
        Threads.@threads for j = 0:nb-1
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

function bound(x, dx)
    # Fill vector with allowed step lengths
    # Replace with -x/dx for negative dx
    b = 1e20 + 0.0 * x
    for i = 1:length(dx)
        if dx[i] < 0.0
            @inbounds b[i] = -x[i] / dx[i]
        end
    end
    return b
end

# The functions bound() and qreg_coef() are from the QuantileRegression.jl package
# copied here for future-proofness and less dependencies
function qreg_coef(Y::Vector, X::Matrix, p)
    # input: X is an n x k matrix of exogenous regressors,
    #        Y is an n x 1 vector of outcome variables
    #        p \in (0,1) is the quantile of interest
    # Output: p^{th} regression quantiles.
    # Construct the dual problem of quantile regression
    c = -Y
    m = size(X, 1)
    u = ones(m)
    x = (1 - p) .* u
    b = X'x
    # Solve a linear program by the interior point method:
    # min(c * u), s.t. A * x = b and 0 < x < u
    # An initial feasible solution has to be provided as x
    # Set some constants
    beta = 0.99995
    small = 1e-6
    max_it = 50
    n, m = size(X)
    # Generate inital feasible point
    s = u - x
    y = -X\Y
    r = c - X*y
    BLAS.axpy!(0.001, (r .== 0.0).*1.0, r)
    z = r .* (r .> 0.0)
    w = z - r
    gap = Base.LinAlg.BLAS.dot(c, x) - Base.LinAlg.BLAS.dot(y, b) + Base.LinAlg.BLAS.dot(w, u)
    # Start iterations
    it = 0
    for it = 1:max_it
        #   Compute affine step
        q = 1 ./ (z ./ x + w ./ s)
        r = z - w
        Q = Diagonal(sqrt.(q)) # Very efficient to do since Q diagonal
        AQtF = qrfact(Q*X, Val{true}) 
        #AQtF = qrfact!(Q*X, pivot = true) # PE 2004
        rhs = Q*r        # "
        dy = AQtF\rhs   # "
        dx = q.*(X*dy - r)
        ds = -dx
        dz = -z .* (1 + dx ./ x)
        dw = -w .* (1 + ds ./ s)
        # Compute maximum allowable step lengths
        fx = bound(x, dx)
        fs = bound(s, ds)
        fw = bound(w, dw)
        fz = bound(z, dz)
        fpv = min.(fx, fs)
        fdv = min.(fw, fz)
        fp = min.(minimum(beta * fpv), 1)
        fd = min.(minimum(beta * fdv), 1)
        # If full step is feasible, take it. Otherwise modify it
        if min(fp, fd) < 1.0
            # Update mu
            mu = Base.LinAlg.BLAS.dot(z, x) + Base.LinAlg.BLAS.dot(w, s)
            g = Base.LinAlg.BLAS.dot(z + fd*dz, x + fp*dx) + Base.LinAlg.BLAS.dot(w + fd*dw, s + fp*ds)
            mu = mu * (g / mu)^3 / (2 * n)
            # Compute modified step
            dxdz = dx .* dz
            dsdw = ds .* dw
            xinv = 1 ./ x
            sinv = 1 ./ s
            xi = mu .* (xinv - sinv)
            #rhs = rhs + Q * (dxdz - dsdw - xi)
            BLAS.axpy!(1.0, Q * (dxdz - dsdw - xi), rhs) # no gemv-wrapper gemv(Q, (dxdz - dsdw - xi), rhs,1,1,n)?
            dy = AQtF\rhs
            dx = q .* (X*dy + xi - r - dxdz + dsdw)
            ds = -dx
            for i = 1:length(dz)
                dz[i] = mu * xinv[i] - z[i] - xinv[i] * z[i] * dx[i] - dxdz[i]
                dw[i] = mu * sinv[i] - w[i] - sinv[i] * w[i] * ds[i] - dsdw[i]
            end
            # Compute maximum allowable step lengths
            fx = bound(x, dx)
            fs = bound(s, ds)
            fw = bound(w, dw)
            fz = bound(z, dz)
            fp = min.(fx, fs)
            fd = min.(fw, fz)
            fp = min.(minimum(beta .* fp), 1.0)
            fd = min.(minimum(beta .* fd), 1.0)
        end
        # Take the steps
        BLAS.axpy!(fp, dx, x)
        BLAS.axpy!(fp, ds, s)
        BLAS.axpy!(fd, dy, y)
        BLAS.axpy!(fd, dw, w)
        BLAS.axpy!(fd, dz, z)
        gap = Base.LinAlg.BLAS.dot(c, x) - Base.LinAlg.BLAS.dot(y, b) + Base.LinAlg.BLAS.dot(w, u)
        if gap < small
            break
        end
    end
    return -y
end
