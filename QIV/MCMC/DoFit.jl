include("npreg.jl")
function DoFit(chain)
    thetas = chain[1:2,:]'
    Zs = chain[3:6,:]'
    params = size(thetas,2)
    Zs = Zs./std(Zs,1)
    lc_fit = -999. *ones(1,4*params)
    ll_fit = -999. *ones(1,4*params)
    for i = 1:params
        y = thetas[:,i]
        lb = 0.05
        ub = 1.5
        nb = 15
        cvscores = LeaveOneOutCV(y, Zs, nb, lb, ub, 500)
        #println("parameter: ", i)
        #prettyprint(cvscores)
        bwLC = cvscores[indmin(cvscores[:,2]),1]
        bwLL = cvscores[indmin(cvscores[:,3]),1]
        lc_fit[:,(i*4-4+1):i*4] = LocalPolynomial(thetas[:,i], Zs, zeros(1,4), bwLC, 0, true, true)
        ll_fit[:,(i*4-4+1):i*4] = LocalPolynomial(thetas[:,i], Zs, zeros(1,4), bwLL, 1, true, true)
    end
    return [lc_fit ll_fit]
end

DoFit(100, 0.5)
            # print results
            println()
            println("n: ", n, "   tau: ", tau)
            m = vec(mean(results[:,1:2],1))
            b = vec(m) - theta
            s = vec(std(results[:,1:2],1)) 
            mse = s.^2 + b.^2
            rmse = sqrt.(mse)
            println("MCMC results")
            output = [m b s rmse]
            prettyprint(output, ["mean", "bias", "st. err.", "rmse"], ["constant", "slope"])
            println()
            println("CI coverage")
            println("beta1 CI coverage: ", mean(results[:,3]))
            println("beta2 CI coverage: ", mean(results[:,4]))
            println()

