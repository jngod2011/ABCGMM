#= 
loads the final particles and stats from the AK data,
does CV bandwidth selection, and does nonparametric
regression and NP quantile regression to get the ABC-GMM
estimates
=#
using Econometrics
include("npreg.jl")
results = zeros(16,9)

data = readdlm("card.data")
# dep var
LNW = data[:,1]
EDUC = data[:,2]
EXPER = data[:,3] 
BLACK = data[:,4]
SMSA = data[:,5]
SOUTH = data[:,6]
NEARC2 = data[:,7]
NEARC4 = data[:,8]
EXPSQ = (EXPER.^2.0)/100.0
constant = ones(size(LNW))
X = [constant EDUC EXPER EXPSQ BLACK SMSA SOUTH]
X = [1.0 14.0 mean(EXPER) (mean(EXPER).^2.0)/100.0 0.0 1.0 0.0]

function main()
    for j = 1:9
        println("tau: ", round(0.1*j,2))
        tau = round(0.1*j,2)
        file = "thetaZ."*sprint(show,tau)
        thetaZ = readdlm(file)
        thetas = thetaZ[:,1:7]
        educ = thetaZ[:,2]
        black = thetaZ[:,5]
        twoyearscollege = vec(X*thetas')
        outputs = [educ twoyearscollege black] # will compute quantiles for these outputs
        Zs = thetaZ[:,8:end]
        Zs = Zs./std(Zs,1)
        ll_fit = -999. *ones(4*size(outputs,2))
        do_cv = true
        for i = 1:size(outputs,2)
            lb = 0.1
            ub = 2.0
            nb = 15
            cvscores = LeaveOneOutCV(outputs[:,i], Zs, nb, lb, ub, 500)
            bwLL = cvscores[indmin(cvscores[:,3]),1]
            println("parameter ", i, " bwLL: ", bwLL)
            ll_fit[(i*4-4+1):i*4] = LocalPolynomial(outputs[:,i], Zs, zeros(1,size(Zs,2)), bwLL, 1, true, true)
        end
        outputs = (outputs .- ll_fit[[1,5,9]]').^2.0
        squares = zeros(3)
        for i = 1:size(outputs,2)
            lb = 0.1
            ub = 2.0
            nb = 15
            cvscores = LeaveOneOutCV(outputs[:,i], Zs, nb, lb, ub, 500)
            bwLL = cvscores[indmin(cvscores[:,3]),1]
            println("parameter ", i, " bwLL: ", bwLL)
            squares[i] = LocalPolynomial(outputs[:,i], Zs, zeros(1,size(Zs,2)), bwLL, 1, false, false)
        end
        results[:,j] = [tau; ll_fit; sqrt.(squares)]
        prettyprint(results)
    end
    writedlm("EstimationResults.out", results)
end
main()
