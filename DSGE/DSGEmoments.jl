# computes the errors as outlined in the notes. Use these to generate
# moment conditions for estimation
function DSGEmoments(thetas, data)
        # break out variables
        y = data[:,1]
        c = data[:,2]
        n = data[:,3]
        r = data[:,4]
        w = data[:,5]
        # break out params    
        alpha = thetas[1]
        beta = thetas[2]
        delta = thetas[3]
        gam = thetas[4]
        rho_z = thetas[5]
        sig_z = thetas[6]
        rho_eta = thetas[7]
        sig_eta = thetas[8]
        nss = thetas[9]
        # recover psi
        c1 = ((1.0/beta + delta - 1.0)/alpha)^(1.0/(1.0-alpha))
        kss = nss/c1
        iss = delta*kss
        yss = kss^alpha * nss^(1-alpha)
        css = yss - iss
        psi =  (css^(-gam)) * (1-alpha) * (kss^alpha) * (nss^(-alpha))
        # use MPL-MRS
        e = log.(w) - gam*log.(c) -log.(psi)
        u = e - rho_eta*lag(e,1)
        e1 = u.^2.0 - sig_eta^2.0
        e2 = u.*lag(u,1)
        shock1 = u
        # now the Euler eqn
        e3 = (1 + r - delta).*beta.*(c.^(-gam)) - lag(c,1).^(-gam) 
        # get K from MPK/MPL eqn: the following is not real capital, it is capital computed
        # assuming the trial values of the parameters
        lagk = (alpha/(1.0-alpha))*lag(n.*w./r,1)
        # production function
        e = log.(y) - alpha*log.(lagk) - (1.0-alpha)*log.(n)
        u = e - rho_z*lag(e,1)
        e4 = u.^2.0 - sig_z^2.0
        e5 = u.*lag(u,1)
        shock2 = u
        # MPL
        e = log.(w) + alpha*(log.(n)-log.(lagk)) - log.(1.0-alpha)
        u = e - rho_z*lag(e,1)
        e6 = u.^2.0 - sig_z^2.0
        e7 = u.*lag(u,1)
        shock3 = u
        # law of motion k: good for delta
        invest = y - c
        e8 = lag(invest,1) + (1 - delta)*lag(lagk,1) - lagk
        # note: crossing all variables with shocks gives highly correlated moments, so use only
        # one variable with each shock. Also, variables crossed with tech shock are highly correlated
        # with the Euler error,so don't use them.
        # also, shock2 and shock 3 are very highly correlated for all parameter values, so use
        # the difference, to get levels right
        errors = [e1 e2 e3 e6 e7 e8 shock1.*shock3 lag(y,1).*shock1 lag(y,1).*shock2]
        errors = errors[3:end,:] # need to drop 2, because lagk uses a lag, and we use lagged k
        return errors
end

