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
        X = lag(e,1)
        u = e-X*rho_eta
        e1 = X.*u 
        e2 = u.^2.0 - sig_eta^2.0
        shock1 = copy(u)
        # now the Euler eqn
        e3 = (1 + r - delta).*beta.*(c.^(-gam)) - lag(c,1).^(-gam) 
        # get K from MPK/MPL eqn: the following is not real capital, it is capital computed
        # assuming the trial values of the parameters
        lagk = (alpha/(1.0-alpha))*lag(n.*w./r,1)
        # production function
        e = log.(y) - alpha*log.(lagk) - (1.0-alpha)*log.(n)
        X = lag(e,1)
        u = e-X*rho_z
        e4 = X.*u 
        e5 = u.^2.0 - sig_z^2.0
        shock2 = copy(u)
        # MPL
        e = log.(w) + alpha*(log.(n)-log.(lagk)) - log.(1.0-alpha)
        X = lag(e,1)
        u = e-X*rho_z
        e6 = X.*u 
        e7 = u.^2.0 - sig_z^2.0
        shock3 = copy(u)
        # law of motion k: good for delta
        invest = y - c
        e8 = lag(invest,1) + (1 - delta)*lag(lagk,1) - lagk
        # shock2 and shock 3 are two alternative ways of recovering the technology shock.
        # They are very highly correlated for all parameter values, though their levels may
        # be different for some parameter values. Thus, use only the difference.
        # Also, don't use e4 and e5, as they are essentially copies of e6 and e7.
        errors = [e1 e2 e3 e6 e7 e8 shock1.*shock3 shock2-shock3 lag(data,1).*shock1 lag(data,1).*shock2]
        errors = errors[3:end,:] # need to drop 2, because lagk uses a lag, and we use lagged k
        return errors
end

