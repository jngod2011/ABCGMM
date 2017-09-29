# computes the errors as outlined in the notes. Use these to generate
# moment conditions for estimation
function DSGEmoments(thetas, realdata)
        # break out variables
        y = realdata[:,1]
        c = realdata[:,2]
        n = realdata[:,3]
        r = realdata[:,4]
        w = realdata[:,5]
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
        # use MUL-MRS
        e = log.(w) - gam*log.(c) -log.(psi)
        shock1 = e
        e1 = (e - rho_eta*lag(e,1))/sig_eta
        e2 = e1.^2.0 - 1.0
        e1 = e.*lag(e,1) - rho_eta/(1.0-rho_eta^2.0)*sig_eta^2.0
        # now the Euler eqn
        #cc = c ./lag(c,1)
        e3 = (beta*c.^(-gam).*(1 + r -delta))-lag(c,1).^(-gam)
        # production function
        # the following is not real capital, it is capital computed
        # assuming the trial values of the parameters
        k = alpha*lag(n,1).*lag(w,1)./lag(r,1)/(1.0-alpha)
        e = log.(y) - alpha*log.(k) - (1.0-alpha)*log.(n)
        e4 = (e - rho_z*lag(e,1))/sig_z
        e5 = e4.^2.0 - 1.0
        e4 = e.*lag(e,1) - rho_z/(1.0-rho_z^2.0)*sig_z^2.0
        shock2 = e
        # MPL
        e = log.(w) + alpha*(log.(n)-log.(k)) - log.(1.0-alpha)
        shock3 = e
        e6 = (e - rho_z*lag(e,1))/sig_z
        e7 = e6.^2.0 - 1.0
        e6 = e.*lag(e,1) - rho_z/(1.0-rho_z^2.0)*sig_z^2.0
        # law of motion k: good for delta
        invest = y - c
        e8 = lag(invest,1) + (1 - delta)*lag(k,1) - k
        errors = [e1 e2/100.0 e3 e4 e5/100.0 e6 e7/100.0 e8 shock1.*shock2 shock1.*shock3]
        errors = errors[3:end,:] # need to drop 2, because k uses a lag, and we use lagged k
        return errors
end

