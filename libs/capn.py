import numpy as np 
import pandas as pd


# deg = param['order']
# lb = param['lowerK']
# ub = param['upperK']
# delta = param['delta']


def approxdef(deg, lb, ub, delta):
    '''
    The function defines an approximation space for all three 
    approximation apporoaches (V, P, and Pdot).

    deg:    An array of degrees of approximation function: degrees of Chebyshev polynomials
    lb:     An array of lower bounds
    ub:     An array of upper bounds
    delta:  discount rate
    '''
    if (delta[0] > 1 or delta[0] < 0):
        raise Exception("delta should be in [0,1]!")

    dn = len(deg)
    dl = len(lb)
    du = len(ub)

    if (dn != dl):
        print("Dimension Mismatch: Stock dimension != lower bounds dimension")    
    elif (dn != du) :
        print("Dimension Mismatch: Stock dimension != upper bounds dimension")
    else:
        param = dict({'degree': deg, 'lowerB': lb, 'upperB': ub, 'delta': delta})
    
    return param



def chebnodegen(n, a, b):
    '''
    The function generates uni-dimensional chebyshev nodes.

    n:   A number of nodes
    a:   The lower bound of inverval [a,b]
    b:   The upper bound of interval [a,b]

    Details:

    A polynomial approximant, S_i, over a bounded interval [a, b] is constructed by,
    s_i = (b + a)/2 + (b-a)/2 * cos((n - i + 0.5)/(n) * pi) for i = 1, 2, ..., n
    '''
    d1 = len(a)
    d2 = len(b)

    n = n[0]; a = a[0]; b = b[0]

    si = (a + b) * 0.5 + (b - a) * 0.5 * np.cos(np.pi * ((np.arange( (n - 1), -1, -1) + 0.5)/n))
    if (d1 != d2):
        raise Exception("Dimension Mismatch: dim(upper) != dim(lower)")
    
    return si



# stock = st[0]
# npol = 50
# a = 5e+06
# b = 359016000
# dorder = None



def chebbasisgen(stock, npol, a, b, dorder = None):
    '''
    The function calculates the monomial basis of Chebyshev polynomials for the 
    given unidimensional nodes, s_i over a bounded interval [a,b].

    stock:      An array of Chebyshev polynomial nodes si (an array of stocks in capn-packages)
    npol:       Number of polynomials (n polynomials = (n-1)-th degree)
    a:          The lower bound of inverval [a,b]
    b:          The upper bound of inverval [a,b]
    dorder:     Degree of partial derivative of the

    Details:

    Suppose there are m numbers of Chebyshev nodes over a bounded interval [a, b]
    
    s_i in [a, b] for i = 1, 2, ..., m

    These nodes can be nomralized to the standard Chebyshev nodes over the domain [-1,1]:

    z_i = 2(s_i - a) / (b - a) - 1

    With normalized Chebyshev nodes, the recurrence relations of Chebyshev polynomials of order
    n is defined as:

    T0(zi) = 1
    T1(zi) = zi
    Tn(zi) = 2 z_i T_(n−1)(zi) − T_(n−2) (zi).

    The interpolation matrix (Vandermonde matrix) of (n-1)-th Chebyshev polynomials 
    with m nodes, Φmn is:
           
           1  T1(z1) · · · Tn−1(z1) 
           1  T1(z2) · · · Tn−1(z2) 
           .    .    . . .    .     
    Φmn =  .    .    . . .    .     
           .    .    . . .    .     
           .    .    . . .    .     
           1  T1(zm) · · · Tn−1(zm) 
           
    The partial derivative of the monomial basis matrix can be found by the relation:
    
    (1 − z_i^2) T'_n (z_i) = n[T_(n - 1) (z_i) - z_i T_n(z_i)]
    '''
    if isinstance(stock, float): 
        nknots = 1
    else:
        nknots = len(stock)

    # stock minus knots minus minimum divided by max - min
    z = (2 * stock - b - a) / (b - a)

    if (npol < 4): 
        raise Exception("Degree of Chebyshev polynomial should be greater than 3!")

    # Initialize base matrix
    bvec = pd.DataFrame({'j=1': np.repeat(1, nknots), 'j=2': z})

    # Normalize Chebyshev nodes
    for j in np.arange(2, (npol)):
        Tj = pd.DataFrame({'jj': (2 * z * bvec.iloc[:, j - 1] - bvec.iloc[:, j - 2])})
        Tj = Tj.rename(columns={'jj': f"j={j+1}"})
        bvec = pd.concat([bvec, Tj], axis=1)
   
    if dorder is None:
        res = bvec
    
    elif (dorder == 1):
        bvecp = pd.DataFrame({'j=1': 0, 'j=2': np.repeat(2/(b - a), nknots)})    
        
        # Generate interpolation matrix with n nodes
        for j in np.arange(2, (npol)):
            Tjp = pd.DataFrame({'jj': ((4/(b - a)) * bvec.iloc[:, j - 1] + 2 * z * bvecp.iloc[:, j - 1] - bvecp.iloc[:, j - 2])})  
            Tjp = Tjp.rename(columns={'jj': f"j={j+1}"})
            bvecp = pd.concat([bvecp, Tjp], axis=1)

        res = bvecp

    else:
        raise Exception("dorder should be NULL or 1!")

    return res


# approxspace = Aspace
# sdata = simuDataV

def vapprox(approxspace, sdata):
    '''
    The function provides the V-approximation coefficients of the 
    defined Chebyshev polynomials in aproxdef.

    degree:       degree of Chebyshev polynomial
    lowerB:       lower bound of Chebyshev nodes
    upperB:       upper bound of Chebyshev nodes
    delta:        discount rate
    coefficient:  Chebyshev polynomial coefficients

    Details:

    The V-approximation is finding the shadow price of i-th stock, pi for 
    i = 1, · · · , d from the relation:
    δV = W(S) + p1s˙1 + p2s˙2 + · · · + pds˙d,
    
    where δ is the given discount rate, V is the intertemporal welfare 
    function, S = (s1, s2, · · · , sd) is a vector of stocks, W(S) is the 
    net benefits accruing to society, and \\dot{s}_i is the growth of stock si
    .
    By the definition of the shadow price, we know:

    p_i = ∂V / ∂s_i

    Consider approximation V (S) = µ(S)β, µ(S) is Chebyshev polynomials and β 
    is their coeffcients. Then, pi = µ_si (S)β by the orthogonality of Chebyshev 
    basis. Adopting the properties above, we can get the unknown coefficient 
    vector β from:

    δµ(S)β = W(S) +\\sum_i=1^d diag(\\dot{s}_i) µ_si (S)β, and thus,

    β = [delta µ(S) - \\sum_i=1^d diag(\\dot{s}_i) µ_si (S)]^(−1) W(S).

    Additional case: over-determined (more nodes than approaximation degrees)

    '''
    deg = approxspace["degree"]
    lb = approxspace["lowerB"]
    ub = approxspace["upperB"]
    delta = approxspace["delta"]
    dd = len([deg])

    if isinstance(sdata, pd.DataFrame):
        sdata = sdata.to_numpy()
    
    if not isinstance(sdata, np.ndarray):
        print("sdata should be a data.frame or matrix of [stock, sdot, w]!")

    if (sdata.shape[1] != (2 * dd + 1)): 
        print("The number of columns in sdata is not right!")

    # INCOMPLETE
    # if (dd > 1):
    #     ordername = f"sdata[, {dd}]"
    #     for di in np.arange(2, dd):
    #         odtemp = f"sdata[, {dd - di + 1}]"
    #         ordername = paste0([ordername, odtemp], sep = ", ")
        
    #     ordername = f"sdata[order({ordername}),]"

    #     sdata <- eval(parse(text = ordername))

    else:
        sdata = sdata[sdata[:, 0].argsort()]

    # Get unique nodes
    st = [np.unique(sdata[:, k]) for k in np.arange(0, dd)] 
    
    # Get sdot values
    sdot = [sdata[:, k] for k in np.arange((dd + 0), (2 * dd))]  

    # Get w (net-benefit) W(S)
    w = sdata[:, (2 * dd + 0)]    

    # Setup matrices
    fphi = np.matrix(1)
    sphi = np.zeros(( int(( np.prod([len(k) for k in st]) * np.prod(deg)) / np.prod(deg)), np.prod(deg) ))

    # Generate Chebychev Approximation matrices
    for di in np.arange(0, dd):
        dk = dd - di - 1
        ftemp = chebbasisgen(st[dk], deg[dk], lb[dk], ub[dk])
        fphi = np.kron(fphi, ftemp)

        stempi = chebbasisgen(st[di], deg[di], lb[di], ub[di], dorder = 1)
        sphitemp = np.matrix(1)
        for dj in np.arange(0, dd):
            dk2 = dd - dj - 1
            if (dk2 != di):
                stemp = chebbasisgen(st[dk2], deg[dk2], lb[dk2], ub[dk2])
            else:
                stemp = stempi
            
            sphitemp = np.kron(sphitemp, stemp)
        
        #Calculate:  \\sum_i=1^d diag(\\dot{s}_i) µ_si (S)
        sphi = np.array(sphi) + np.array(sphitemp) * sdot[di][:, np.newaxis]

    # Calculate: [delta µ(S) - \\sum_i=1^d diag(\\dot{s}_i) µ_si (S)]^(−1)
    nsqr = delta[0] * np.array(fphi) - np.array(sphi)

    # Solve for all betas (shadow price)    
    if (fphi.shape[0] == fphi.shape[1]):
        coeff = np.linalg.lstsq(nsqr, w, rcond=None)[0]
        res = dict({'degree': deg, 'lowerB': lb, 'upperB': ub, 
            'delta': delta, 'coefficient': coeff})
    
    # Solve for beta when over-determined
    elif (fphi.shape[0] != fphi.shape[1]):
        coeff = np.linalg.lstsq(nsqr.T @ nsqr, nsqr.T @ w, rcond=None)[0]
        res = dict({'degree': deg, 'lowerB': lb, 'upperB': ub, 
            'delta': delta, 'coefficient': coeff})
    
    return res



# approxspace = Aspace
# stock = simuDataP.iloc[:, 0]
# sdot = simuDataP.iloc[:, 1]
# dsdotds = simuDataP.iloc[:, 2]
# dwds = simuDataP.iloc[:, 3]


def papprox(approxspace, stock, sdot, dsdotds, dwds):
    '''
    The function provides the P-approximation coefficients of the 
    defined Chebyshev polynomials in aproxdef. For now, only 
    unidimensional case is developed.

    aproxspace:   An approximation space defined by aproxdef function
    stock:        An array of stock 
    sdot:         An array of ds/dt
    dsdotds:      An array of d(sdot)/ds
    dwds:         An array of dw/ds

    Details:

    The P-approximation is finding the shadow price of a stock, p from the relation:


    '''
    deg = approxspace["degree"]
    lb = approxspace["lowerB"]
    ub = approxspace["upperB"]
    delta = approxspace["delta"]
    
    # if isinstance(stock, np.matrix):
    stock = np.array(stock)

    # if isinstance(sdot, np.matrix):
    sdot = np.array(sdot)
    
    # if isinstance(dsdotds, np.matrix):
    dsdotds = np.array(dsdotds)
    
    # if isinstance(dwds, np.matrix):
    dwds = np.array(dwds)
    
    fphi = chebbasisgen(stock, deg[0], lb[0], ub[0])
    sphi = chebbasisgen(stock, deg[0], lb[0], ub[0], dorder = 1)
    nsqr = (delta[0] - dsdotds)[:, np.newaxis] * fphi - sdot[:, np.newaxis] * sphi

    if (deg[0] == len(stock)):        
        coeff = np.linalg.lstsq(nsqr, dwds, rcond=None)[0]
        res = dict({'degree': deg, 'lowerB': lb, 'upperB': ub, 
            'delta': delta, 'coefficient': coeff})
    elif (deg[0] < len(stock)):
        coeff = np.linalg.lstsq((nsqr.T @ nsqr).T, nsqr.T @ dwds, rcond=None)[0]
        res = dict({'degree': deg, 'lowerB': lb, 'upperB': ub, 
            'delta': delta, 'coefficient': coeff})

    return res



# approxspace = Aspace
# stock = simuDataPdot.iloc[:, 0]
# sdot = simuDataPdot.iloc[:, 1]
# dsdotds = simuDataPdot.iloc[:, 2]
# dsdotdss = simuDataPdot.iloc[:, 3]
# dwds = simuDataPdot.iloc[:, 4]
# dwdss = simuDataPdot.iloc[:, 5]

def pdotapprox(approxspace, stock, sdot, dsdotds, dsdotdss, dwds, dwdss):
    deg = approxspace["degree"]
    lb = approxspace["lowerB"]
    ub = approxspace["upperB"]
    delta = approxspace["delta"]
    
    # if isinstance(stock, np.matrix):
    stock = np.array(stock)

    # if isinstance(sdot, np.matrix):
    sdot = np.array(sdot)
    
    # if isinstance(dsdotds, np.matrix):
    dsdotds = np.array(dsdotds)

    # if isinstance(dsdotdss, np.matrix):
    dsdotdss = np.array(dsdotdss)
    
    # if isinstance(dwds, np.matrix):
    dwds = np.array(dwds)

    # if isinstance(dwds, np.matrix):
    dwdss = np.array(dwdss)

    fphi = chebbasisgen(stock, deg[0], lb[0], ub[0])
    sphi = chebbasisgen(stock, deg[0], lb[0], ub[0], dorder = 1)
    # nsqr good
    nsqr = (((delta[0] - dsdotds)**2)[:, np.newaxis] * fphi - sdot[:, np.newaxis] * (delta[0] - dsdotds[:, np.newaxis]) * sphi - dsdotdss[:, np.newaxis] * sdot[:, np.newaxis] * fphi)
    # nsqr2 calc good
    nsqr2 = dwdss * sdot * (delta[0] - dsdotds) + dwds * dsdotdss * sdot

    if (deg[0] == len(stock)):       
        coeff = (np.linalg.lstsq(nsqr, nsqr2, rcond=-1)[0])
        res = dict({'degree': deg, 'lowerB': lb, 'upperB': ub, 
            'delta': delta, 'coefficient': coeff})
    elif (deg[0] < len(stock)):
        coeff = np.linalg.lstsq(nsqr.T @ nsqr, nsqr.T @ nsqr2, rcond=-1)[0]
        res = dict({'degree': deg, 'lowerB': lb, 'upperB': ub, 
            'delta': delta, 'coefficient': coeff})

    return res



# vcoeff = vC
# adata = simuDataV.iloc[:, 0]
# wval = profit(nodes, param)


def vsim(vcoeff, adata, wval=None):
    '''
   The function provides the V-approximation simulation by adopting 
   the results of vaprox. Available for multiple stock problems. 
    '''
   
    deg = vcoeff["degree"]
    lb = vcoeff["lowerB"]
    ub = vcoeff["upperB"]
    delta = vcoeff["delta"]
    coeff = vcoeff['coefficient']
    nnodes = len(adata)
    dd = len(deg)

    if isinstance(adata, pd.DataFrame):
        st = adata.to_numpy()
    elif isinstance(adata, pd.Series):
        st = adata.to_numpy()
    elif isinstance(adata, np.matrix):
        st = adata
    else:
        raise Exception("st is not a matrix or data.frame")

    accp = np.zeros( (nnodes * dd, dd) )
    Bmat = np.zeros( (int(nnodes * np.prod(deg) / np.prod(deg)), np.prod(deg)) )

    for di in np.arange(0, dd):
        Bprime = np.zeros( (int(nnodes * np.prod(deg) / np.prod(deg)), np.prod(deg)))
        for ni in np.arange(0, nnodes):
            sti = st[[ni]]
            dk = dd - di - 1
            fphi = np.matrix(1)
            ftemp = chebbasisgen(sti[di], deg[di], lb[di], ub[di])
            sphi = np.matrix(1)
            stempd = chebbasisgen(sti[di], deg[di], lb[di], ub[di], dorder=1)
            for dj in np.arange(0, dd):
                dk2 = dd - dj - 1
                if (dk2 != di):
                    ftemp = chebbasisgen(sti[dk2], deg[dk2], lb[dk2], ub[dk2])
                    stemp = ftemp
                else:
                    stemp = stempd
                fphi = np.kron(fphi, ftemp)
                sphi = np.kron(sphi, stemp)

            Bmat[ni, :] = fphi 
            Bprime[ni, :] = sphi
        accp[:, di] = Bprime @ coeff
    # NEED TO FIGURE OUT WHY THIS IS OFF
    iwhat = accp.T * st   

    # test = pd.DataFrame(Bmat)
    # np.sum(test.iloc[:, 10].values, dtype='float128')
    # np.sum(test.iloc[:, 10].values, dtype='float128') == -1.21486154469608e-13

    # test = pd.DataFrame(Bprime)
    # np.sum(test.iloc[:, 10].values, dtype='float128')
    # np.sum(test.iloc[:, 10].values, dtype='float128') == -1.47767544626473e-20 

    # iwhat.sum() == 197912936059.535

    iw = iwhat.ravel()
    vhat = Bmat @ coeff
    # np.sum(vhat, dtype='float128') == 717027680655.274

    # colnames(accp) <- paste("acc.price", 1:dd, sep = "")   # "acc.price1"
    # colnames(iwhat) <- paste("iw", 1:dd, sep = "")   # "iw1"
    # colnames(iw) <- c("iw")   # iw

    if not isinstance(wval, np.ndarray):
        wval = "wval is not provided"

    res = dict({'shadowp': accp, 'iweach': iwhat, 'iw': iw, 
            'vfun': vhat, 'stock': st, 'wval': wval})
    
    return res



# pcoeff = pC
# stock = simuDataP.iloc[:, 0]
# wval = profit(nodes, param)
# sdot = simuDataP.iloc[: , 1]

def psim(pcoeff, stock, wval = None, sdot = None):
    '''
    The function provides the P-approximation simulation.

    '''
    deg = pcoeff["degree"]
    lb = pcoeff["lowerB"]
    ub = pcoeff["upperB"]
    delta = pcoeff["delta"]
    coeff = pcoeff['coefficient'].T
    nnodes = len(stock)
    nullcheck = isinstance(wval, type(None)) == isinstance(sdot, type(None))

    if (nullcheck != True):
        raise Exception("wval and sdot are both None")

    dd = len(deg)
    accp = np.zeros( (int(nnodes * dd / dd), dd) )
    iw = np.zeros( (int(nnodes * dd / dd), dd) )
    vhat = "wval and sdot not provided"
    for ni in np.arange(0, nnodes):
        sti = stock[ni]
        accp[ni, :] = chebbasisgen(sti, deg[0], lb[0], ub[0]) @ coeff
        iw[ni, :] = accp[ni, :] * sti
    if not isinstance(wval, type(None)):
        vhat = np.zeros( (int(nnodes * dd / dd), dd) )
        for ni in np.arange(0, nnodes):
            vhat[ni, :] = (wval[ni] + accp[ni, :] * sdot[ni])/delta

    if isinstance(wval, type(None)):
        wval = "wval and sdot not provided"

    res = dict({'shadowp': accp, 'iw': iw, 'vfun': vhat, 
        'stock': stock, 'wval': wval})

    return res



# pdotcoeff = pdotC
# stock = simuDataPdot.iloc[:, 0]
# sdot = simuDataPdot.iloc[:, 1]
# dsdotds = simuDataPdot.iloc[:, 2]
# wval = profit(nodes,param)
# dwds = simuDataPdot.iloc[:, 4]


def pdotsim(pdotcoeff, stock, sdot, dsdotds, wval, dwds):
    '''
    The function provides the Pdot-approximation coefficients of the defined 
    Chebyshev polynomials in aproxdef. For now, only unidimensional case is developed.
    '''
    deg = pdotcoeff["degree"]  
    lb = pdotcoeff["lowerB"]
    ub = pdotcoeff["upperB"]
    delta = pdotcoeff["delta"]
    coeff = pdotcoeff['coefficient'].T
    nnodes = len(stock)
    dd = len(deg)
    accp = np.zeros( (int(nnodes * dd / dd), dd) )
    iw = np.zeros( (int(nnodes * dd / dd), dd) )
    vhat = np.zeros( (int(nnodes * dd / dd), dd) )
    for ni in np.arange(0, nnodes):   
        sti = stock[ni]
        pdoti = chebbasisgen(sti, deg[0], lb[0], ub[0]) @ coeff
        accp[ni, :] = (dwds[ni] + pdoti) / (delta - dsdotds[ni])
        iw[ni, :] = (accp[ni, :] * sti)
        vhat[ni, :] = (wval[ni] + accp[ni, :] * sdot[ni]) / delta

    res = dict({'shadowp': accp, 'iw': iw, 'vfun': vhat, 
                'stock': stock, 'wval': wval})

    return res



