'''
Harmonic Oscillator Volitility Script
'''

#Startup Code Begin
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from scipy.stats import t,norm
from scipy.special import hermite
from scipy.integrate import quad
from scipy.optimize import curve_fit
from datetime import datetime,timedelta
import itertools
import pymc3 as pm

#Startup Code End

def load_data(ticker,reader=False):
    ticker = ticker.upper()
    fname = r'E:/Investing/historical_data/{}.csv'.format(ticker)

    if reader==False:
        df = pd.read_csv(fname,index_col='Date')
    else:
        df = web.DataReader(ticker,'yahoo')

    df['logs'] = np.log(df.Close)
    df['lr'] = np.log(1+df.Close.pct_change())

    df.dropna(inplace=True)
    lreturns = df.lr.values
    lreturns = lreturns[lreturns.nonzero()[0]]

    return(df,lreturns)




def get_paramaters(data,lreturns,n_levels,market_cap):
    '''
    :type data: pd dataframe
    :param data: Dataframe with log returns with key 'lr'

    :type n_levels: int
    :param n_levels: number of energy levels to model -> default to Beta**2
    '''
    #Set Constants
    eV = 1.60218e-19    #electron volt eV
    hbar = 6.582119e-16     #eV*s
    total_market_cap = 34e12
    total_avg_volume = (1.7e12)/12.

    #Weekly log returns
    weeks = int(len(lreturns)/5.)
    days = int(weeks*5)

    log_returns = lreturns[:days].reshape(weeks,5)
    wvar = log_returns.var(axis=1)
    

    dvar = max(wvar)/n_levels
    de = dvar*eV

    #Get mass -> m = mc/total_mc * avg_vol/total_avg_vol
    vol = data.Volume.mean()
    m = (market_cap/total_market_cap)*(vol/total_avg_volume)

    #get omega -> w = sqrt(kf/m)
    kf = (m*de**2)/hbar**2
    w = np.sqrt(kf/m)

    alpha = np.sqrt(hbar/(m*w))

    params = dict(energy=de,levels=n_levels,kf=kf,m=m,w=w,a=alpha)
    return(params)

def get_cv(log_returns,params,window):
    '''
    :type data: dataframe
    :param data: data for the ticker

    :type params: dict
    :param params: paramaters for the stock:
    Key:Value
    energy = change in energy between levels
    levels = number of levels for a given company
    kf = force constant
    m = mass of business
    w = omega
    a = alpha

    :type window: int
    :param window: the window in time to check: something like last thirty days
    '''
    eV = 1.60218e-19
    current_var = log_returns[-window:].var()
    cv = int(np.round(np.abs(current_var/log_returns.var())*2.0,0))

    return(cv)



def wavefunction(params,level):

    '''
    :type: params: dict
    :param params: key:value pairs -> energy = dE,kf = force constant,m=mass,w = omega

    :type num_levels: int
    :param num_levels: number of energy levels -> matches get_paramaters
    '''
    #Set Constants
    hbar = 6.582119e-16     #eV*s
    omega = params['w']
    kf = params['kf']
    m = params['m']

    #Define the first five Hermite polynomials
    H0 = 1
    H1 = lambda y: 2*y
    H2 = lambda y: 4*y**2 - 2
    H3 = lambda y: 8*y**3 - 12*y
    H4 = lambda y: 16*y**4 - 48*y**2 + 12
    H5 = lambda y: 32*y**5 - 160*y**3 + 120*y

    #Define Normalization Constant
    Nv = lambda v: ((m*omega)/(np.pi*hbar))**(1/4) * (1/(np.sqrt(2**v*np.math.factorial(v))))

    #Define alpha
    alpha = params['a']

    #Calculate Displacement
    Ev = hbar*omega*(level+.5)
    displacement = np.sqrt(Ev/(m*omega**2))

    #Define the integral -> has the form polynomial * exp(-y**2)
    #Get polynomial function -> HX(x)**2 * exp(-x**2)
    if level == 0:
        wf = lambda x: Nv(level)*np.exp(-x**2/2.)
        pdf = lambda x: Nv(level)**2 * np.exp(-x**2)
        fx = lambda x: alpha*Nv(level)**2 * np.exp(-x**2)
    else:
        if level == 1:
            wf = lambda x: Nv(level)*H1(x)*np.exp(-x**2/2)
            pdf = lambda x: Nv(level)**2 * H1(x)**2 * np.exp(-x**2)   #x = x/alpha
            fx = lambda x: alpha*Nv(level)**2 * H1(x)**2 * np.exp(-x**2)     #x=y
        elif level == 2:
            wf = lambda x: Nv(level)*H2(x)*np.exp(-x**2/2)
            pdf = lambda x: Nv(level)**2 * H2(x)**2 * np.exp(-x**2)
            fx = lambda x: alpha*Nv(level)**2 * H2(x)**2 * np.exp(-x**2)
        elif level == 3:
            wf = lambda x: Nv(level)*H3(x)*np.exp(-x**2/2)
            pdf = lambda x: Nv(level)**2 * H3(x)**2 * np.exp(-x**2)
            fx = lambda x: alpha*Nv(level)**2 * H3(x)**2 * np.exp(-x**2)
        elif level == 4:
            wf = lambda x: Nv(level)*H4(x)*np.exp(-x**2/2)
            pdf = lambda x: Nv(level)**2 * H4(x)**2 * np.exp(-x**2)
            fx = lambda x: alpha*Nv(level)**2 * H4(x)**2 * np.exp(-x**2)
        elif level == 5:
            wf = lambda x: Nv(level)*H5(x)*np.exp(-x**2/2)
            pdf = lambda x: Nv(level)**2 * H5(x)**2 * np.exp(-x**2)
            fx = lambda x: alpha*Nv(level)**2 * H5(x)**2 * np.exp(-x**2)
        else:
            print("Invalid energy level")
    

    return(wf,pdf,fx)

def create_cdf(params,pdf,fx,level):
    xr = np.arange(-5,5,.01)
    alpha = params['a']
    xa = xr/alpha
    xa = xa[(xa>-10.0)&(xa<10.0)]

    if level==0:
        pd = np.round(pdf(xa),3)

        #Create dist
        cdist = []
        for x in xa:
            cdist.append(quad(fx,-np.inf,x)[0])

        distributions = list(zip(xa,cdist))


    else:

        #find where prob density != 0
        pd = np.round(pdf(xa),2)
        nz = pd.nonzero()[0]
        ind = nz[np.where(np.diff(nz)!=1)[0]+1]

        #Get x values between index values
        indexes = []
        start = nz[0]
        for i in range(len(ind)):
            indexes.append(xa[start:ind[i]])
            start = ind[i]
        indexes.append(xa[ind[-1]:nz[-1]])
        
        

        #Create Distribution
        c_dist = []
        total = len(indexes)
        last_max = 0
        for i in range(len(indexes)):
            a = indexes[i][0]
            temp=[]
            for j in indexes[i]:
                temp.append(quad(fx,a,j)[0])
            
            
            
            c_dist.append(np.array(temp)+last_max)
            last_max += temp[-1]

        distributions = list(zip(indexes,c_dist))

    return(distributions)


def fit_cdf(cdfs,pdf,level):
    '''
    :type cdfs: list of tuples
    :param cdfs: tuple containing the xvalues and distribution x[0]=x x[1]=dist
    '''

    lfit = lambda x,A,k,x0: A/(1+np.exp(-k*(x-x0)))
    lfit2 = lambda x,A,k,x0,b: A/(1+np.exp(-k*(x-x0))) + b

    if level==0:
        xr = [x[0] for x in cdfs]
        cd = [x[1] for x in cdfs]
        popt,pcov = curve_fit(lfit,xr,cd)
        fit_list = [popt]

    else:
        fit_list = []
        for i in range(len(cdfs)):
            xr = cdfs[i][0]
            cd = cdfs[i][1]
            
            pdxa = pdf(xr)
            mean_index = np.argmax(pdxa)
            sig_mean = xr[mean_index]
            amplitude = max(cd)-min(cd)
            
            if i==0:
                popt,pcov = curve_fit(lfit,xr,cd,p0=[amplitude,1.0,sig_mean])
                fit_list.append(popt)
            else:
                offset = max(cd) - amplitude
                popt,pcov = curve_fit(lfit2,xr,cd,p0 = [amplitude,1.0,sig_mean,offset])
                fit_list.append(popt)

    return(fit_list)                
            

def forecast_returns(cdf,pdf,level,time_steps,paths):
    '''
    :type fit_params: list or nested list
    :param fit_list: paramaters of fit for a logistic regression. Used to create quantile function to sample.

    :type level: int
    :param level: volatility level to sample. Instructs the function to sample each prob dist.

    :type shape: tuple
    :param shape: shape of the random number array

    This function is to be used for one energy level at a time, not a sample of multiple energy levels.
    '''
    sample_size = time_steps*paths*2
    quantile = lambda y,A,k,x0: np.log((y*np.exp(k*x0))/(A-y))/k
    quantile2 = lambda y,A,k,x0,b: np.log(((y-b)*np.exp(k*x0))/(A+b-y))/k
    rsample = np.random.random(sample_size)

    if level==0:    #Defined everywhere
        fit = fit_cdf(cdf,pdf,level)[0]
        sample = quantile(rsample,*fit)
        rselection = np.random.choice(sample,time_steps*paths)
        returns = rselection.reshape(time_steps,paths)

    else:
        fit_list = fit_cdf(cdf,pdf,level)
        plist = [0]
        for x in cdf:
            plist.append(x[1][-1])
        p_ranges = []
        for i in range(1,len(plist)):
            p_ranges.append((plist[i-1],plist[i]))
        
        sample = []
        #Break rsample apart
        for i in range(len(p_ranges)):
            rs = rsample[(rsample>p_ranges[i][0])&(rsample<p_ranges[i][1])]
            fit = fit_list[i]
            if i==0:
                sample.append(quantile(rs,*fit))
            else:
                sample.append(quantile2(rs,*fit))

        sample = np.concatenate(sample)
        np.random.shuffle(sample)
        rselection = np.random.choice(sample,time_steps*paths)
        returns = rselection.reshape(time_steps,paths)

    return(returns)


def init_session():
    '''
    '''
    ticker = input("Enter the ticker symbol: ").upper()
    mcap = float(input("Enter the market cap for the underlying: "))
    levels = int(input("Enter the number of volatility levels: "))

    print("Loading Data and Calculating Paramaters")

    df,lreturns = load_data(ticker,reader=True)
    p = get_paramaters(df,lreturns,levels,mcap)

    #vwindow = int(input("What window to calculate volatility: "))
    vlevel = int(input("What is the Current Volatility? "))

    print("Creating Wavefunction and Cumulative Distribution")
    psi,pdf,fx = wavefunction(p,vlevel)
    cdf = create_cdf(p,pdf,fx,vlevel)

    print("Created wavefunctions for: {}, market cap: {}, and volatility level: {}".format(ticker,mcap,vlevel))

    days = int(input("How many days to forecast: "))
    npaths = int(input("How many paths to create: "))

    returns = forecast_returns(cdf,pdf,level=vlevel,time_steps=days,paths=npaths)

    print("Forecasting returns")
    print("Dictionary Keys: data,params,psi,pdf,fx,cdf,preturns")

    results = dict(ticker=ticker,data=df,log_returns=lreturns,params=p,psi=psi,pdf=pdf,fx=fx,cdf=cdf,preturns=returns)

    return(results)



def test_fit():
    ticker = input("Enter the ticker symbol: ").upper()
    mcap = float(input("Enter the market cap for the underlying: "))
    levels = 5

    print("Getting the Paramaters and Wavefunctions")


    df,lreturns = load_data(ticker)
    p = get_paramaters(df,lreturns,levels,mcap)
    
    lfit = lambda x,A,k,x0: A/(1+np.exp(-k*(x-x0)))
    lfit2 = lambda x,A,k,x0,b: A/(1+np.exp(-k*(x-x0))) + b

    while True:
        elevel = input("What Energy Level to Test or Q to Quit? ")
        xr = np.arange(-5,5,.01)/p['a']
        xr = xr[(xr>-10.0)&(xr<10.0)]

        if elevel=='0':
            
            psi0,pdf0,fx0 = wavefunction(p,0)
            cd0 = create_cdf(p,pdf0,fx0,0)
            cdx = [x[0] for x in cd0]
            cdd = [x[1] for x in cd0]
            cf0 = fit_cdf(cd0,pdf0,0)

            fig,(ax1,ax2) = plt.subplots(2)
            fig.suptitle("{}: Level 0".format(ticker))
            ax1.plot(xr,pdf0(xr))
            ax1.set_title("Probability Density")
            ax2.scatter(xr,cdd,color='b',alpha=.5)
            ax2.plot(xr,lfit(xr,*cf0[0]),color='r')
            ax2.set_title("Cumulative Distribution and Logistic Fit")

            plt.show()


        elif elevel=='1':
            psi1,pdf1,fx1 = wavefunction(p,1)
            cd1 = create_cdf(p,pdf1,fx1,1)
            cf1 = fit_cdf(cd1,pdf1,1)

            cdx1 = cd1[0][0]
            cdd1 = cd1[0][1]

            cdx2 = cd1[1][0]
            cdd2 = cd1[1][1]


            fig,(ax1,ax2,ax3) = plt.subplots(3)
            fig.suptitle("{}: Level 1".format(ticker))
            ax1.plot(xr,pdf1(xr))
            ax1.set_title("Probability Density Function")
            ax2.scatter(cdx1,cdd1,color='b',alpha=.5)
            ax2.plot(cdx1,lfit(cdx1,*cf1[0]),color='r')
            ax2.set_title("Cumulative Distribution 1")
            ax3.scatter(cdx2,cdd2,color='b',alpha=.5)
            ax3.plot(cdx2,lfit2(cdx2,*cf1[1]),color='r')
            ax3.set_title("Cumulative Distribution 2")

            plt.show()


        elif elevel=='2':
            psi2,pdf2,fx2 = wavefunction(p,2)
            cd2 = create_cdf(p,pdf2,fx2,2)
            cf2 = fit_cdf(cd2,pdf2,2)

            cdx1 = cd2[0][0]
            cdd1 = cd2[0][1]
            cdx2 = cd2[1][0]
            cdd2 = cd2[1][1]
            cdx3 = cd2[2][0]
            cdd3 = cd2[2][1]

            fig,(ax1,ax2,ax3,ax4) = plt.subplots(4)
            fig.suptitle("{}: Level 2".format(ticker))
            ax1.plot(xr,pdf2(xr))
            ax1.set_title("Probability Density")

            ax2.scatter(cdx1,cdd1,color='b',alpha=.5)
            ax2.plot(cdx1,lfit(cdx1,*cf2[0]),color='r')

            ax3.scatter(cdx2,cdd2,color='b',alpha=.5)
            ax3.plot(cdx2,lfit2(cdx2,*cf2[1]),color='r')

            ax4.scatter(cdx3,cdd3,color='b',alpha=.5)
            ax4.plot(cdx3,lfit2(cdx3,*cf2[2]),color='r')

            plt.show()


        elif elevel=='3':
            psi3,pdf3,fx3 = wavefunction(p,3)

            cd3 = create_cdf(p,pdf3,fx3,3)
            cf3 = fit_cdf(cd3,pdf3,3)

            cdx1 = cd3[0][0]
            cdd1 = cd3[0][1]
            cdx2 = cd3[1][0]
            cdd2 = cd3[1][1]
            cdx3 = cd3[2][0]
            cdd3 = cd3[2][1]
            cdx4 = cd3[3][0]
            cdd4 = cd3[3][1]

            fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5)
            fig.suptitle("{}: Level 3".format(ticker))
            ax1.plot(xr,pdf3(xr))
            ax1.set_title("Probability Density")

            ax2.scatter(cdx1,cdd1,color='b',alpha=.5)
            ax2.plot(cdx1,lfit(cdx1,*cf3[0]),color='r')

            ax3.scatter(cdx2,cdd2,color='b',alpha=.5)
            ax3.plot(cdx2,lfit2(cdx2,*cf3[1]),color='r')

            ax4.scatter(cdx3,cdd3,color='b',alpha=.5)
            ax4.plot(cdx3,lfit2(cdx3,*cf3[2]),color='r')

            ax5.scatter(cdx4,cdd4,color='b',alpha=.5)
            ax5.plot(cdx4,lfit2(cdx4,*cf3[3]),color='r')

            plt.show()


        elif elevel=='4':
            psi4,pdf4,fx4 = wavefunction(p,4)

            cd4 = create_cdf(p,pdf4,fx4,4)
            cf4 = fit_cdf(cd4,pdf4,4)

            cdx1 = cd4[0][0]
            cdd1 = cd4[0][1]
            cdx2 = cd4[1][0]
            cdd2 = cd4[1][1]
            cdx3 = cd4[2][0]
            cdd3 = cd4[2][1]
            cdx4 = cd4[3][0]
            cdd4 = cd4[3][1]
            cdx5 = cd4[4][0]
            cdd5 = cd4[4][1]

            fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6)
            fig.suptitle("{}: Level 4".format(ticker))
            ax1.plot(xr,pdf4(xr))
            ax1.set_title("Probability Density")

            ax2.scatter(cdx1,cdd1,color='b',alpha=.5)
            ax2.plot(cdx1,lfit(cdx1,*cf4[0]),color='r')

            ax3.scatter(cdx2,cdd2,color='b',alpha=.5)
            ax3.plot(cdx2,lfit2(cdx2,*cf4[1]),color='r')

            ax4.scatter(cdx3,cdd3,color='b',alpha=.5)
            ax4.plot(cdx3,lfit2(cdx3,*cf4[2]),color='r')

            ax5.scatter(cdx4,cdd4,color='b',alpha=.5)
            ax5.plot(cdx4,lfit2(cdx4,*cf4[3]),color='r')

            ax6.scatter(cdx5,cdd5,color='b',alpha=.5)
            ax6.plot(cdx5,lfit2(cdx5,*cf4[4]),color='r')

            plt.show()

        elif elevel=='5':
            psi5,pdf5,fx5 = wavefunction(p,5)

            cd5 = create_cdf(p,pdf5,fx5,5)
            cf5 = fit_cdf(cd5,pdf5,5)

            cdx1 = cd5[0][0]
            cdd1 = cd5[0][1]
            cdx2 = cd5[1][0]
            cdd2 = cd5[1][1]
            cdx3 = cd5[2][0]
            cdd3 = cd5[2][1]
            cdx4 = cd5[3][0]
            cdd4 = cd5[3][1]
            cdx5 = cd5[4][0]
            cdd5 = cd5[4][1]
            cdx6 = cd5[5][0]
            cdd6 = cd5[5][1]

            fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7)
            fig.suptitle("{}: Level 5".format(ticker))
            ax1.plot(xr,pdf5(xr))
            ax1.set_title("Probability Density")

            ax2.scatter(cdx1,cdd1,color='b',alpha=.5)
            ax2.plot(cdx1,lfit(cdx1,*cf5[0]),color='r')

            ax3.scatter(cdx2,cdd2,color='b',alpha=.5)
            ax3.plot(cdx2,lfit2(cdx2,*cf5[1]),color='r')

            ax4.scatter(cdx3,cdd3,color='b',alpha=.5)
            ax4.plot(cdx3,lfit2(cdx3,*cf5[2]),color='r')

            ax5.scatter(cdx4,cdd4,color='b',alpha=.5)
            ax5.plot(cdx4,lfit2(cdx4,*cf5[3]),color='r')

            ax6.scatter(cdx5,cdd5,color='b',alpha=.5)
            ax6.plot(cdx5,lfit2(cdx5,*cf5[4]),color='r')

            ax7.scatter(cdx6,cdd6,color='b',alpha=.5)
            ax7.plot(cdx6,lfit2(cdx6,*cf5[5]),color='r')

            plt.show()


        elif elevel.upper() =='Q':
            break

    print("Goodbye!")


def compare(data,ho_dist):
    '''
    '''
    s0 = data.Close[-30]

    log_returns = data.lr.dropna().values
    log_returns = log_returns[log_returns.nonzero()[0]]
    hshape = ho_dist.shape

    u = log_returns.mean()
    drift = u-.5*log_returns.var()
    sigma = log_returns.std()
    wt = np.random.normal(0,1,size=(hshape[0],hshape[1]-1)).T
    gbm = np.exp(drift + sigma*wt)

    prices = np.vstack([np.ones(hshape[0]),gbm])
    gbm_forecast = s0*prices.cumprod(axis=0)

    #Get cumulative returns

    #HO dist cumulative returns
    hod = np.vstack((np.ones(ho_dist.shape[1]),ho_dist)).T
    ho_dist[:,0]+=s0
    ho_forecast = ho_dist.cumsum(axis=1)


    #Plot the results against the actual data
    x = np.arange(30)
    observed = data.Close[-30:].values

    fig,ax = plt.subplots(sharex=True,figsize=(14,6))
    ax.plot(x,gbm_forecast,color='orange',alpha=.4)
    ax.plot(x,ho_forecast.T,color='blue',alpha=.4)
    ax.plot(x,observed,color='k')
    plt.show()


def compare_2():
    df,lreturns = load_data('chwy')
    p = get_paramaters(df,lreturns,4,15.9e9)
    psi0,pdf0,fx0=wavefunction(p,0)
    psi1,pdf1,fx1=wavefunction(p,1)
    psi2,pdf2,fx2=wavefunction(p,2)
    
    cd0 = create_cdf(p,pdf0,fx0,0)
    cd1 = create_cdf(p,pdf1,fx1,1)
    cd2 = create_cdf(p,pdf2,fx2,2)
    
    
    days = 30
    npaths = 500

    r0 = forecast_returns(cd0,level=0,time_steps=days,paths=npaths)
    r1 = forecast_returns(cd1,level=1,time_steps=days,paths=npaths)
    r2 = forecast_returns(cd2,level=2,time_steps=days,paths=npaths)
    
    s0 = df.Close[-1]
    r0[:,0]+=s0
    r1[:,0]+=s0
    r2[:,0]+=s0
    
    pr0 = r0.cumsum(axis=1)
    pr1 = r1.cumsum(axis=1)
    pr2 = r2.cumsum(axis=1)
    
    
    fig,(ax1,ax2,ax3) = plt.subplots(3,sharex=True)
    ax1.plot(pr0.T)
    ax2.plot(pr1.T)
    ax3.plot(pr2.T)
    
    plt.show()
    
    


def predict(results,plot_results=True,use_trend=False,trace=None):
    '''
    '''
    df = results['data']
    lr = results['log_returns']
    pdr = results['preturns']  #shape = (paths,time_steps)

    #Get the trend using pymc3 Volatility Model
    if use_trend==True:
        if trace==None:
            print("Preparing Stochastic Volatility Model")
            num_samples = int(input("Enter the number of draws: "))
            num_tune = int(input("Enter the number of tuning steps: "))
            obs = lr[-60:]
            with pm.Model() as svm:
                step_size = pm.Exponential('step_size',10.)
                volatility = pm.GaussianRandomWalk('volatility',sigma=step_size,shape=len(obs))
                nu = pm.Exponential('nu',0.1)
                returns = pm.StudentT('returns',nu=nu,lam=np.exp(-2*volatility),observed=obs)
                trace = pm.sample(draws=num_samples,tune=num_tune)
        else:
            pass
    
    npaths = pdr.shape[0]
    nsteps = pdr.shape[1]

    #Get last price
    s0 = df.Close[-1]
    predicted_returns = np.vstack((np.zeros(pdr.shape[1]),pdr)).T
    predicted_returns[:,0]+=s0
    predicted_prices = predicted_returns.cumsum(axis=1)


    if plot_results==True:

        time_steps = predicted_prices.shape[1]
        x = np.arange(time_steps)

        if use_trend==True:
            vol = np.exp(trace.get_values('volatility',burn=500,thin=50))
            v = vol.mean(axis=0)
            trend=[]
            lr2 = lr[-60:]
            for i in range(len(lr2)):
                if lr2[i]>=0:
                    trend.append(lr2[i]-v[i])
                else:
                    trend.append(lr2[i]+v[i])
            trend = np.asarray(trend)
            prd = np.log(df.Close)[0] + trend.cumsum()
            t_prices = np.exp(prd)

            drift = t_prices[-time_steps:]
            slope = np.polyfit(x,drift,1)[0]
            y = slope*x + s0

            er = t_prices.std()
            e_pos,e_neg = y+er,y-er

            fig,(ax1,ax2) = plt.subplots(2,figsize=(14,4),sharex=False)
            ax1.plot(df.Close.values[-60:],'b',label='Observed')
            ax1.plot(t_prices,'r',label='Predicted')
            ax1.set_title("Closing Prices for {}".format(results['ticker']))
            ax1.legend()

            ax2.plot(x,predicted_prices.T,color='blue',linewidth=.3,alpha=.4)
            ax2.plot(x,y,color='k',linestyle='dashed')
            ax2.plot(x,e_pos,color='r')
            ax2.plot(x,e_neg,color='r')
            ax2.fill_between(x,y,e_pos,color='r',alpha=.2)
            ax2.fill_between(x,y,e_neg,color='r',alpha=.2)
            ax2.grid(True,axis='y')
            ax2.set_title("Predicted Prices for {}".format(results['ticker']))

            plt.show()


        else:
            past = df.Close[-time_steps:]
            slope = np.polyfit(x,past,1)[0]
            y = slope*x + s0

            #Get Error Lines for y
            er = past.std()
            e_pos,e_neg = y+er,y-er

            #Plot the data
            fig,ax = plt.subplots(figsize=(14,6))
            ax.plot(x,predicted_prices.T,color='blue',linewidth=.3,alpha=.4)
            ax.plot(x,y,color='k',linestyle='dashed')
            ax.plot(x,e_pos,color='r')
            ax.plot(x,e_neg,color='r')
            ax.fill_between(x,y,e_pos,color='r',alpha=.2)
            ax.fill_between(x,y,e_neg,color='r',alpha=.2)
            ax.grid(True,axis='y')
            ax.set_title("Predicted Prices for {}".format(results['ticker']))

            plt.show()

    else:
        pass

    if use_trend == False:
        prediction_dict = dict(data=df,pdr=predicted_returns,last_close=s0,price_forecast=predicted_prices)
    else:
        prediction_dict = dict(data=df,pdr=predicted_returns,last_close=s0,price_forecast=predicted_prices,model=svm,trace=trace)

    return(prediction_dict)




def create_report(results,strike):
    prediction = predict(results,plot_results=False,use_trend=False)
    forecast = prediction['price_forecast']

    #Forecast has shape num_paths,time_steps
    #Transpose the array:
    prices = forecast.T

    #Get overall probability
    total = prices.shape[0]*prices.shape[1]
    call_prob = len(np.where(prices>=strike)[0])/total
    put_prob = len(np.where(prices<=strike)[0])/total

    #Get daily probabilities
    daily_call_probs = []
    daily_put_probs = []
    for i in range(len(prices)):
        day = i
        dprices = prices[i]

        dc_prob = len(np.where(dprices>=strike)[0])/len(dprices)
        if dc_prob>0:
            daily_call_probs.append((day,dc_prob))
        else:
            pass

        dp_prob = len(np.where(dprices<=strike)[0])/len(dprices)
        if dp_prob>0:
            daily_put_probs.append((day,dp_prob))
        else:
            pass


    print("Probability Report for Strike Price of {}".format(strike))
    print("-----------------------------------------------------------")
    print("Overall Probabilities:\n")
    print("Price above Strike: {}".format(call_prob))
    print("Price below Strike: {}\n".format(put_prob))
    print("-----------------------------------------------------------")
    print("Daily Probabilities Where Close is Greater Than Strike\n")

    for x in daily_call_probs:
        print("Day {}: {}".format(x[0],x[1]))

    print("-----------------------------------------------------------")
    print("Daily Probabilities Where Close is Less Than Strike\n")

    for x in daily_put_probs:
        print("Day {}: {}".format(x[0],x[1]))

    print("-----------------------------------------------------------")
    print("\nEnd of Report")











