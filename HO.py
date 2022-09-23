'''
Harmonic Oscillator Model v2.0
'''

#Startup Code Begin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import lognorm,t
import itertools
import os
#Startup Code End


class HO(object):
    def __init__(self,ticker):
        '''
        :type ticker: str
        :param ticker: ticker symbol for the stock

        :type market_cap: float
        :param market_cap: market cap for the stock ***in Billions*** -> eg. 16e9 == 16

        :type log_returns: ndarray
        :param log_returns: log returns (closing prices) for the stock

        :type closing_price: ndarray
        :param closing_prices: closing prices for the stock

        :type params: dict
        :param params: the inputs for the wavefunctions

        :type wavefunctions: dict
        :param wavefunctions: {elevel:(wf,pdf,df)...}

        :type cdfs: dict
        :param cdfs: {elevel:(xvalues,yvalues),elevel1-5:[(x1,y1),(x2,y2)...]}
        '''
        self.ticker = ticker.upper()
        self.close,self.log_returns,self.params = self.get_paramaters()
        self.cdfs = self.create_cdfs()
        self.ppts = self.create_ppts()

    # @classmethod
    # def create_model(cls,ticker):
    #     #Load Volatility Values and Data
    #     ticker=ticker.upper()
    #     fname = r'E:/Investing/historical_data/{}.csv'.format(ticker)
    #     data = pd.read_csv(fname,index_col='Date')
    #     data['lr_sma'] = np.log(1+data.)
        
    #     model = cls(ticker,mcap)
    #     return(model)


    def get_paramaters(self):
        #load data
        fname = r'E:/Investing/tenm/{}.csv'.format(self.ticker)
        data = pd.read_csv(fname,index_col='Time')
        days = sorted(np.unique(data.Day.values),key=lambda x:x.split('/')[2])

        #Get log returns and volatility
        lr = [np.log(1+data.loc[data.Day==x].Close.pct_change()).dropna().values for x in days]
        log_returns = np.array([x[:38] for x in lr if len(x)>37])
        volatility = log_returns.std(axis=1)*np.sqrt(38.0)

        vfit = lognorm.fit(volatility)

        #Get kf
        fx = lambda x,k: .5*k*x**2
        v = np.array([[x]*38 for x in volatility])
        kf = curve_fit(fx,np.concatenate(log_returns),np.concatenate(v))[0][0]
        

        #get params
        #Set Constants
        amu = 1.66054e-27       #kg  
        hbar = 1.0545718e-34     #J*s

        #Get Omega (w) and Mass (m)
        v0 = min(volatility)
        w = (2*v0)/hbar #V = .5*hw
        m = kf/w**2

        hw = hbar*w

        #Get T-Fit for V0
        ind = np.where(volatility<=.5*hw)[0]
        tfit = t.fit(np.concatenate(log_returns[ind]))

        alpha = np.sqrt(hbar/(m*w))

        params = dict(m=m,w=w,a=alpha,kf=kf,hw=hw,v_dist=vfit,tfit=tfit)

        return(data.Close.values,log_returns,params)

    def wavefunction(self,elevel):

        '''
        :type: params: dict
        :param params: key:value pairs -> energy = dE,kf = force constant,m=mass,w = omega

        :type num_levels: int
        :param num_levels: number of energy levels -> matches get_paramaters
        '''
        #Set Constants
        hbar = 1.0545718e-34     #eV*s
        omega = self.params['w']
        kf = self.params['kf']
        m = self.params['m']
        alpha = self.params['a']

        #Define the first five Hermite polynomials
        H0 = 1
        H1 = lambda y: 2*y
        H2 = lambda y: 4*y**2 - 2
        H3 = lambda y: 8*y**3 - 12*y
        H4 = lambda y: 16*y**4 - 48*y**2 + 12
        H5 = lambda y: 32*y**5 - 160*y**3 + 120*y

        hlist = [H0,H1,H2,H3,H4,H5]

        #Define Normalization Constant
        Nv = lambda v: ((m*omega)/(np.pi*hbar))**(1/4) * (1/(np.sqrt(2**v*np.math.factorial(v))))
        N = Nv(elevel)
        if elevel==0:
            wf = lambda x: alpha*N*np.exp(-x**2/2.)
            pdf = lambda x: N**2 * np.exp(-x**2)
            fx = lambda x: alpha**2 * Nv(0)**2 * np.exp(-x**2)
            
        elif elevel==1:
            wf = lambda x: N*hlist[elevel](x/alpha)*np.exp(-(x/alpha)**2/2)
            pdf = lambda x: 4*N**2*x**2*np.exp(-x**2/alpha**2)/alpha**2
            fx = lambda x: 4*N**2*x**2*np.exp(-x**2/alpha**2)/alpha
        
        elif elevel==2:
            wf = lambda x: N*hlist[elevel](x/alpha)*np.exp(-(x/alpha)**2/2)
            pdf = lambda x: 4*N**2*(alpha**2 - 2*x**2)**2*np.exp(-x**2/alpha**2)/alpha**4
            fx = lambda x: 4*N**2*(alpha**2 - 2*x**2)**2*np.exp(-x**2/alpha**2)/alpha**3
        
        elif elevel==3:
            wf = lambda x: N*hlist[elevel](x/alpha)*np.exp(-(x/alpha)**2/2)
            pdf = lambda x: 16*N**2*x**2*(3*alpha**2 - 2*x**2)**2*np.exp(-x**2/alpha**2)/alpha**6
            fx = lambda x: 16*N**2*x**2*(3*alpha**2 - 2*x**2)**2*np.exp(-x**2/alpha**2)/alpha**5
        
        elif elevel==4:
            wf = lambda x: N*hlist[elevel](x/alpha)*np.exp(-(x/alpha)**2/2)
            pdf = lambda x: 16*N**2*(3*alpha**4 - 12*alpha**2*x**2 + 4*x**4)**2*np.exp(-x**2/alpha**2)/alpha**8
            fx = lambda x:  16*N**2*(3*alpha**4 - 12*alpha**2*x**2 + 4*x**4)**2*np.exp(-x**2/alpha**2)/alpha**7
        
        elif elevel==5:
            wf = lambda x: N*hlist[elevel](x/alpha)*np.exp(-(x/alpha)**2/2)
            pdf = lambda x: 64*N**2*x**2*(15*alpha**4 - 20*alpha**2*x**2 + 4*x**4)**2*np.exp(-x**2/alpha**2)/alpha**10 
            fx = lambda x: 64*N**2*x**2*(15*alpha**4 - 20*alpha**2*x**2 + 4*x**4)**2*np.exp(-x**2/alpha**2)/alpha**9
            
        else:
            pass
        

        return(wf,pdf,fx)


    def create_cdfs(self):
        distributions = {}
        x1 = (self.params['hw']*(5+.5))*2.
        xr = np.linspace(-x1,x1,50000)
        
        for i in range(1,6):
            
            #find where prob density != 0
            pdist = self.wavefunction(i)[1](xr)
            ind = np.where(np.diff(pdist)>0)[0]
            ind2 = np.where(np.diff(ind)!=1)[0]
            
            peaks = [ind[x] for x in ind2]+ind[-1]
            peaks = np.asarray(peaks)
            
            if len(peaks)<i+1:
                print("Something went wrong at level {}: too many peaks!!!".format(i))
            
            num_zeros = len(peaks)-1
            z_ind = []
            for i in range(num_zeros-1):
                xmin,xmax = peaks[i],peaks[i+1]
                z = np.argmin(pdist[xmin:xmax])+xmin
                z_ind.append(z)
            
            
            dist_ranges = [xr[:z_ind[i]]]
            for i in range(len(z_ind)-1):
                dist_ranges.append(xr[z_ind[i]:z_ind[i+1]])
            dist_ranges.append(xr[z_ind[-1]:])
                
           
            #Create Distribution
            c_dist = []
            last_max = 0
            for j in range(len(dist_ranges)):
                a = dist_ranges[j][0]
                temp=[]
                for k in dist_ranges[j]:
                    temp.append(quad(self.wavefunction(i)[2],a,k)[0])
                c_dist.append(np.array(temp)+last_max)
                last_max += temp[-1]

            dist = list(zip(dist_ranges,c_dist))

            distributions[str(i)] = dist

        return(distributions)


    def create_ppts(self):
        '''
        :type cdfs: list of tuples
        :param cdfs: tuple containing the xvalues and distribution x[0]=x x[1]=dist
        '''

        lfit = lambda x,A,k,x0: A/(1+np.exp(-k*(x-x0)))
        lfit2 = lambda x,A,k,x0,b: A/(1+np.exp(-k*(x-x0))) + b

        fit_dict = {}
        for i in range(1,6):
            fit_list = []
            current_cdf = self.cdfs[str(i)]
            for j in range(len(current_cdf)):
                xr = current_cdf[j][0]
                cd = current_cdf[j][1]
                
                pdxr = self.wavefunction(i)[1](xr)
                mean_index = np.argmax(pdxr)
                sig_mean = xr[mean_index]
                amplitude = max(cd)-min(cd)
                
                if j==0:
                    popt,pcov = curve_fit(lfit,xr,cd,p0=[amplitude,1.0,sig_mean])
                    fit_list.append(popt)
                else:
                    offset = max(cd) - amplitude
                    popt,pcov = curve_fit(lfit2,xr,cd,p0 = [amplitude,1.0,sig_mean,offset])
                    fit_list.append(popt)

            fit_dict[str(i)] = fit_list

        return(fit_dict)

    def predict_return(self,num_paths,volatility,level):
        quantile = lambda y,A,k,x0: np.log((y*np.exp(k*x0))/(A-y))/k
        quantile2 = lambda y,A,k,x0,b: np.log(((y-b)*np.exp(k*x0))/(A+b-y))/k

        prediction = []
        if level==0:
            fit = self.params['tfit']
            draw = t.rvs(*fit,size=39)    #39 Assumes 10 Minute Data
            prediction.append(np.exp(draw))

        else:
            fit_list = self.ppts[str(level)]
            returns = []
            plist = [0]
            for x in self.cdfs[str(level)]:
                plist.append(x[1][-1])
            p_ranges = []
            for i in range(1,len(plist)):
                p_ranges.append((plist[i-1],plist[i]))
            
            choices = np.concatenate([x[1] for x in self.cdfs[str(level)]])
            np.random.shuffle(choices)
            draw = np.random.choice(choices,size=39)
            sample = []

            for i in range(len(p_ranges)):
                if i==0:
                    rs = draw[draw<=p_ranges[0][1]]
                    fit = fit_list[0]
                    sample.append(volatility*quantile(rs,*fit))
                else:
                    rs = draw[(draw>p_ranges[i][0])&(draw<=p_ranges[i][1])]
                    fit = fit_list[i]
                    sample.append(quantile2(rs,*fit))
                
            prediction.append(np.exp(np.concatenate(sample)))

        return(prediction)



        prediction = np.exp(np.concatenate(prd_returns))
        np.random.shuffle(prediction)
        prediction = prediction.reshape(time_steps,paths)

        s0 = self.close[-1]
        plist = np.vstack([np.ones(paths),prediction])
        prd_prices = s0*np.cumprod(plist,axis=0)

        #Get last price
        
        return({'preturns':prediction,'pprices':prd_prices})



    def run_simulation(self):
        num_paths = int(input("How many paths to simulate? "))
        time_steps = int(input("How many days to forecast? "))
        s0 = float(input("What was the last closing price?: "))
        
        #Get Volatility Info and Levels
        hw = self.params['hw']
        vlevels = np.array([hw*(x+.5) for x in range(6)])
        vdist = self.params['v_dist']   #Distribution of past daily volatility

        prediction = []
        for i in range(time_steps):
            vols = lognorm.rvs(*vdist,size=num_paths)
            prd_returns = []
            for v in vols:
                for j in range(6):
                    if j<5:
                        if v<=vlevels[j]:
                            level = j
                            break
                        else:
                            pass
                    else:
                        level = 5   #current volatility greater than any past volatility

                prd_returns.append(self.predict_return(num_paths=num_paths,volatility=v,level=level))
            prediction.append(np.vstack(prd_returns))

        time_series = np.concatenate(prediction,axis=1).T


        returns = np.vstack((np.ones(num_paths),time_series))
        price_prediction = s0*returns.cumprod(axis=0)

        #Get Probability Cone
        sp = np.sort(price_prediction,axis=1)
        pcts = [.05,.25,.75,.95]
        prob_cone = np.array([sp[:,int(x*num_paths)] for x in pcts])

        save_fname = r'E:/Investing/Saved_HO_Simulations/{}_{}.csv'.format(self.ticker,datetime.today().strftime('%Y-%m-%d'))
        np.savetxt(save_fname,prob_cone,delimiter=',')

        return(price_prediction,prob_cone)







#------------------------------------------------------------------------------------------------------------------------------------#
#Static Methods
def plot_forecast(ticker):
    f = os.listdir(r'E:/Investing/Saved_HO_Simulations')
    fname = [x for x in f if ticker.upper() in f]
    if len(fname)>1:
        print("Two many files")
    else:
        pass

    fpath = r'E:/Investing/Saved_HO_Simulations/{}'.format(fname)

    #Load Data
    prob_cone = np.loadtxt(fpath,delimiter=',')

    #Load past n days of daily data
    date = fname.split('_')[1].split('.')[0]
    start = '{}-{}-{}'.format(date[0],date[1],date[2])
    end = datetime.today().strftime('%Y-%m-%d')

    past_data = web.DataReader(ticker.upper(),'yahoo',start,end)
    observed = past_data.Close.values

    #Plot the data
    xr1 = np.arange(len(prob_cone[0]))  #All prob cones are the same length
    [plt.plot(x,color='r',linewidth=2) for x in prob_cone]
    for i in range(3):
        if i==1:
            plt.fill_between(xr1,prob_cone[i],prob_cone[i+1],color='b',alpha=.5)
        else:
            plt.fill_between(xr1,prob_cone[i],prob_cone[i+1],color='r',alpha=.5)

    xr2 = np.arange(len(observed))*39
    plt.plot(xr2,observed,color='k')

    plt.grid(True)
    plt.xticks(ticks=np.arange(5)*39,labels=np.arange(5)*39)
    plt.title("Price Forecast for {} from {} to {}".format(ticker.upper(),start,end))

    plt.show()



    



def test_fit(model):  
    lfit = lambda x,A,k,x0: A/(1+np.exp(-k*(x-x0)))
    lfit2 = lambda x,A,k,x0,b: A/(1+np.exp(-k*(x-x0))) + b
    fy = lambda y: np.sqrt(2*y/model.params['kf'])

    while True:
        elevel = input("What Energy Level to Test or Q to Quit? ")
        xr1 = model.params['hw']*(int(elevel)+.5)
        xrr = np.linspace(-fy(xr1),fy(xr1),50000)
        xr = np.linspace(-xr1,xr1,50000)/model.params['a']

        if elevel=='0':
            lr = np.concatenate(model.log_returns)
            plt.hist(lr,bins='auto',density=True)
            fit = model.params['tfit']
            xr = np.linspace(min(lr),max(lr),500)
            plt.plot(xr,t.pdf(xr,*fit),color='r')
            plt.show()


        elif elevel=='1':
            n=int(elevel)
            psi1,pdf1 = model.wavefunction(n)[:2]
            cd1 = model.cdfs[elevel]
            cf1 = model.ppts[elevel]

            cdx1 = cd1[0][0]
            cdd1 = cd1[0][1]

            cdx2 = cd1[1][0]
            cdd2 = cd1[1][1]


            fig,(ax1,ax2,ax3) = plt.subplots(3)
            fig.suptitle("{}: Level 1".format(model.ticker))
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
            n=int(elevel)
            psi2,pdf2 = model.wavefunction(n)[:2]
            cd2 = model.cdfs[elevel]
            cf2 = model.ppts[elevel]

            cdx1 = cd2[0][0]
            cdd1 = cd2[0][1]
            cdx2 = cd2[1][0]
            cdd2 = cd2[1][1]
            cdx3 = cd2[2][0]
            cdd3 = cd2[2][1]

            fig,(ax1,ax2,ax3,ax4) = plt.subplots(4)
            fig.suptitle("{}: Level 2".format(model.ticker))
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
            n=int(elevel)
            psi3,pdf3 = model.wavefunction(n)[:2]
            cd3 = model.cdfs[elevel]
            cf3 = model.ppts[elevel]

            cdx1 = cd3[0][0]
            cdd1 = cd3[0][1]
            cdx2 = cd3[1][0]
            cdd2 = cd3[1][1]
            cdx3 = cd3[2][0]
            cdd3 = cd3[2][1]
            cdx4 = cd3[3][0]
            cdd4 = cd3[3][1]

            fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5)
            fig.suptitle("{}: Level 3".format(model.ticker))
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
            n=int(elevel)
            psi4,pdf4 = model.wavefunction(n)[:2]
            cd4 = model.cdfs[elevel]
            cf4 = model.ppts[elevel]

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
            fig.suptitle("{}: Level 4".format(model.ticker))
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
            n=int(elevel)
            psi5,pdf5 = model.wavefunction(n)[:2]
            cd5 = model.cdfs[elevel]
            cf5 = model.ppts[elevel]

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
            fig.suptitle("{}: Level 5".format(model.ticker))
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

        elif elevel=='q' or elevel=='Q':
            break




