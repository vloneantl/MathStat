import numpy as np
import scipy.stats
import math
import statsmodels.stats.weightstats as st
import matplotlib.pyplot as plt

#генерация случайных величин N(5,3); N(5,1)
mu1 = 5
mu2 = 5
sigma1 = 3
sigma2 = 1
n1 = 250
n2 = 250
alpha = 0.05

np.random.seed(1)
X1 = []
for _ in range(n1):
    x1 = sigma1 * scipy.stats.norm.rvs() + mu1
    X1.append(x1)
X1 = np.array(X1)
print('X1',scipy.stats.describe(X1))

np.random.seed(2)
X2 = []
for _ in range(n2):
    x2 = sigma2 * scipy.stats.norm.rvs() + mu2
    X2.append(x2)
X2 = np.array(X2)
print('X2',scipy.stats.describe(X2))

#2 Однопараметрические критерии
print()
def ztest(X, m, sigma):
    z = (X.mean() - m)/(sigma / math.sqrt(X.size))
    p = 2 * min(scipy.stats.norm.cdf(z), 1- scipy.stats.norm.cdf(z))
    return z, p

print('Z-test:  statistic',ztest(X1, mu1, sigma1)[0],'pvalue',ztest(X1,mu1, sigma1)[1])  #ztest H0: m1 = mu1
print('T-test:  statistic',scipy.stats.ttest_1samp(X1,mu1)[0],'pvalue', scipy.stats.ttest_1samp(X1,mu1)[1])  #ttest_1samp H0: m1 = mu1
# t-test для alpha = 0.05 и df= 250   == 1.651

def chisquaretest(X1,sigma, mu = None):
    def sigma0(X1, mu):
        s = 0
        for item in X1:
            s += (mu - item)**2
        return math.sqrt(s/(X1.size-1))

    n = X1.size
    if mu is None:
        z = (n-1) * X1.var() / (sigma ** 2)
        p = 2 * min(scipy.stats.chi2.cdf(z, n-1),1-scipy.stats.chi2.cdf(z, n-1))
    else:
        z = n * (sigma0(X1,mu)**2) / (sigma ** 2)
        p = 2 * min(scipy.stats.chi2.cdf(z, n - 1), 1 - scipy.stats.chi2.cdf(z, n - 1))
    return z, p

print('chisquare test m-не изв.',chisquaretest(X1,sigma1))
print('chisquare test m-изв.',chisquaretest(X1,sigma1,mu1))

#3 Двухвыборочные критерии
print()
print('2-sample t-test',scipy.stats.ttest_ind(X1,X2)) #H0: m1 = m2

def ftest(X1, X2, m= None):
    def sigma(X1, mu):
        s = 0
        for item in X1:
            s += (mu - item)**2
        return s/(X1.size-1)

    if m is None:
        f = X1.var()/ X2.var()
        df1 = X1.size - 1
        df2 = X2.size - 1
        p = 2 * min(1 - scipy.stats.f.cdf(f, df1, df2),scipy.stats.f.cdf(f, df1, df2))
    else:
        f = sigma(X1, m)/ sigma(X2, m)
        df1 = X1.size - 1
        df2 = X2.size - 1
        p = 2 * min(1 - scipy.stats.f.cdf(f, df1, df2),scipy.stats.f.cdf(f, df1, df2))
    return f, p

print('f test (м не изв.): statistic',ftest(X1,X2)[0],'pvalue',ftest(X1,X2)[1])
print('f test (м изв.): statistic',ftest(X1,X2, mu1)[0],'pvalue',ftest(X1,X2, mu1)[1])


#3 исследование распределение статистик критерия
#H0: m1=m2      sigma1, sigma2 известно

def zstat(X1, X2, sigma1, sigma2):
    z = (X1.mean() - X2.mean()) / math.sqrt((sigma1**2) / X1.size + (sigma2**2) / X2.size)
    p = scipy.stats.norm.sf(abs(z))
    return z, min(p, 1-p) * 2


print(zstat(X1, X2, sigma1, sigma2))

zlist = []
plist = []

for _ in range(1000):
    X3 = []
    for _ in range(n1):
        x = sigma1 * scipy.stats.norm.rvs() + mu1
        X3.append(x)
    X4 = []
    for _ in range(n2):
        x = sigma2 * scipy.stats.norm.rvs() + mu2
        X4.append(x)
    X3 = np.array(X3)
    X4 = np.array(X4)
    z, p = zstat(X3, X4, sigma1, sigma2)
    zlist.append(z)
    plist.append(p)
zlist = np.array(zlist)
plist = np.array(plist)
print(scipy.stats.describe(zlist))
print(scipy.stats.describe(plist))

plt.hist(zlist, density=True, bins= 10, label='Гистограмма частот статистики Z')
x_axis = np.arange(-2,2,0.0000025)
plt.plot(x_axis, scipy.stats.norm.pdf(x_axis, 0, 1), label='Теоретическая функция Fz(Z|H0)')
plt.legend()
plt.show()
x_axis = np.arange(0,1)
plt.hist(plist, density=True, label='Гистограмма частот статистики P-value')
plt.plot([0.,1.],[1.,1.], label = 'Теоретичкская функция Fp(p|H0)')
plt.legend()
plt.show()

print(scipy.stats.describe(np.hstack((X1,X2))))