import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt
import statsmodels.distributions
import statsmodels.stats.descriptivestats

#генерация выборок N(5,3); N(5,1)
mu1 = 5
mu2 = 5
sigma1 = 3
sigma2 = 1
n1 = 250
n2 = 250
alpha = 0.05

bins = 15

np.random.seed(1)
x = scipy.stats.norm.rvs(loc=mu1,scale=sigma1,size=n1)
np.random.seed(2)
y = scipy.stats.norm.rvs(loc=mu2,scale=sigma2,size=n2)

print(scipy.stats.describe(x))
print(scipy.stats.describe(y))
#plt.hist(x,bins=bins)
#plt.show()

def zstat(xp, yp, n):
    z = 0
    for i in range(len(xp)):
        z += ((xp[i]-yp[i])**2)/yp[i]
    z *= n
    p = 1 - scipy.stats.chi2.cdf(z, df=len(xp)-1)
    return z, p

def chi2(x,bins=5):
    a = min(x)
    b = max(x)
    h = (b-a)/bins
    arr = np.zeros(bins)
    intervals = [0]*(bins)
    c = a
    for i in range(bins):
        intervals[i] = (c, c+h)
        c = c + h
    for item in x:
        index = (item - a) // h
        if index == bins:
            arr[int(index)-1] += 1
        else:
            arr[int(index)] += 1
    arr /= len(x)
    return arr, intervals

def z(x, mu, sigma):
    return (x-mu) / sigma

#H0: X~N
x_obs, intervals = chi2(x,bins=bins)
x_exp = np.ones(bins)
for i in range(bins):
    if i == bins-1:
        x_exp[i] = 1 - scipy.stats.norm.cdf(z(intervals[i][0], mu1, sigma1))
    else:
        x_exp[i] = scipy.stats.norm.cdf(z(intervals[i+1][0],mu1,sigma1)) - scipy.stats.norm.cdf(z(intervals[i][0],mu1,sigma1))
print('x_exp', x_exp)
print('x_obs',chi2(x, bins))
print('H0: X~N', zstat(x_obs, x_exp,n1))
print('H0: X~N', scipy.stats.chisquare(x_obs, x_exp))
print()
plt.hist(x, bins=5, density=True)
plt.show()
plt.hist(x, bins=10, density=True)
plt.show()
plt.hist(x, bins=15, density=True)
plt.show()
plt.hist(x, bins=20, density=True)
plt.show()



#H0: X~R
x_expr = np.ones(bins)
x_expr /= bins
print('X~R ',x_expr)
print('H0: X~R', zstat(x_obs, x_expr, n1))
print('H0: X~R', scipy.stats.chisquare(x_obs, x_expr))
print()

#H0: X~chi^2(5)
x_expx = np.ones(bins)

for i in range(bins):
    if i == bins-1:
        x_expx[i] = 1 - scipy.stats.chi2.cdf(intervals[i][0], 5)
    else:
        x_expx[i] = scipy.stats.chi2.cdf(intervals[i+1][0], 5) - scipy.stats.chi2.cdf(intervals[i][0], 5)
print('X~chi^2(5)',x_expx)
print(zstat(x_obs,x_expx, n1))
print(scipy.stats.chisquare(x_obs,x_expx))
print()

#plt.show()

#4 kolmogorov test
print('norm',scipy.stats.kstest(x,'norm', args=(mu1,sigma1)))
print('uniform',scipy.stats.kstest(x, lambda x : scipy.stats.uniform.cdf(x, loc=min(x), scale=max(x))))
print('chi2',scipy.stats.kstest(x, lambda x : scipy.stats.chi2.cdf(x, df=5)))
print()

x_axis = np.arange(min(x),max(x),0.005)
plt.plot(x_axis, scipy.stats.norm.cdf(x_axis, loc=mu1, scale=sigma1), label = 'Нормальное распределение')
plt.plot(x_axis, scipy.stats.chi2.cdf(x_axis, 5), label = 'Хи-квадрат распределение')
plt.plot(x_axis, scipy.stats.uniform.cdf(x_axis, loc=min(x), scale=max(x)-min(x)), label = 'Равномерное распределение')
ecdfx = statsmodels.distributions.empirical_distribution.ECDF(x)
plt.plot(x_axis, ecdfx(x_axis), label = 'Эмпирическая функция распределения x')
plt.legend()
plt.show()

plt.hist(x, label = 'x')
plt.hist(y, label = 'y')
plt.legend()
plt.show()

ecdfy = statsmodels.distributions.empirical_distribution.ECDF(y)
plt.plot(x_axis, ecdfx(x_axis), label = 'Эмпирическая функция распределения x')
plt.plot(x_axis, ecdfy(x_axis), label = 'Эмпирическая функция распределения y')
plt.legend()
plt.show()


def chisquaretwosample(x, y, k):
    nx = x.size
    ny = y.size
    a = min(min(x), min(y))
    b = max(max(x), max(y))
    h = (b - a) / bins
    fx = np.zeros(k)
    fy = np.zeros(k)
    c = a
    for item in x:
        index = (item - a) // h
        if index == bins:
            fx[int(index) - 1] += 1
        else:
            fx[int(index)] += 1
    for item in y:
        index = (item - a) // h
        if index == bins:
            fy[int(index) - 1] += 1
        else:
            fy[int(index)] += 1
    z = 0
    for i in range(k):
        z += ((fx[i]/nx - fy[i]/ny)**2) / (fx[i] + fy[i])
    z *= nx * ny
    p = 1 - scipy.stats.chi2.cdf(z ,df=k-1)
    return z, p


y_obs, intervals = chi2(y, bins)
print(scipy.stats.chisquare(x,y))
print(chisquaretwosample(x,y,bins))
print('ks 2samp',scipy.stats.ks_2samp(x,y))
print('sign test',statsmodels.stats.descriptivestats.sign_test(x-y))#Fx(ksi)=Fy(ksi)
#print('wilcoxon sign test',scipy.stats.wilcoxon(x, y))
print('u-test',scipy.stats.ranksums(x, y))