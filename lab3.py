import numpy as np
import scipy.stats
import math
import itertools
import matplotlib.pyplot as plt
import statsmodels.stats.multicomp
import statsmodels.stats.descriptivestats

#генерация выборок N(5,3); N(5,1)
mu1 = 5
mu2 = 5
mu3 = 5
mu4 = 5
sigma1 = 3
sigma2 = 1
sigma3 = 5
sigma4 = 5
n1 = 200
n2 = 250
n3 = 200
n4 = 200
alpha = 0.05

np.random.seed(1)
x1 = scipy.stats.norm.rvs(loc=mu1,scale=sigma1,size=n1)
np.random.seed(2)
x2 = scipy.stats.norm.rvs(loc=mu2,scale=sigma2,size=n2)
np.random.seed(4)
x3 = scipy.stats.norm.rvs(loc=mu3,scale=sigma3,size=n3)
np.random.seed(7)
x4 = scipy.stats.norm.rvs(loc=mu4,scale=sigma4,size=n4)
pooled = np.concatenate((x1,x2,x3,x4))
print('x1 :',scipy.stats.describe(x1))
print('x2 :',scipy.stats.describe(x2))
print('x3 :',scipy.stats.describe(x3))
print('x4 :',scipy.stats.describe(x4))
print('pooled :',scipy.stats.describe(pooled))
print()


plt.boxplot((x1,x2, x3, x4), labels=('x1: mu=5 sigma=3','x2: mu=5 sigma=1','x3: mu=5 sigma=5','x4: mu=5 sigma=5'))
plt.show()


#3 проверка условия применимости дисперсионного анализа
# H0: sigma1^2=...=sigma4^2
#критерий бартлетта
print('3. Критерий Бартлетта:',scipy.stats.bartlett(x1,x2,x3,x4))

#4 Однофакторный дисперсионный анализ
print('общее среднее',scipy.stats.describe(pooled)[2])
print('общая дисперсия',scipy.stats.describe(pooled)[3])
Dw = ((n1-1)*scipy.stats.describe(x1)[3]+
      (n2-1)*scipy.stats.describe(x2)[3]+
      (n3-1)*scipy.stats.describe(x3)[3]+
      (n4-1)*scipy.stats.describe(x4)[3]) \
     / (scipy.stats.describe(pooled)[0]-1)
Db = (n1*(scipy.stats.describe(x1)[2]-scipy.stats.describe(pooled)[2])**2+
      n2*(scipy.stats.describe(x2)[2]-scipy.stats.describe(pooled)[2])**2+
      n3*(scipy.stats.describe(x3)[2]-scipy.stats.describe(pooled)[2])**2+
      n4*(scipy.stats.describe(x4)[2]-scipy.stats.describe(pooled)[2])**2) \
     / scipy.stats.describe(pooled)[0]
print('Dw+Db', Dw+Db)
print('Внутригрупповая дисперсия',Dw)
print('Межгрупповая дисперсия',Db)
nusquare = Db / (Db + Dw)
print('ЭКД =',nusquare)
print('ЭКО =',math.sqrt(nusquare))
print()
print(scipy.stats.f_oneway(x1,x2,x3,x4))
print()


intervals = []
for x in [x1,x2,x3,x4]:
    t = scipy.stats.t.ppf(0.975,df=len(x))
    s = math.sqrt(scipy.stats.describe(x)[3])
    intervals.append(t*s/math.sqrt(len(x)))
print(intervals)
mean_list = [x1.mean(),x2.mean(),x3.mean(),x4.mean()]
plt.errorbar(x = mean_list, y = range(1,5,1), fmt='o', xerr=intervals)
plt.axvline(5.,ls='--', color='y', label='m=5')
plt.legend(loc=3)
plt.xlabel('x')
plt.ylabel('номер случайной величины')
plt.show()

def tukey_hsd (lst, ind, n):
    data_arr = np.hstack (lst)
    ind_arr = np.repeat (ind, n)
    print(statsmodels.stats.multicomp.pairwise_tukeyhsd (data_arr, ind_arr))
tukey_hsd ((x1,x2,x3,x4), list('ABCD'), (n1,n2,n3,n4))


for x, y in itertools.combinations([x1,x2,x3,x4], 2):
    print(x.mean(), y.mean(), end=' ')
    cv = scipy.stats.describe(y)[2] - scipy.stats.describe(x)[2]
    sigmav = (Dw * len(pooled) / (len(pooled)-4))  * (1/len(x) + 1/len(y))
    f = scipy.stats.f.ppf(0.95, dfn=3,dfd=len(pooled))
    conf_inv = cv - math.sqrt(sigmav*3*f), cv + math.sqrt(sigmav*3*f)
    print(conf_inv)

