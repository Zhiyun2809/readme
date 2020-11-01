import pandas as pd
import researchpy as rp
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp

# ====================================================
# Assumption check 
# ====================================================
# other ways
# Homogeneity of Variance (Variance of the two or more samples are equal)
# Levene Test for samples from significantly non-normal population
stats.levene(df['libido'][df['dose'] == 'placebo'],
            df['libido'][df['dose'] == 'low'],
            df['libido'][df['dose'] == 'high'])
# Bartlett test  equal variance check
stats.bartlett(df['libido'][df['dose'] == 'placebo'],
            df['libido'][df['dose'] == 'low'],
            df['libido'][df['dose'] == 'high'])

stats.levene(setosa['petal_length'],virginica['petal_length'])
## if the homogeneity of variance is violated-> go to Welch's t-test
# Normality (results of linear regression) (check if data follows normal distribution)
# normaliy tests are used to determine if a data set is well-modeled by a normal distribution and to compute how likely it is a random variable underlying the dataset to be normally distributed. 
# Shapiro-Wilk test
stats.shapiro(results.resid)
results.diagn

setosa = df[(df['species']=='Iris-setosa')]
virginica = df[(df['species']=='Iris-virginica')]
# check mornality of the dataset
stats.shapiro(setosa)
stats.shapiro(virginica)

# ====================================================
# T-test (pairwise) using stats (normal t test)
#calculate T-test for the means of two independent samples os scores
# t-test with equal variants
stats.ttest_ind(setosa['petal_length'],virginica['petal_length'])

# Welch-test test  t-test with unequal variants
stats.ttest_ind(setosa['petal_length'],virginica['petal_length'],equal_var = False)
stats.ttest_ind(x_,y_,equal_var=False)

# T-test using researchpy
descriptives,results = rp.ttest(df['libido'][df['dose'] == 'high'],)df['libido'][df['dose'] == 'low'],
results


# ====================================================
## T-Test Welch's t-test
# ====================================================
# check homogeneity of variance Levene's test


# ====================================================
# F test 
# ====================================================
# extremely sensitive to non-normality.
p_value = scipy.statis.f.cdf(F,df1,df2)

# ====================================================
# One way ANOVA 
# ====================================================

# get summary statistics
rp.summary_cont(df['libido'])
rp.summary_cont(df['libido'].groupby(df['dose']))

rp.summary_cont(df.groupby(['Fert','Water']))['Yield']

# oneway ANOVA
stats.f_oneway(df['libido'][df['dose']=='high'],
                df['libido'][df['dose']=='low'],
                df['libido'][df['dose']=='placebo'])

# or using linear regression
results = ols('libido ~C(dose)',data=df).fit()
results.summary()
# to test between the group
# ANOVA table
aov_table = sm.stats.anova_lm(results,typ=2)
aov_table


# ====================================================
# Two way ANOVA 
# construct a combination of components
# ====================================================
# 
rp.summary_cont(df.groupby(['Fert','Water']))['Yield']
# test interaction term 
model = ols('Yield ~C(Fert)*C(Water)',data=df).fit()
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
model.summary()
# create anova table (typ 2 sum of squares)
res= sm.stats.anova_lm(model,typ=2)
print(res)

# 
model2 = ols('Yield ~C(Fert)+C(Water)',df).fit()
model2.summary()
#get anova table
res2 = sm.stats.anova_lm(model2, typ= 2)
print(res2)
# post-hoc testing
mc = statsmodels.stats.multicomp.MultiComparison(df['Yield'],df['Fert'])
mc_results = mc.tukeyhsd()
print(mc_results)
mc = statsmodels.stats.multicomp.MultiComparison(df['Yield'],df['Water'])
mc_results = mc.tukeyhsd()
print(mc_results)

# 
formula = 'volume ~ C(keywork) + C(engine) + C(month) + C(keyword):C(engine) 
        + C(engine):C(month)'
lm = ols(formula, df).fit()
lm.summary()
table = sm.stas.anova_lm(lm,typ=2)
print(table)

# ====================================================
# multi
# Tukeys multi-comparison method
# Tukeys hsd (honesly significant difference) test
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
MultiComp = MultiComparison(df['result'],df['log'])
# show all pair-wise comparisons
print(MultiComp.tukeyhsd().summary())

from statismodels.stats.multicomp import pairwise_tukeyhsd
m_comp = pairwise_tukeyhsd(endog=df['volume'],groups=df['keyword'], alpha=0.05)
print(m_comp)
tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:],columns=m_comp._results_table.data[0])
group1_comp = tukey_data[tukey_data['reject']==True].groupby('group1')

# ====================================================
# Holm-Bonferroni Method
comp = MultiComp.allpairtest(stats.ttest_rel,method='Holm')
print(comp[0])

# ====================================================
# QQ-plot
import numpy as np
import statsmodels.api as sm
a = np.random.normal(5,5,250)
sm.qqplot(a)
plt.show()

import scipy
import matplotlib.pyplot as plt
scipy.stts.probplot(a,dist='norm',plt=plt)

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_qqplot as sqp
iris = sns.load_dataset('iris')
sqp.qqplot(iris,x='sepal_length',y='petal_length',height=4,aspec=1.5)

# ====================================================
# statsmodels

import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_precdiction_std
# list all method
dir(smf)
# 
res = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df,missing='drop' ).fit()
prstd,iv_l,iv_u = wls_prediction_std(res)
ax.plot(x,res.fittedvalues,'r--.',label='OLS')
ax.plot(x,iv_u,'r--')
ax.plot(x,iv_l,fittedvalues,'r--')
# view parameters
print(res1.params)
print(res1.summary())
print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('Predicted values: ', res.predict())

# use Region as Categorical variables
m1 = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df,missing='drop' ).fit()
# remove intercept from a model
m1 = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region) -1', data=df,missing='drop' ).fit()


# The * means that we want to interaction term in addition each term seperately
m1 = smf.ols(formula='Lottery ~ Literacy * Wealth + C(Region)', data=df,missing='drop' ).fit()
# The : means that we want only interaction term 
m1 = smf.ols(formula='Lottery ~ Literacy : Wealth + C(Region)', data=df,missing='drop' ).fit()


x = pd.DataFrame({'lstat':np.linspace(df['lstat'].min(),df['lstat'].max(),100)})
m = smf.ols(formula='medv ~ 1+lstat',data=df).fit()
plt.plot(x.lstat,m.predict(x),label='Poly $R^2$=%.2f'%m.rsquared,alpha=0.9)

# 1-st Order polynomial
poly_1 = smf.ols(formula='medv ~ 1 + lstat', data=df).fit()
# 2-st Order polynomial
poly_2 = smf.ols(formula='medv ~ 1 + lstat + I(lstat ** 2.0)', data=df).fit()
# 3-st Order polynomial
poly_3 = smf.ols(formula='medv ~ 1 + lstat + I(lstat ** 2.0) + I(lstat ** 3.0)', data=df).fit()

