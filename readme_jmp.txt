# Sample data folder
C:\Program Files\SAS\JMP\15\Samples\Data

# Confidence interval of mean
Jmp->Help->Sample Data->Teaching resources->Calculators->Confidence Interval
for one Mean

# calucate Confidence Interval
load data->Analyze->Distribution->(add columns to) [Y,Columns] -> OK
(change confidence interval) [Click column name]->Confidence Interval

# prediction intervals: range of plausible values for the next observation, or
next n obervations
load data->Analyze->Distribution->(add columns to) [Y,Columns] -> OK
(Prediction interval) [Click column name]->Prediction Interval

# Tolerance intervals: Range of values for a specified proportion of
observations with a certain degree of confidence. Eg. We are 95% confident
that at least 90% of future values will fall between 15 and 25
# to specify the tolerance we need: confidence level (1-\alpha) AND
proportion of the population to include
(Tolerance interval) [Click column name]->Prediction Interval

# Sample size and Power
[Main]->DOE->Design Diagnostics->Sample Size and Power

# one sample T test
Analyze->Distribution->Data
[Data Column]->Test Mean
[Data Column]->Test Mean->PValue animation

# two sample T test
Analyzer->Fit Y by X
# get CI
[]->Means and Std Dev 
# unpooled T-Test (non equal variance)
[]->T test
# pooled T-Test (equal variance)
[]->Means/Anova/Pooled t
(Select data column)->Display Options->Points Jittered

# paired T-test
Analyze->Specilized modeling->matched Pairs

# Oneway ANOVA
Analyze->Fit Y by X->[]->Means/Anova
[]->Compare Means->
# Welch Anova Test
# test variance 
[]->Unequal variance

# Two way ANOVA
Analyze->Fit Model

# Non parametric test
Analyze->Distribution->->Fit Y by X->Test Mean->[tick] Wilcoxon Signed Rank

# Correlation and Regression
# load data
-> Graph-> Graph Builder->(choose X and Y)-> ellipse -> [tick] Correlation
->Analyze->Fit X Y->[]->Density Ellipse
-> Analyze->Multivariate Methods-> Multivariate ->[] Density Ellipse

# Linear Regression
-> Analyze->Fix Y by X->Fit Line->[Linear Fit]->Plot Residual
-> Analyze->Fix Y by X->Fit Line->[Linear Fit]->Confid Curves Fit

# Polynomial Regression

# Fit model
-> Analyze->Fit Model -> Pick Role Variables [Y] (target) -> Construct Model
Effects [Add] -> Personality: [Standard Least Squares] -> [Run] 
	-> []->Effect Summary
# use Prediction Profiler
	[]->Factor Profiling->Profiler
# Studentized Residual
	->[]->Row Diagnostics->Plot Studentized Residuals
# Cook's D  (Outlier indicator)
	->[]->Save Columns->Cook's D Influence
# show prediction expression
	->[]->Estimates->Show Prediction Expression
# profiler
	->[]->Factor Profiling->Profiler
# Catagorigal value regression -> effect Coding (default JMP)-1,1
# Catagorigal value regression -> dummy Coding 0,1
	->[]->Estimates->Indicator Parameterization Estimates

# Fitting model with interactions
Analyze->Fit Model->Construct Model Effects->[Marcos]->Factorial to degree
	->[]->Regression Reports->Analysis of Variance
	->[]->Factor Profiling ->Interaction Plots

# Variable selection

# Akaike Information Criterion (AIC)
# Bayesian Information Criterion (BIC)

# VAriance INflation Factor VIF -> measure of how much the standard error of
the estimate of the coefficient is inflated due to multicollinearity
Analyze->Fit Model->Parameter Estimates->[right click]->columns->VIF


# Logistic Regression
Analyze->Fit Y by X (Y shall be category data) ->Target Level (can be changed)
	Logit is Parameter Estimaters
	[]->Save Probability Formula 
	Fit Details->Misclassification Rate


# interaction
Analyze->Fit model ->[] ->Factor Profilling->Interaction Plots
