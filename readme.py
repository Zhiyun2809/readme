from sklearn import precrocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

forcast_out = 5
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X=X[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

#===============================
#Regressor type
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

from sklearn import svm
regressor = svm.SVR(kernal='poly')
regressor = svm.SVC(gamma=0.001,C=100)
#===============================

regressor.fit(X_train, y_train)
y_prediction = regression.predict(X_test)

#linear regression
clf = LinearRegression(n_jobs=-1) # as many as possible
#svm SVM
clf = svm.SVR(kernel='poly') 
#
clf = svm.SVC(gamma=0.001,C=100)
#
clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                        ('knn', neighbors.KNeighborsClassifier()),
                        ('rfor',RandomForestClassifier())])
# train the model
clf.fit(X_train, y_train)


#dump class save in working directory
with open('class.pickle','wb') as f:
    pickle.dump(clf, f)
#load pickle
    f = open('class.pickle',rb)
    clf = pickle.load(f)

#check accuracy
accuracy = clf.score(X_test, y_test)

#predict
forecast = clf.predict(X_lately)

#===============================
#method for scaling
from sklearn import preprocessing
X = np.array(df)
X = preprocessing.scale(X)
#alternative method
from sklern.processing import StandardScaler
X = StandardScaler().fit_transform(select_df)


score= accuracy_score(data_true = data_test, data_pred=predictions)

#===============================
from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction)

#===============================
#Pandas dataframe
pd.describe().transpose()
# displace numerical feature
pd.describe()
# displace categorical feature (no numerical values)
pd.describe(include=['O'])
pd.describe(include='all')

df.sample(10)
#===============================
# get index of mininum column value
df['printtime'].idxmin(axis=1)
#===============================
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['log','red_move']].groupby(['log'],as_index=False).agg(['mean','count']).sort_values(by=('red_move','mean'),ascending=False)


df1.groupby(['district','year']).count()['count']

#===============================
train_df.columns
train_df.index
#===============================
# plot using seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('paper')
sns.set_color_codes('pastel')
sns.set_color_codes('muted')
sns.set(style="darkgrid")

sns.scatterplot(x="total_bill",y="tip",hue="size",style="smoke",data=df)
#move legend outside
plt.legend(bbox_to_anchor=(1.05,1),loc=2)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxesspad=0.)

sns.jointplot(x="x",y="y",data=df)
sns.jointplot(x="x",y="y",data=df,kind="kde")

#QQ plot
#https://seaborn-qqplot.readthedocs.io/en/latest/
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn_qqplot as sqp
iris = sns.load_dataset('iris')
sqp.qqplot(iris,x="sepal_length",y="petal_length",height=4,aspec=1.5,)

#
matplotlib.rc('xtick',labelsize=14)
matplotlib.rc('ytick',labelsize=14)

# plot histgram
# only col
grid = sns.FacetGrid(train_df, col='Survived')
grid.map(plt.hist,'Age',bins=20, density=1, alpha=0.5)
# col and row
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=10)
#
# using pointplot
# palette = 'deep','muted','pastel','bright','dark'
grid = sns.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1,2,3], hue_order=['female','male'])
grid.add_legend()
#
# barplot
sns.barplot(train_df['Sex'],train_df['Survived'])
#ci= None, or sd 
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.5, asepect=1.6,)
grid.map(sns.barplot, 'Sex','Fare', order=['female','male'],alpha=.5, ci=None)

sns.barplot(x='Accuracy mean', y='MlA', data=MLA_compare, coloer = 'm')
sns.barplot(x='Accuracy mean', y='MlA', data=MLA_compare, coloer = 'm',ci=None)
sns.barplot(x='Accuracy mean', y='MlA', data=MLA_compare, coloer = 'm',ci=None,label='',edgecolor='')
# countplot
sns.countplot(x='sample',data=df)
# Title
plt.title('Machinea \n')
plt.xlabel('Accuracy Score ')
plt.ylabel('Algorithm')

#================================================== 



#================================================== 
# flexible use of subplot
#================================================== 
# option 1
fig = plt.figure(figsize=(10,14))
ax1 = fig.add_subplot(211)   # (row, col, current_plot_location)
ax1.plot(x,y,'k-',label='jkld')
ax1.set_ylim((0,0.2))
ax1.set_xlable('n_estimater')
ax1.set_ylable('error rate')
leg = ax1.legend(loc='upper right',fancybox=True)
leg.get_frame().set_alpha(0.7)
ax2 = fig.add_subplot(212)   # (row, col, current_plot_location)
ax2.plot(x,y,'k-',label='jkld')
ax2.set_ylim((0,0.2))
ax2.set_xlable('n_estimater')
ax2.set_ylable('error rate')
leg = ax1.legend(loc='upper right',fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()
# option 2
plt.figure(figsize=(14,12))
plt.subplot(2,1,1)
plt.plot(x1,y1,'o-',label='test data')
plt.title('title 1')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(x2,y2,'o-',label='training data')
plt.title('title 2')
plt.xlabel('label 2')
plt.show()
plot_color = 'br'
class_names='AB'
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y==1)
    plt.scatter(X[idx,0],X[idx,1],
                c = c, cmap = plt.cm.Paired,
                s=20, edgecolor='k',
                label='Class %s' % n)
# option 3
fig, saxis = plt.subplots(2,2,figsize=(16,12))
sns.barplot(x='Embarked',y='Survived',data=data1, ax=saxis[0,0])
axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('x')
sns.barplot(x='Pclass',y='Survived',data=data1, ax=saxis[0,1])
sns.barplot(x='Embarked',y='Survived',data=data1, ax=saxis[1,0])
sns.barplot(x='Pclass',y='Survived',data=data1, ax=saxis[1,1])
# option 4
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(14,10))
sns.boxplot(x='Pclass',y='Fare', hue='Survived',data=data1, ax=ax1)
axis1.set_title('title 1')
sns.violinplot(x='Pclass',y='Age',hue='Survived',data=data1, ax=ax2)
ax2.set_title('title 2')
# option 5
fig, ax = plt.subplots(2,1,figsize=(14,10))
sns.boxplot(x='Pclass',y='Fare', hue='Survived',data=data1, ax=ax[0])
axis1.set_title('title 1')
sns.violinplot(x='Pclass',y='Age',hue='Survived',data=data1, ax=ax[1])
ax2.set_title('title 2')


# sns annotation
splot = sns.boxplot(x='x',y='y',data=df)
form = '2d'
form = '.2f'
for p iin splot.patches:
    splot.annotate(format(p.get_height(),form),
            (p.get_x()+0.5*p.get_width(),p.get_height()+1),
            ha='center',
            va='center',
            xytext=(0,10),
            textcoords='offset points')

# hide legend
axs.get_legend().set_visible(False)

# plot meshgrid
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                    np.arange(y_min, y_max, plot_step))
Z = dbt.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx,yy,z,cmap=plt.cm.Paired)

# subplot title
fig,axs = plt.subplots(2,1,sharex=True)
st=fig.suptitle('title',fontsize='x-large')
st.set_y(0.9)
st.set_x(0.45)

#===============================
# PANDAS
#===============================
# cleanup rare title names
stat_min = 10
title_names = (data1['Title'].value_counts() < stat_min)
data1['Title'] = data1['Title'].apply(lambda x:'Misc' if title_names.loc[x] == True else x)
#===============================
# enclodeing
#http://pbpython.com/categorical-encoding.html
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#https://pandas.pydata.org/pandas-docs/stable/categorical.html
label = LabelEncoder()
dataset['Sex_Code'] = label.fit_transform(dataet['Sex'])
#===============================
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
#===============================
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
pd.crosstab(data1['Title'], data1['Survived'])
#===============================
combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    ps.crosstab(train_df['Title'], train_df['Sex'])
# apply
import re
def get_title(name):
    title_search = re.search(' (A-Za-z)+)\.'), name)
    if title_serach:
        return title_serach.group(1)
    return ''
dataset['Title'] = dataset['Name'].apply(get_title)

# convering categorial to number
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
#mapping
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
#fillna
df['Age'].fillna(method='ffill')    # forward fill
df['Age'].fillna(method='bfill')    # backward fill
#or
df = df.ffill(axis=0)       

# check is any nan in the dataframe
df.isnull().any() -> return list of False or True 
df.isnull().any().any() -> return False or True
# alternative
df.isnull().sum()
df.isnull().sum().sum()
#  shift col
df[ticker].shift(-i)
# find common index
idx=df1.index.intersection(df2.index)

# transfer dataframe to dict
grps = df['group'].unique().tolist()
d_data = {grp:data['weight'][data['group']==grp] for grp in grps}
part_grp = data.groupby('group').size()[0]

# ANOVA
ifrom scipy import stats
F,p = stats.f_oneway(d_data['ctrl'],d_data['trt1'],d_data['trt2'])

# 
import sklearn.model_selection import train_test_split
from sklearn.ensemble import £RandomForestClassifier
from sklearn.metrics import precision_score

#ML
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)
tickers = df.columns.values.tolist()
clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                        ('knn',neighbors.KNeighborsClassifier()),
                        ('rfor',RandomForestClassifier())])
confidence = clf.score(X_test,y,test)
predictions = cls.predict(X_test)

#===============================
# dataframe operator [] -> return list
#                    [[]]-> return a dataframe
#===============================
# drop nameb
df.drop(['Name','PassengerId'],axis=1)
#===============================

# create age bands
import pandas as pd
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
or
train_df[['AgeBand','Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True )

train_df.columns.values
# replace
df['dose'].replace({1:'placebo',2:'low'}, inplace=True)

stat_min = 10
title_names = (data1['Title'].value_counts()< stat_min)

# lambda x:
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x)
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x)==float else 1)

df['Tax %'] = df['Income'].apply(lambda x: .15 if 10000<x<40000 else .2 if 40000<x<80000 else .25)
df ['Taxes owed']= df['Income'] * df['Tax %']

# write value depends on col
df['Test Col'] = False
df.loc[df['Income']<6000,'Test Col'] = True

#


#===============================
age_avg = dataset['Age'].mean()
age_std = dataset['Age'].std()
age_null_count = dataset['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg-age_std, age_avg+age_std, size=age_null_count)
dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list
dataset['Age'] = dataset['Age'].astype(int)
#===============================
# cut vs qcut
dataset['FareBin']=pd.qcut(dataset['Fare'],4)
dataset['AgeBin'] =pd.cut(dataset['Age'].astype(int),5)

#===============================
# loc
dataset.loc[dataset['Age']<=16, 'Age'] = 0
dataset.loc['IsAlone'].loc[dataset['FamilySize']>1] = 0
# loc
dataset.loc[dataset['FamilizeSize']==1, 'IsAlone'] = 1
dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# map
df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)
# replace
df['dose'].replace({1:'placebo',2:'low'}, inplace=True)

# get the most frequent status
freq_port = train_df['Embarked'].dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
# sort
df.sort_values(by=['Age'],ascending=False,inplace=True)
# set value
for index, row in data2.iterrows():
    data1.set_value(index,'radom',1)


# dropna
df.dropna(axis=1)  # drop all columns with NaN
df.dropna(axis=0)  # drop rows iwth na
df.dropna(how='all') # drop rows where all elements are missing
df.dropna(thresh=2)  # drop rows with at least 2 NA vales
df.dropna(subset=['name'],axis=0) # look for NA only in subset columns

# split column into multiple columns
new = df['pos'].str.split('_',n=1,expand=True)
df[['new1','new2']]= df['pos'].str.split('_',n=1,expand=True)
# int to string, fill zero ahead of 2 digits
cad['s']=cad['s'].astype(str).str.zfill(2)

# replace by the first column
df['First']=df['First'].str.split(expand=True)

#===============================
df = df.replace(np.nan, 'N/A', regex=True)

#===============================
#MultiIndex 
sample_multiindex = pd.MultiIndex.from_tuples([('A',1),('A',2),('A',3),('B',1),('B',1)], names=['letter','number'])
sample_series_with_multiindex=pd.Series(list(range(100,105)),index=sample_multiindex)
#===============================
# unstack -> transform a dataset with a pd.MultiIndex into a 2 dimensional array
map_2d=planck_map.unstack()
type(map_2d)
#===============================
#simplest matplot example
import matplotlib.pyplot as plt
df.plot()
df.plot(label='X={} deg'.format(X))
plt.title('Emission at different latitudes')
plt.legend();
#===============================
# pandas correlation
df_corr = df.corr()
#===============================
# if contains string
df_fandry = df[df['log'].str.contains('3s',regex=True)]
# if not contains string
df_airdry = df[~df['log'].str.contains('3s',regex=True)]
df_airdry = df[~df['log'].str.contains('3s',regex=True,na=False)]

# change column datatype
df['slide'] = df['slide'].astype(int)

#===============================
# heatmap
sns.heatmap(df_corr)
fig,ax = plt.subplots(figsize=(14,12))
colormap = sns.diverging_palette(220,10,as_cmap=True)
fig = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws=={'shrink':.9},
        ax=ax,
        annote=True,
        linewidths=.1,vmax=1.0,linecolor='white',
        annot_kws={'fontsize':12}
        )
plt.title('title',y=1.05, size=15)

colormap=plt.cm.RdBu
plt.figure(figsize=(14,12))
sns.heatmap(train.astype(float).corr(),linewidth=0.1,vmax=1.0,
        square=True,cmap=colormap, linecolor='white',annote=True)

# heatmap seaborn using pivot
flights_log = sns.long_dataset("flights")
flights = flights_long.pivot("month","year","passengers")
fig,ax=plt.subplots(1,1,figsize=(15,15),dpi=300)
sns.heatmap(flights,annot=True,fmt="d",linewidth=.5)
ax.set_ylabel('')
ax.set_xlabel('')


#set fontsize seaborn
sns.set_context('poster',font_scale=1)




#plot feature 
import seaborn as sns 
g = sns.FacetGrid(train_df, col='Survived') 
g.map(plt.hist, 'Age', bins=20) 
g = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6) 
g.map(plt.hist, 'Age', alpha=.5, bins = 10) 
grid.add_legend() 
g = sns.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.6) 
g.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep', 
order=[1,2,3], hue_order=['female','male']) 
#================================================== 
fig,(axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5)) 
sns.countplot(x='Embarked',data=train_df,ax=axis1) 
sns.countplot(x='Survived',hue='Embarked',data=train_df,order=[0,1],ax=axis2) 
embark_perc = train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean() 
sns.barplot(x='Embarked',y='Survived',data=train_df, order=['S','C','Q'],ax=axis3) 
#================================================== 
sns.factorplot('Embarked','Survived',data=train_df, size=4, aspect=3) 
fig,(axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5)) 
sns.factorplot('Family',data=train_df, kind='count',ax=axis1) 
sns.countplot(x='Family',data=train_df, order=[1,0],ax=axis1) 
family_perc=train_df[['Family','Survived']].groupby(['Family'],as_index=False).mean() 
sns.barplot(x='Family',y='Survived',data=family_perc, order=[1,0],ax=axis2) 
axis1.set_xticklabels(['with family','alone'], rotation=0) 
sns.factorplot('Pclass','Survived',order=[1,2,3],data=train_df,size=5) 
#================================================== 
splot=sns.catplot(x='log',y='printtime',hue='donor',jitter=0.2,data=df)
splot.set_xticklabels(rotation=90)
#================================================== 
grid=sns.FacetGrid(train_df,hue='Survived',aspect=4) 
grid.map(sns.kdeplot, 'Age',shade=True) 
grid.set(xlim=(0,train_df['Age'].max())) 
grid.add_legend() 
#================================================== 
sns.heatmap(df.corr) 
colormap=plt.cm.RdBu 
plt.figure(figuresize=(14,12)) 
plt.title('title', y=1.05, size=15) 
sns.heatmap(train_df.corr(),linewidths=0.1,vmax=1.0,square=True,
        cmap=colormap,linecolor='white',annot=True) 
#df2= df.iloc[:,[16:38:2]] 
df2= df.iloc[:,[range{16:38:2}]] 
corr = df2.corr() 
corr = (corr)   #why 
sns.heatmap(corr, 
        xticklabels=corr.columns.values, 
        yticklabels=corr.columns.values, 
        cmap='Blues') 
#========================================= ========= 
# visualization 
# 
%pylab inline
# 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab 
import seaborn as sns 
from pandas.tools.plotting import scatter_matrix 
%matplotlib inline 
mpl.style.use('ggplot') 
sns.set_style('white') 
pylab.rcParams['figure.figsize']=12.8 
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
#================================================== 
#we will use seaborn graphics for multi-variable comparison: https://seaborn.pydata.org/api.html

#graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])

#graph distribution of qualitative data: Pclass
#we know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')

#graph distribution of qualitative data: Sex
#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')

#plot distributions of age of passengers who survived or did not survive
a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data1['Age'].max()))
a.add_legend()


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data1)
#================================================== 
# common model algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble,discriminant_analysis, gaussian_process
#common model helpers
from sklean.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection

#Ensemble methods
ensemble.AdaBoostClassifier()
ensemble.BaggingClassifier()
ensemble.ExtraTreesClassifier(),
ensemble.GradientBoostingClassifier(),
ensemble.RandomForestClassifier()

#Gaussian processes
gaussian_process.GaussianProcessClassifier()

#GLM
linear_model.LogisticRegressionCV(),
linear_model.PassiveAggressiveClassifier(),
linear_model.RidgeClassifierCV(),
linear_model.SGDClassifier(),
linear_model.Perceptron(),

#Navies Bayes
naive_bayes.BernoulliNB(),
naive_bayes.GaussianNB(),

#Nearest Neighbor
neighbors.KNeighborsClassifier(),

#SVM
svm.SVC(probability=True),
svm.NuSVC(probability=True),
svm.LinearSVC(),

#Trees    
tree.DecisionTreeClassifier(),
tree.ExtraTreeClassifier(),

#Discriminant Analysis
discriminant_analysis.LinearDiscriminantAnalysis(),
discriminant_analysis.QuadraticDiscriminantAnalysis(),

#XGBooster
#XGBClassifier()

#metrics
from sklearn import metrics
metrics.accuracy_score(df['target'],df['predict'])
#Accuracy Summary Report with http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
#Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
#And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
print(metrics.classification_report(data1['Survived'], Tree_Predict))
# confusion matrix
cnf_matrix=metrics.confusion_matrix(data1['Survived'],Tree_Predict)

#================================================== 
# numpy
#================================================== 
import numpy as np
# compute Covariance matrix
Sigma = np.cov(train_data, rowvar=1, bias=1)
# compute coordinate-wise variances in increasig order
coordinate_variances = np.sort(Sigma.diagonal())
# compute variances in eigenvector directions, in increasing order
eigenvector_vairances = np.sort(np.linalg.eigvalsh(Sigma))
#
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

#list and array
#list operation
x=arange(0,2*pi,2*pi/365)
v = []
v.append(np.array(cos(0*x))*c/sqrt(2))
v.append(np.array(sin(x))*c)

a = ['a','b','c']
maxlen = 10

result = [(0 if i+1 > len(a) else a[i]) for i in range(maxlen)]

# find common elements in two lists
common = [i for i in x if i in y]
# list to array
A=vstack(v)

#
np.insert(arr,0,0)
np.argmin(arr,axis=0)
np.argsort()
np.argpartition()
np.stack(A)

np.nansum(mat)
np.nanmax(mat)
np.nanmin(mat)

# remove the last element in the list
my_list.pop()
# remove the first element in the list
my_list.popleft()

#================================================== 
# Adaboost
#================================================== 
# Adaboost regression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, AdaBosstClassifier
from numpy.core.umath_tests import inner1d
rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                         n_estimators=100, random_state=rng)
regr.fig(X,y)
y_1 = regr.predict(X)
ada_err = np.zeros((n_estimators,))
for i, y_pred in enumerator(regr.staged_predict(X)):
    ada.err[i] = np.linalg.norm((y_pred-y))

# Adaboost Classifier
from sklearn.metrics import zero_one_loss
learning_rate = 1.
dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
ada_discrete= AdaBoostClassifier(base_estimator=dt,
                              learning_rate = learning_rate,
                              n_estimators = n_estimators,
                              algorithm='SAMME')
ada_discrete.fit(X_train, y_train)
y_pred = ada_discrete.fit(X_test)

ada_real = AdaBoostClassifier( base_estimator=dt,
                            learning_rate = learning_rate,
                            n_estimators=n_estimators,
                            algorithm='SAMME.R')
ada_real.fit(X_train, y_train)
y_pred = ada_discrete.fit(X_test)

ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)
        
# Train test split
train_x,test_x,train_y,test_y = model_selection.train_test_split(data_x, data_y,random_state = 0)
# Cross validation
from sklearn iport model_selection
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
model_selection.train_test_split
MLA = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        #Gaussian Processes
        gaussian_process.GaussianProcessClassifier()
        ]
MLA_columns = ['NLA Name', 'MLA Parameters']
MLA_compare = pd.DataFrame(columns = MLA_columns)
row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())


    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv=cv_split)
    # component in cv_results 'train_score', 'test_score', 




# metrics
import 
score = metrics.accuracy_score(train_Y, pred_Y)
metrics.classification_ report(train_Y, pred_Y)

#================================
from sklearn.svm import SVC
clf = SVC(kernel='rbf',C=C_value,gamma=1.0/(s_value*s_value))
clf = SVC(kernel='poly', degree=2, C=C_value, coef0=1.0)
clf = SVC(kernel='linear'

#================================
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
svc = svm.SVC(C=1,kernel='linear')
k_fold = KFold(n_splits=5)
score_array=cross_val_score(svc,X_train,Y_train,cv=k_fold,n_jobs=-1)

#================================
from sklearn.cross_validtion import KFold
ntrain = train.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

for i, (train_index, test_index) in enumerate(kf):
    x_tr = x_train[train_index]
    y_tr = y_train[train_index]
    x_te = x_train[test_index]
    clf.fit(x_tr, y_tr)

#================================

colormap = sns.diverging_palette(220,10,as_cmap=True)
colormap=plt.cm.RdBu 
#================================
# save figure to pdf
#================================
%pylab inline
from matplotlib.backends.backend_pdf import PdfPages
from lib.PlotTime import PlotTime
pp = PdfPages('MemoryBlockFigure.pdf')
figure(figuresize=(6,4))
Colors='bgrcmyk'    # color
lineStype=['-']
for m_i in range(len(m_list)):
    Color = Colors[m_i % len(m_list)]
    PlotTime()

plt.savefig('fivure.png',bbox_inches="tight")
if not show:
    plt.close()

    

#===============================
# jupyter
#================================================== 
# write cell content to file
%%writefile ../lib/myfuncs.py
# load file to cell
%load ../lib/myfuncs.py

# Enable automatic reload of libraries
%load_ext autoreload 
%autoreload 2 # means that all modules are reloaded before every command
# load function from lib to execute but not display
from lib.funcs import *

#================================================== 
# pandas
#================================================== 
df.filter(regex='e$',axis=0) # select row name ending with 'e'
df.filter(regex='^c',axis=0) # select row name starting with 'c'

# remove any instance(s) which has 2 digits followed by 5 digits and again 5 digits
df = df[df['Name'].str.contains(r'\d{2} \d{5} \d{5}')==False]

df = df.reset_index(drop = True)
# handling redundacy
df.drop_duplicates()
# fine top 3 words 
all_chat_list=[]
for i in range(len(df['Name'].drop_duplicates())):
    temp= df['Convo'][df['Name']==df['Name'].drop_duplicates().reset_index(drop=True)[i]]
    temp= temp.reset_index(drop = True)
    for j in range(1,len(temp)):
        temp[0] +=' '+temp[j]
    all_chat_list.append(temp[0])
    del temp

from scipy.stats import itemfreq
fg = itemfreq(list(all_chat_list)[0].split(' '))
fg = fg[fg[:,1].astype(float).argsort()][::-1]
print(fg[1:4])

# select column
# get list of seceltion
df[i'libido'][df['dose']=='high']

#================================================== 
# jupyter
#================================================== 
# load py code 
%load filename.py
# write below line at the begining of the cell to write to py code
%%writefile filename.py
# run python code
%%run filename.py
# get help on magic function
%magic
%lsmagic
%magic_name
# run unix command
!python --version
!cat file.txt
# split cell
Ctrl+Shift+-
# merge cell
shift M
#================================================== 
from tqdm import tqdm
#================================================== 
#Os
import os
ticker_list = os.listdir(statspath)
os.remove(f"{statspath}/.DS_Store")
ticker_list.remove(".DS_Store")

os.makedirs(stock_dfs)
os.path.isfile(filename)
os.path.exists(filename)
os.path.join(path,"zlog")+"\\"
os.listdir(path)
os.remove('file.txt')
#get path pwd
path=os.path.abspath(os.getcwd())
# get paht 1 level up
path = os.path.abspath(os.path.join(os.getcwd(),".."))
# get paht 2 level up
path = os.path.abspath(os.path.join(os.getcwd(),"../.."))

# rename folder
os.rename(os.path.join(base_folder,old),os.path.join(base_folder,new))
# copy files 
import shutil
shutil.copy2(f_from,f_to)


my_png = os.path.join(path,"my_png")+"\\"
os.listdir(my_png)
#alternative
file_list = glob.glob(my_png+"_10x_performance.txt")
#list only files
file_lsit = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]

# rename a file
os.rename(src,dst)

import pandas as pd
my_dict = {"tester": 1, "testers": 2}
df=pd.DataFrame(my_dict,index=[0])
df.to_csv("path and name of your csv.csv")
# set to data frame type to plot
df = pd.read_csv('ABC.csv') #read cvs
df2 = df.loc[0:12] #select rows
# get rid of "Unnamed:0"
df = pd.read_csv('file.txt',index_col=[0])

# set index as Date
df2['Date'] = pd.to_datetime(df2['Date'])
df2.set_index('Date', inplace=True) #set Date as index
df2['Close'].plot() #plot
# add 5 business days
import datetime as dt
from pandas.tseries.offsets import BDay
ts = pd.Timestamp(dt.datetime.now())
>>> ts + BDay(5)

base = datetime.datetime.today()
date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]

# list operation: remove 
unwanted = [2, 3, 4]
item_list = [e for e in item_list if e not in unwanted] 

df.rename(columns={'close':ticker}, inplace=True)

df2.index=df2.index.values.astype('<M8[D]')
# x label overlap
fig, axs = plt.subplots(figsize=(6,4))
axs.hist(x)
axs.set_xticks(axs.get_xticks()[::2])
plt.show()

# 
plt.rcParams.update({'figure.figsize':(8,6),'figure.dpi':120})

#list NaN
df_null = df[pd.isnull(df['info'])]
df.sample(10)
df.isnull().sum()

#sort index
df.sort_index(axis=0)
df.sort_index(axis=1)

# fillna with another column
df['info'].fillna(df['add_info'],inplace=True)

# find duplicated index
duplicated_idx =  df[df.index.duplicated()].index
df.drop(duplicated_idx, axis = 0, inplace=True)
# get unique label

#construct empty dataframe
df_empty = pd.DataFrame({'A': []})
return df_empty  #True
if df_empty.empty:
    pass

#filter row by list of strings
cleaned = df[~df['stn'].isin(remove_list)]

# tolist
slide_list = df['slide'].tolist()

#get row with index
df.log[idx]
#update colume with new values
df.update(slide_info)
# merge two dataframe
c = pd.merge(df_left,df_right,on=key)


# pillow image compression
from PIL import Image
img = "myimg.png"
im = Image.open(img)
im_resize = im.resize((600,600),Image.ANTIALIAS)    # compression optimized
im_resize.save("resized_"+img)
im_resize.show()    # external window
display(im_resize)  # jupyter

#conver between a PIL image and a numpy array
im = Image.open("sample.png")
np_im = numpy.array(im)

new_im = Image.framarray(np_im)

# create new empty image
from PIL import Image
image = Image.new('RGB',(800,1280),(255,255,255))
img.save('image.png','PNG')


# get rid of \n
.rstrip()

# transfer dataframe to dict
grps = df['group'].unique().tolist()
d_data = {grp:data['weight'][data['group']==grp] for grp in grps}
part_grp = data.groupby('group').size()[0]

# ANOVA with scipy
from scipy import stats
F,p = stats.f_oneway(d_data['ctrl'],d_data['trt1'],d_data['trt2'])

# ANOVA with scipy.stats
import scipy.stats as stats
stats.f_oneway(data_group1,data_group2,data_group3,data_groupN)
# ANOVA with statsmodels
import statsmodels.formula.api import ols
import statsmodels.api as sm
mod = ols('outcome_variable ~ C(groups_variable)',data=your_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
# ANOVA pairwise Comparisons
mod = ols('weight ~ group',data=data).fit()
pair_t = mod.t_test_pairwise('group')
pair_t.result_frame


#===============================
# convert list to dictionary keys
list_samples
list_donors
dict_sample = dict.fromkeys(list_samples,1.0)
# create dict from 2 lists
zipObj= zip(list_samples,list_donors)
dict_sample = dict(zipObj)
df['donor']=df['sample'].map(dict_sample)


#===============================
# arg
args = [i for i in my_list]
F,p = stats.f_oneway(*arg*args)
#===============================
# format
form = '.2f'
splot.annotate(format(p.get_height(),form))
# left align
print('{:20s}'.format('Align Left'))
print('{:>20s}'.format('Align Right'))
print('{:^20s}'.format('Align center'))

# string format
print('{:5d}'.format(1))-> 00001

print('stat=%.3f, p=%.3f'%(stat,p))

print('Hello, %s' %str)

#===============================
#list
a = []
a.append(b)
sorted(a)
sorted(a,reverse=True)
#map list using dictionary
l_map = list(map(my_dict.get,low_mag))


#===============================
#Jupyter notebook
#list magic command
%lsmagic
%debug
%timeit
#-------------------------------
%%latex
$e^{i pi} = -1$
#-------------------------------
# get build-in __doc__
# add ? after fuction name
FuncName?
FundName.__doc__
#-------------------------------
# set envirnment variables
%env OMP_NUM_THREADS%env OMP_NUM_TREADS=4
#-------------------------------
#timing
import time
#return information about a single run of the code in one cell
%%time

#uses the Python timeit module which runs a statement 100,000 time
#(by default) and provides the man of the fastest three times.
%timeit
#-------------------------------
#shows (in a popup) the syntax highlighted contents of an external fiel
%pycat xx.py
#-------------------------------
#debug tool
%pdb
#-------------------------------
#suppress output by adding ;
#-------------------------------
#execute shell commands
!ls *.csv
!pip install numpy
!pip list |grep pandas
#-------------------------------
hold <Alt> column selection







#===============================
from openpyxl.workbook import Workbook

to_excel = df.to_excel('df.xlsx')

#===============================
# python 2
# virtual environment
$pip install virtualenv
# add virtualenv pytest_27_venv
$virtualen pytest_27_venv
# activate pytest_27_venv
$source ./pytest_27_venv/bin/activate
#install packet
$pip install pytest
$deactivate
# deleate the top level directiory 

# python 3 build-in virtual envio
# create virtual environment
$python3 -m venv pytest_3_vene
$source pytest_3_venv/bin/activate
$pip install pytest
$deactivate


if __name__ =="__main__":
    print("Hello World!")


#===============================
# pf to excel
#===============================
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
wb = Workbook()
ws = wb.active

ws2 = wb.create_sheet('new sheet')

ws.title='MySheet'
rows = dataframe_to_rows(df,index=False)
for i, row in enumerate(rows,1):
    for j,col in enumerate(row,1):
        ws.cell(row=i,column=j,value=col)
wb.save('file.xlsx')

#===============================
# 
try:
    ..
except Exception as e:
    print(e)
    raise e


#===============================
# print dict
data={'city':1,'region':12,'country':23}
print("{city}, {region},{country}".format(**data))

#===============================
$pip install requests
$pip install numpy simpleaudio
# opencv
$pip install opencv-contrib-python imageio
# PEP python enhance protocol 
# pycodestyle: check python code against style convention
$pip install pycodestyle
$pycodestyle code.py
# flake8: comtinges a debutter, pyflates with pycodestyle
$pip install flake8
$flake8 code.py
# auto formatters
$pip install black
$black code.py
# alter linelength limit
$black --line-length=79 code.py

#===============================
file_name[-3:]=='jpg'
if file_name.endswith('jpg'):
    print(file_name)
file_name[:3]=='jpg'
if file_name.startswith('jpg'):


#===============================
#unit test pandas
import pandas as pd
pd.testing.assert_frame_equal(my_df,expected_df)
pd.testing.assert_series_equal(my_series,expected_series)
pd.testing.assert_index_euqal(my_index,expected_index)

#===============================
# python program strucutre
# smart_door.py
def close():
    pass
def open():
    pass

import smart_door
smart_door.open()
smart_door.close()
# alternative
from smart_door import open
open()
