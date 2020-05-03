#===============================
#Numpy
#===============================
#method for scaling
from sklearn import preprocessing
X = np.array(df)
X = preprocessing.scale(X)
#alternative method
from sklern.processing import StandardScaler
X = StandardScaler().fit_transform(select_df)

score= accuracy_score(data_true = data_test, data_pred=predictions)

# 
from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction)

#===============================
#Pandas dataframe
#===============================
# displace numerical feature
pd.describe()
pd.describe().transpose()

# get columns or row index
train_df.columns
train_df.index
train_df.columns.values

# displace categorical feature (no numerical values)
pd.describe(include=['O'])
pd.describe(include='all')

# get 10 instances from df. 
df.sample(10)
df.sample(100,replace=True) #allow duplication

# get index of mininum column value
df['printtime'].idxmin(axis=1)

train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['log','red_move']].groupby(['log'],as_index=False).agg(['mean','count']).sort_values(by=('red_move','mean'),ascending=False)

train_df['smoker'].value_counts()
train_df['smoker'].value_counts(normalize=True)

df1.groupby(['district','year']).count()['count']

#===============================
# cleanup rare title names
stat_min = 10
title_names = (data1['Title'].value_counts() < stat_min)
data1['Title'] = data1['Title'].apply(lambda x:'Misc' if title_names.loc[x] == True else x)
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
#
age_avg = dataset['Age'].mean()
age_std = dataset['Age'].std()
age_null_count = dataset['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg-age_std, age_avg+age_std, size=age_null_count)
dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list
dataset['Age'] = dataset['Age'].astype(int)

# transfer dataframe to dict
grps = df['group'].unique().tolist()
d_data = {grp:data['weight'][data['group']==grp] for grp in grps}
part_grp = data.groupby('group').size()[0]

dict_vote = {'vote':np.where(np.random.rand(n) < 0.1,'Brown','Green')}
df_vote = pd.DataFrame({'vote':np.where(np.random.rand(n) < 0.1,'Brown','Green')})

#filter row by list of strings
cleaned = df[~df['stn'].isin(remove_list)]
# 
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
# select column
# get list of seceltion
df[i'libido'][df['dose']=='high']

# cut vs qcut
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand','Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True )
smoking['ageGroup'] = pd.cut(smoking['age'],[0,30,40,53,54],labels['0-30','30-40','40-53','53-60'])
dataset['FareBin']=pd.qcut(dataset['Fare'],4)
dataset['AgeBin'] =pd.cut(dataset['Age'].astype(int),5)

# replace
df['dose'].replace({1:'placebo',2:'low'}, inplace=True)

stat_min = 10
title_names = (data1['Title'].value_counts()< stat_min)

# lambda x:
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x)
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x)==float else 1)

df['Tax %'] = df['Income'].apply(lambda x: .15 if 10000<x<40000 else .2 if 40000<x<80000 else .25)
df ['Taxes owed']= df['Income'] * df['Tax %']

# filter value
df['Test Col'] = False
df.loc[df['Income']<6000,'Test Col'] = True
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
# drop name
df.drop(['Name','PassengerId'],axis=1)

# split column into multiple columns
new = df['pos'].str.split('_',n=1,expand=True)
df[['new1','new2']]= df['pos'].str.split('_',n=1,expand=True)
# int to string, fill zero ahead of 2 digits
cad['s']=cad['s'].astype(str).str.zfill(2)
# replace by the first column
df['First']=df['First'].str.split(expand=True)
# slice by the column df[].str.slice(start,stop,step)
df['sex'] = df['sexage'].str.slice(0,1)
df['age'] = df['sexage'].str.slide(1)

# if contains string
df_fandry = df[df['log'].str.contains('3s',regex=True)]
# if not contains string
df_airdry = df[~df['log'].str.contains('3s',regex=True)]
df_airdry = df[~df['log'].str.contains('3s',regex=True,na=False)]

# change column datatype
df['slide'] = df['slide'].astype(int)

# replace
df = df.replace(np.nan, 'N/A', regex=True)
# sample 
df.sample(100,replace=True).describe()
# stack -> transfer multiindex dataframe to flat 

# unstack -> transform a dataset with a pd.MultiIndex into a 2 dimensional array
map_2d=planck_map.unstack()
type(map_2d)

# get corration
df_corr = df.corr()

df.query()
df.query('country="China" and year == 2015')

#MultiIndex 
sample_multiindex = pd.MultiIndex.from_tuples([('A',1),('A',2),('A',3),('B',1),('B',1)], names=['letter','number'])
sample_series_with_multiindex=pd.Series(list(range(100,105)),index=sample_multiindex)

import pandas as pd
my_dict = {"tester": 1, "testers": 2}
df=pd.DataFrame(my_dict,index=[0])
df.to_csv("path and name of your csv.csv")
# set to data frame type to plot
df = pd.read_csv('ABC.csv') #read cvs
df2 = df.loc[0:12] #select rows
# get rid of "Unnamed:0"
df = pd.read_csv('file.txt',index_col=[0])
# selective read
df = pd.read_csv('file.csv').query('year==1986')
# read first nrows 
df = pd.read_csv('file.csv',nrows=100)
# sort data
df.sort_values('age',inplace=True)
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

#
size=np.where(df['year']%10 ==0, 30,2)
color=np.where(df['country']=='China', 'red', 'orange')
ax = df.plot.scatter('x','y',logx=True,s=size,c=color)
df.plot.line(x='yy',y='life',ax=ax)

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

if df is None:
    pass


# tolist
slide_list = df['slide'].tolist()

#get row with index
df.log[idx]
#update colume with new values
df.update(slide_info)
# merge two dataframe
c = pd.merge(df_left,df_right,on=key)
# concat two dataframe
c = pd.concat([df1,df2],axis=0,join='outer',sort=False)

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

np.random([0,1,4,5],size=n,p=[0.65,0.15,0.1,0.1]) # p proprotion to 
#list operation
x=arange(0,2*pi,2*pi/365)
v = []
v.append(np.array(cos(0*x))*c/sqrt(2))
v.append(np.array(sin(x))*c)

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

ret = [2,3.4,54]
mean = np.mean(ret)


#transfer boolean into string
string_array = np.where(np.random.rand(5)>0.51,'Brown','Green')
# 
dict_vote = {'vote':np.where(np.random.rand(n) < 0.1,'Brown','Green')}
df_vote = pd.DataFrame({'vote':np.where(np.random.rand(n) < 0.1,'Brown','Green')})

#================================================== 
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
os.path.isdir(folder)
#get path pwd
path=os.path.abspath(os.getcwd())
# get paht 1 level up
path = os.path.abspath(os.path.join(os.getcwd(),".."))
# get paht 2 level up
path = os.path.abspath(os.path.join(os.getcwd(),"../.."))

# rename folder
os.rename(os.path.join(base_folder,old),os.path.join(base_folder,new))

# rename a file
os.rename(src,dst)

# copy files 
import shutil
shutil.copy2(f_from,f_to)

my_png = os.path.join(path,"my_png")+"\\"
os.listdir(my_png)
#alternative
file_list = glob.glob(my_png+"_10x_performance.txt")

# =======================================================
# List comprehension
# =======================================================
#list only files
file_lsit = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]

a = ['a','b','c']
maxlen = 10
result = [(0 if i+1 > len(a) else a[i]) for i in range(maxlen)]

# find common elements in two lists
common = [i for i in x if i in y]

# remove the last element in the list
my_list.pop()
# remove the first element in the list
my_list.popleft()

#
# list comprehension
# transfer all string in a list to int
int_list = list(map(int, string_list))

#===============================
# convert list to dictionary keys
list_samples
list_donors
dict_sample = dict.fromkeys(list_samples,1.0)
# create dict from 2 lists
zipObj= zip(list_samples,list_donors)
dict_sample = dict(zipObj)
df['donor']=df['sample'].map(dict_sample)

#list
a = []
a.append(b)
sorted(a)
sorted(a,reverse=True)
#map list using dictionary
l_map = list(map(my_dict.get,low_mag))




#================================
# plot
#================================
colormap = sns.diverging_palette(220,10,as_cmap=True)
colormap=plt.cm.RdBu 

# save figure to pdf
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

# x label overlap
fig, axs = plt.subplots(figsize=(6,4))
axs.hist(x)
axs.set_xticks(axs.get_xticks()[::2])
plt.show()

# 
plt.rcParams.update({'figure.figsize':(8,6),'figure.dpi':120})

# pillow image compression
from PIL import Image
img = "myimg.png"
im = Image.open(img)
im_resize = im.resize((600,600),Image.ANTIALIAS)    # compression optimized
im_resize.save("resized_"+img)
im_resize.show()    # external window
display(im_resize)  # jupyter

#conver between a PIL image and a numpy array
em = Image.open("sample.png")
np_im = numpy.array(im)

new_im = Image.framarray(np_im)

# create new empty image
from PIL import Image
image = Image.new('RGB',(800,1280),(255,255,255))
img.save('image.png','PNG')


#===============================
# get rid of \n
.rstrip()



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
# df to excel
#===============================

from openpyxl.workbook import Workbook
to_excel = df.to_excel('df.xlsx')

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

#===============================
#bootstrap: sample iwth replacement
df.sample(100)-> randomyl choose 100 instances from df
df.sample(100,replace=True).describe()
bootstrap=pd.DataFrame({'meangrad':[df.sample(100,replace=True).grade.mean() for i in range(1000)]})

# get pd quantile
df['meangrade'].quantile(0.025)


#===============================
import scipy
#normal distribution
n = scipy.stats.norm(mu,sigma)
scipy.stats.percentileofscore(df['count'],340) -> return confidence p of getting 340
100-scipy.stats.percentileofscore(df['Brown'],0.511)

from scipy.stats import itemfreq
fg = itemfreq(list(all_chat_list)[0].split(' '))
fg = fg[fg[:,1].astype(float).argsort()][::-1]
print(fg[1:4])


# ANOVA
from scipy import stats
F,p = stats.f_oneway(d_data['ctrl'],d_data['trt1'],d_data['trt2'])

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

#================================================== 
# matplotlib

# flexible use of subplot
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
axs[0].set_title('title1')
axs[1].set_title('title1')

#rotate xticks
g = sns.boxplot(x='log',y='count',data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

#matplotlit
plt.xticks(rotation=90)
plt.xticks(df.index[::10],rotation=90)

#================================================== 
#assign same color palette to different plot
sns.set_style('white')
pal = sns.color_pallette('Paired')

sns.boxplot(x='site',y='value',hue='label',data=df,
        palette=pal,fliersize=0)
sns.stripplot(x='site',y='value',hue='label',data=df,
        jitter=True,split=True,linewidth=0.5,palette=pal)
plt.legend(loc='upper left')

#specify standard color to label 
pal = {'healthy':'green','disease':'blue'}

#specify specified color to label
dark_brown = '#B25116'
dark_pink = '#FB84D1'
pal = {'healthy': dark_brown,'disease': dark_pink}

# pass setting using dict
fig,ax = plt.subplots()
stripplot_kwargs = dict({'linewidth':0.6, 'size':6,'alpha': 0.7}, **kwargs)
sns.stripplot(x='x',y='value',hue='label',data=df,fliersize=0,ax=ax,**kwargs)
ax.legend_.remove()
# legend -> keep first 2 legend
handles, labels = ax.get_legendhandles_labels()
lgd = ax.legend(handles[0:2], labels[0:2],
        loc = 'upper left',
        fontsize = 'large',
        handletextpad = 0.5)
lgd.legendHandles[0]._sizes=[40]
lgd.legendHandles[1]._sizes=[40]

# add text to plot
sns.scatter(x='time',y='printtime',data=di,ax=ax)
for idx in di.index:
    ax.text(di.at[idx,'time'],di.at[idx,'printtime'],idx)


#================================================== 
# plt vertical line
# c for color
plt.axvline(df['income'].mean(),c='C1')
plt.axvline(df['income'].median(),c='C1',linestyle='--')
plt.axvline(df['income'].quantile(0.24),c='C1',linestyle=':')
plt.axvline(df['income'].quantile(0.75),c='C1',linestyle=':')


#================================================== 
#pandas plot
df['income'].plot(kind='hist',histtype='step',bins=3,density=True)
df['income'].plot.density(bw_method=0.5)
plt.axis(xmin=0,xmax=3)

df['income'].plot(kind='hist',histtype='step',bins=20,density=True)
levels=[0.25,0,1,2,3]
plt.xticks(np.logs10(levels),levels) #(location,label)


#===============================
# dataframe plot
#simplest matplot example
import matplotlib.pyplot as plt
df.plot()
df.plot(label='X={} deg'.format(X))
plt.title('Emission at different latitudes')
plt.legend();

#===============================
#read png
img = matplotlib.image.imread('london.png')
plt.imshow(img,extend=[-0.38,0.38,-0.38,0.38]) # resize
plt.scatter(df['x'],df['y'],color='b')
#===============================

