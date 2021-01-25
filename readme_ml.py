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


import sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#ML
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)
tickers = df.columns.values.tolist()
clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                        ('knn',neighbors.KNeighborsClassifier()),
                        ('rfor',RandomForestClassifier())])
confidence = clf.score(X_test,y,test)
predictions = cls.predict(X_test)


# Arima
#https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# check stationarity of time series
from pmdarima.arima.stationarity import ADFTest
adf_test = ADFTest(alpha=0.05)
p_val, should_diff = adf_test.should_diff(y)



# FASTAI
# tabular
dep_var = 'Survived'
#cat_names = data.select_dtypes(exclude=['int', 'float']).columns
cat_names = [ 'Sex', 'Ticket', 'Cabin', 'Embarked']

#cont_names = data.select_dtypes([np.number]).columns
cont_names = [ 'Age', 'SibSp', 'Parch', 'Fare']

# Transformations
procs = [FillMissing, Categorify, Normalize]

# Test Tabular List
test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)

# Train Data Bunch
data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                        .split_by_idx(list(range(0,200)))
                        .label_from_df(cols = dep_var)
                        .add_test(test, label=0)
                        .databunch())

data.show_batch(rows=10)



# Create deep learning model
#learn = tabular_learner(data, layers=[1000, 200, 15], metrics=accuracy, emb_drop=0.1, callback_fns=ShowGraph)
learn = tabular_learner(data, layers=[1000, 200, 15], metrics=rmse, emb_drop=0.1, callback_fns=ShowGraph)

# select the appropriate learning rate
learn.lr_find()

# we typically find the point where the slope is steepest
learn.recorder.plot()

# Fit the model based on selected learning rate
learn.fit_one_cycle(15, max_lr=slice(1e-03))

# Analyse our model
learn.model
learn.recorder.plot_losses()

# save model
learn.model_dir = '/content/drive/My Drive/Colab/zcml2/model/'
learn.save('model')

# Predict our target value
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)

# create submission file to submit in Kaggle competition
submission = pd.DataFrame({'PassengerId': test_id, 'Survived': labels})
submission.to_csv('submission.csv', index=False)
submission.head()

# load model for further training
learn2 = tabular_learner(data,layers=[1000,200,15],metrics=rmse,emb_drop=0.1,callback_fns=ShowGraph)
learn2.model_dir = '/content/drive/My Drive/Colab/zcml2/model/'
learn2.load('model')
learn2.fit_one_cycle(100,1e-2)
