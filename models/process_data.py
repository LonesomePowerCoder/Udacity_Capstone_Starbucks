import pickle
import json
import zipfile
import pandas as pd
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def unziper(path=None, file_name=None):
    """
    Function to unzip json files
    :param path: variable pointing to file path
    :return: pandas dataframe from json file
    """
    try:
        with zipfile.ZipFile(path, 'r') as zip_file:
            with zip_file.open(file_name) as json_file:
                data = json.load(json_file)
                data = pd.json_normalize(data)
    except:
        raise Exception("Could not unzip json file")

    return data

#load zipped Starbucks data
portfolio = unziper('../data/portfolio.zip', 'portfolio.json')
profile = unziper('../data/profile.zip', 'profile.json')
transcript = unziper('../data/transcript.zip', 'transcript.json')

#One-hot encode offer channels
portfolio['ch_email'] = 0
portfolio['ch_web'] = 0
portfolio['ch_mobile'] = 0
portfolio['ch_social'] = 0

for i in range(portfolio.shape[0]):
    if 'email' in portfolio.iloc[i].channels:
        portfolio.loc[i,'ch_email'] = 1
    if 'web' in portfolio.iloc[i].channels:
        portfolio.loc[i,'ch_web'] = 1
    if 'mobile' in portfolio.iloc[i].channels:
        portfolio.loc[i,'ch_mobile'] = 1
    if 'social' in portfolio.iloc[i].channels:
        portfolio.loc[i,'ch_social'] = 1

portfolio.drop(columns=['channels'], inplace=True)
#One-hot encode offer type
portfolio = pd.concat([portfolio, pd.get_dummies(portfolio["offer_type"], dtype=int)], axis=1)
portfolio.rename(columns={"id":"offer_id"},inplace=True)

#remove gender nas, which also correlate with wrong age
profile = profile.loc[profile.gender.notna(),:]

#calculate the no. of days that some one is a member assuming now equals max date plus one
profile["became_member_on"] = pd.to_datetime(profile["became_member_on"], format='%Y%m%d')
today = profile["became_member_on"].max() + datetime.timedelta(days=1)
profile["member_days"] = (today - profile["became_member_on"]).dt.days

#rename id column
profile.rename(columns={"id":"user_id"},inplace= True)

#impute missing income with sample mean
profile['income'] = profile.income.fillna(profile.income.mean())

#######
#data wrangling for model 1
#######

#filter on received offers
transcript_received = transcript.loc[transcript.event == 'offer received',:]
transcript_received = transcript_received.rename(columns={'value.offer id': 'offer_id'})

#filter on viewed offers
transcript_viewed = transcript.loc[transcript.event == 'offer viewed',:]
transcript_viewed = transcript_viewed.rename(columns={'value.offer id': 'offer_id'})

#prepare cumulated number of viewed offers per person by time
transcript_viewed["cumcount_views"] = 1
transcript_viewed.sort_values(['person','time'],inplace = True)
transcript_viewed["cumsum_views"] = transcript_viewed.groupby(['person'])["cumcount_views"].cumsum()

#merge received and viewed offers. NAs can be interpreted as offers not viewed
transcript_received_viewed = transcript_received.merge(transcript_viewed.loc[:,['person','time','offer_id','cumsum_views']],how = 'left',on=['person','offer_id'])
transcript_received_viewed = transcript_received_viewed.rename(columns={'time_y': 'offer_viewed','person': 'user_id'})
transcript_received_viewed['offer_viewed'] = transcript_received_viewed.apply(lambda row: 0 if pd.isnull(row['offer_viewed']) else 1, axis=1)

#allocate the cumulated views and fill nas
transcript_received_viewed.sort_values(['user_id','time_x'], inplace = True)
transcript_received_viewed['cumsum_views'] = transcript_received_viewed['cumsum_views']-1
transcript_received_viewed['cumsum_views'] = transcript_received_viewed.groupby('user_id')['cumsum_views'].fillna(method = 'ffill')

#drop columns not needed
transcript_received_viewed = transcript_received_viewed.drop(columns=['time_x','event', 'value.amount', 'value.reward'])

#merge profile data in the received offers
transcript_received_viewed = transcript_received_viewed.merge(profile.loc[:,['user_id','age','gender','income','member_days']],how = 'inner',on=['user_id'])
transcript_received_viewed = pd.concat([transcript_received_viewed, pd.get_dummies(transcript_received_viewed["gender"], dtype=int)], axis=1)

#merge portfolio data in the received offers
transcript_received_viewed = transcript_received_viewed.merge(portfolio.loc[:,['offer_id','offer_type','duration','ch_email','ch_mobile','ch_social','ch_web']],how = 'left',on=['offer_id'])
#get dummies for offer type
transcript_received_viewed = pd.concat([transcript_received_viewed, pd.get_dummies(transcript_received_viewed["offer_type"], dtype=int)], axis=1)

#drop nas
transcript_received_viewed = transcript_received_viewed.dropna()

#######
#data wrangling for model 2
#######

#filter on completed offers
transcript_completed = transcript.loc[transcript.event == 'offer completed',:]
transcript_completed = transcript_completed.rename(columns={'value.offer id': 'offer_id'})

#merge received and viewed offers. NAs can be interpreted as offers not viewed
transcript_completed_viewed = transcript_completed.merge(transcript_viewed.loc[:,['person','time','offer_id','cumsum_views']],how = 'left',on=['person','offer_id'])
transcript_completed_viewed = transcript_completed_viewed.rename(columns={'time_y': 'offer_viewed','person': 'user_id'})
transcript_completed_viewed['offer_viewed'] = transcript_completed_viewed.apply(lambda row: 0 if pd.isnull(row['offer_viewed']) else 1, axis=1)

#allocate the cumulated views and fill nas
transcript_completed_viewed.sort_values(['user_id','time_x'], inplace = True)
transcript_completed_viewed['cumsum_views'] = transcript_completed_viewed['cumsum_views']-1
transcript_completed_viewed['cumsum_views'] = transcript_completed_viewed.groupby('user_id')['cumsum_views'].fillna(method = 'ffill')

#drop columns not needed
transcript_completed_viewed = transcript_completed_viewed.drop(columns=['time_x','event','value.reward', 'value.amount'])

#merge profile data in the received offers
transcript_completed_viewed = transcript_completed_viewed.merge(profile.loc[:,['user_id','age','gender','income','member_days']],how = 'inner',on=['user_id'])
transcript_completed_viewed = pd.concat([transcript_completed_viewed, pd.get_dummies(transcript_completed_viewed["gender"], dtype=int)], axis=1)

#merge portfolio data in the received offers
transcript_completed_viewed = transcript_completed_viewed.merge(portfolio.loc[:,['offer_id','offer_type','duration','ch_email','ch_mobile','ch_social','ch_web']],how = 'left',on=['offer_id'])
#filter out informational offers which can not be completed by nature
transcript_completed_viewed = transcript_completed_viewed.loc[transcript_completed_viewed.offer_type != 'informational',:]
#get dummies for offer type
transcript_completed_viewed = pd.concat([transcript_completed_viewed, pd.get_dummies(transcript_completed_viewed["offer_type"], dtype=int)], axis=1)

#drop nas
transcript_completed_viewed = transcript_completed_viewed.dropna()


###########
#Model 1

#split data into features (X) and endogenous variable (y)
X = transcript_received_viewed.drop(columns=['user_id','offer_id','gender','offer_viewed','offer_type'])
y = transcript_received_viewed.offer_viewed

#randomly assign data to train or test sample. stratify on y to guarantee same distribution in training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#set up components for Pipeline
pca = PCA()
scaler = StandardScaler()
clf = RandomForestClassifier()

pipeline = Pipeline([
            ('pca', pca),
            ('scaler', scaler),
            ('clf', clf)
        ])

#define parameter space for grid search
param_grid = {
            #"pca__n_components": [2, 5],
            'clf__n_estimators': [50, 100],
            'clf__min_samples_split': [3, 4],
            'clf__max_depth': [3,5,10],
            'clf__class_weight': ['balanced']
        }

#searching for best parameter set by cross-validation and fit that model on training data
cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
cv.fit(X_train,y_train)
print("Best parameter (CV score=%0.3f):" % cv.best_score_)
print(cv.best_params_)

#show some accuracy measures
y_pred_test = cv.predict(X_test)
print(classification_report(y_test, y_pred_test))

#dump model pickle file into repository
file_name = 'viewer_model.pkl'
pickle.dump(cv, open(file_name, 'wb'))

###########
#Model 2

#split data into features (X) and endogenous variable (y)
X = transcript_completed_viewed.drop(columns=['user_id','offer_id','gender','offer_viewed','offer_type'])
y = transcript_completed_viewed.offer_viewed

#randomly assign data to train or test sample. stratify on y to guarantee same distribution in training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#set up components for Pipeline
pca = PCA()
scaler = StandardScaler()
clf = RandomForestClassifier()

pipeline = Pipeline([
            ('pca', pca),
            ('scaler', scaler),
            ('clf', clf)
        ])

#define parameter space for grid search
param_grid = {
            #"pca__n_components": [2, 5],
            'clf__n_estimators': [50, 100],
            'clf__min_samples_split': [3, 4],
            'clf__max_depth': [3, 5, 10],
            'clf__class_weight': ['balanced']
        }

#searching for best parameter set by cross-validation and fit that model on training data
cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
cv.fit(X_train,y_train)
print("Best parameter (CV score=%0.3f):" % cv.best_score_)
print(cv.best_params_)

#show some accuracy measures
y_pred_test = cv.predict(X_test)
print(classification_report(y_test, y_pred_test))

#dump model pickle file into repository
file_name = 'completer_model.pkl'
pickle.dump(cv, open(file_name, 'wb'))