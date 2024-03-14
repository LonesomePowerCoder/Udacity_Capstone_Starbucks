import json
import plotly
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from matplotlib.figure import Figure
import base64
from io import BytesIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from flask import Flask
from flask import render_template, request, jsonify
import data.process_data as process_data
import datetime

#from plotly.graph_objs import Bar
#from sklearn.externals import joblib
#from sqlalchemy import create_engine

DP = process_data.data_processor()

portfolio = DP.unziper('../data/portfolio.zip','portfolio.json')
profile = DP.unziper('../data/profile.zip','profile.json')
transcript = DP.unziper('../data/transcript.zip','transcript.json')

##### Tidy Data
#portfolio
#portfolio['duration_in_days'] = pd.to_timedelta(portfolio['duration'], unit = 'D')
#portfolio['duration_in_hours'] = portfolio['duration']*24

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
portfolio = pd.concat([portfolio, pd.get_dummies(portfolio["offer_type"], dtype=int)], axis=1)
portfolio.rename(columns={"id":"offer_id"},inplace=True)

#profile
profile = profile.loc[profile.gender.notna(),:]
profile["became_member_on"] = pd.to_datetime(profile["became_member_on"], format='%Y%m%d')
today = profile["became_member_on"].max() + datetime.timedelta(days=1)
profile["member_days"] = (today - profile["became_member_on"]).dt.days
profile.rename(columns={"id":"user_id"},inplace= True)
profile['income'] = profile.income.fillna(profile.income.mean())

#transcript
transcript_received = transcript.loc[transcript.event == 'offer received',:]
transcript_received = transcript_received.rename(columns={'value.offer id': 'offer_id'})

transcript_viewed = transcript.loc[transcript.event == 'offer viewed',:]
transcript_viewed = transcript_viewed.rename(columns={'value.offer id': 'offer_id'})

transcript_viewed["cumcount_views"] = 1
transcript_viewed.sort_values(['person','time'],inplace = True)
transcript_viewed["cumsum_views"] = transcript_viewed.groupby(['person'])["cumcount_views"].cumsum()

transcript_received_viewed = transcript_received.merge(transcript_viewed.loc[:,['person','time','offer_id','cumsum_views']],how = 'left',on=['person','offer_id'])

transcript_received_viewed = transcript_received_viewed.rename(columns={'time_y': 'offer_viewed','person': 'user_id'})
transcript_received_viewed['offer_viewed'] = transcript_received_viewed.apply(lambda row: 0 if pd.isnull(row['offer_viewed']) else 1, axis=1)

transcript_received_viewed.sort_values(['user_id','time_x'], inplace = True)
transcript_received_viewed['cumsum_views'] = transcript_received_viewed['cumsum_views']-1
transcript_received_viewed['cumsum_views'] = transcript_received_viewed.groupby('user_id')['cumsum_views'].fillna(method = 'ffill')

transcript_received_viewed = transcript_received_viewed.drop(columns=['time_x','event'])

transcript_received_viewed = transcript_received_viewed.merge(profile.loc[:,['user_id','age','gender','income','member_days']],how = 'left',on=['user_id'])
transcript_received_viewed = pd.concat([transcript_received_viewed, pd.get_dummies(transcript_received_viewed["gender"], dtype=int)], axis=1)
#transcript_received_viewed = transcript_received_viewed.dropna()
transcript_received_viewed = transcript_received_viewed.merge(portfolio.loc[:,['offer_id','offer_type','duration','ch_email','ch_mobile','ch_social','ch_web']],how = 'left',on=['offer_id'])
transcript_received_viewed = pd.concat([transcript_received_viewed, pd.get_dummies(transcript_received_viewed["offer_type"], dtype=int)], axis=1)













transcript_completed = transcript.loc[transcript.event == 'offer completed',:]
transcript_completed = transcript_completed.rename(columns={'value.offer id': 'offer_id'})

transcript_completed_viewed = transcript_completed.merge(transcript_viewed.loc[:,['person','time','offer_id','cumsum_views']],how = 'left',on=['person','offer_id'])

transcript_completed_viewed = transcript_completed_viewed.rename(columns={'time_y': 'offer_viewed','person': 'user_id'})
transcript_completed_viewed['offer_viewed'] = transcript_completed_viewed.apply(lambda row: 0 if pd.isnull(row['offer_viewed']) else 1, axis=1)

transcript_completed_viewed.sort_values(['user_id','time_x'], inplace = True)
transcript_completed_viewed['cumsum_views'] = transcript_completed_viewed['cumsum_views']-1
transcript_completed_viewed['cumsum_views'] = transcript_completed_viewed.groupby('user_id')['cumsum_views'].fillna(method = 'ffill')

transcript_completed_viewed = transcript_completed_viewed.drop(columns=['time_x','event'])

transcript_completed_viewed = transcript_completed_viewed.merge(profile.loc[:,['user_id','age','gender','income','member_days']],how = 'left',on=['user_id'])
transcript_completed_viewed = pd.concat([transcript_completed_viewed, pd.get_dummies(transcript_completed_viewed["gender"], dtype=int)], axis=1)
#transcript_received_viewed = transcript_received_viewed.dropna()
transcript_completed_viewed = transcript_completed_viewed.merge(portfolio.loc[:,['offer_id','offer_type','duration','ch_email','ch_mobile','ch_social','ch_web']],how = 'left',on=['offer_id'])
transcript_completed_viewed = transcript_completed_viewed.loc[transcript_completed_viewed.offer_type != 'informational',:]
transcript_completed_viewed = pd.concat([transcript_completed_viewed, pd.get_dummies(transcript_completed_viewed["offer_type"], dtype=int)], axis=1)

###########
#Model 1
X = transcript_received_viewed.drop(columns=['user_id','offer_id','gender','offer_viewed','offer_type','value.reward', 'value.amount'])
y = transcript_received_viewed.offer_viewed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# pca = PCA()
# scaler = StandardScaler()
# clf = RandomForestClassifier()
#
# pipeline = Pipeline([
#             ('pca', pca),
#             ('scaler', scaler),
#             ('clf', clf)
#         ])
#
# param_grid = {
#             #"pca__n_components": [2, 5],
#             'clf__n_estimators': [100],#, 200],
#             'clf__min_samples_split': [3],#, 4],
#             'clf__max_depth': [3],#,5,10],
#             'clf__class_weight':['balanced']
#         }
#
# cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
# cv.fit(X_train,y_train)
# print("Best parameter (CV score=%0.3f):" % cv.best_score_)
# print(cv.best_params_)
#
# y_pred_test = cv.predict(X_test)

clf = RandomForestClassifier(max_depth=10, n_estimators = 100, random_state=0, class_weight='balanced_subsample')
clf.fit(X, y)
y_pred_test = clf.predict(X_test)

print(classification_report(y_test, y_pred_test))

###########
#Model 2

X = transcript_completed_viewed.drop(columns=['user_id','offer_id','gender','offer_viewed','offer_type','value.reward', 'value.amount'])
y = transcript_completed_viewed.offer_viewed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# pca = PCA()
# scaler = StandardScaler()
# clf = RandomForestClassifier()
#
# pipeline = Pipeline([
#             ('pca', pca),
#             ('scaler', scaler),
#             ('clf', clf)
#         ])
#
# param_grid = {
#             #"pca__n_components": [2, 5],
#             'clf__n_estimators': [100, 200],
#             'clf__min_samples_split': [3, 4],
#             'clf__max_depth': [3,5,10],
#             'clf__class_weight':['balanced']
#         }
#
# cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
# cv.fit(X_train,y_train)
# print("Best parameter (CV score=%0.3f):" % cv.best_score_)
# print(cv.best_params_)
#
# y_pred_test = cv.predict(X_test)

clf = RandomForestClassifier(max_depth=10, n_estimators = 100, random_state=0, class_weight='balanced_subsample')
clf.fit(X, y)
y_pred_test = clf.predict(X_test)

print(classification_report(y_test, y_pred_test))

##Web app part



# portfolio['ch_email'] = 0
# portfolio['ch_web'] = 0
# portfolio['ch_mobile'] = 0
# portfolio['ch_social'] = 0
#
# for i in range(portfolio.shape[0]):
#     if 'email' in portfolio.iloc[i].channels:
#         portfolio.loc[i,'ch_email'] = 1
#     if 'web' in portfolio.iloc[i].channels:
#         portfolio.loc[i,'ch_web'] = 1
#     if 'mobile' in portfolio.iloc[i].channels:
#         portfolio.loc[i,'ch_mobile'] = 1
#     if 'social' in portfolio.iloc[i].channels:
#         portfolio.loc[i,'ch_social'] = 1
#
# portfolio.drop(columns=['channels'], inplace=True)
#
# portfolio['duration_in_days'] = pd.to_timedelta(portfolio['duration'], unit = 'D')
#


# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens
#
#
# # load data
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
# df = pd.read_sql_table('DisasterResponse', engine)
#
# # load model
# model = joblib.load("../models/classifier.pkl")

#web app part
# app = Flask(__name__, template_folder='template')
# @app.route('/')
# def hello():
#
#     def figure1():
#         # Generate the figure **without using pyplot**.
#         fig = Figure()
#         ax = fig.subplots()
#         ax.plot([1, 2])
#         # Save it to a temporary buffer.
#         buf = BytesIO()
#         fig.savefig(buf, format="png")
#         # Embed the result in the html output.
#         data = base64.b64encode(buf.getbuffer()).decode("ascii")
#         return f"<img src='data:image/png;base64,{data}'/>"
#
#     fig = figure1()
#     return render_template('index.html', plot=fig)
#
# @app.route('/index')
# def index():
#
#     def con_matrix():
#         # Define the confusion matrix and labels
#         confusion_matrix = [[10, 2, 3], [4, 5, 6], [7, 8, 9]]
#         labels = ['Class 1', 'Class 2', 'Class 3']
#
#         # Create the heatmap trace
#         heatmap = go.Heatmap(z=confusion_matrix, x=labels, y=labels, colorscale='Viridis')
#
#         # Create the figure and add the heatmap trace
#         fig = go.Figure(data=[heatmap])
#
#         # Add annotations for each cell in the confusion matrix
#         for i in range(len(confusion_matrix)):
#             for j in range(len(confusion_matrix[i])):
#                 fig.add_annotation(
#                     text=confusion_matrix[i][j],
#                     x=labels[j],
#                     y=labels[i],
#                     xref='x',
#                     yref='y'
#                 )
#
#         # Show the figure
#         fig.show()
#
#     confusion_matrix = con_matrix()
#     return render_template('index.html', plot=confusion_matrix)
#
# # web page that handles user query and displays model results
# @app.route('/Project')
# def Project():
#
#     # This will render the go.html Please see that file.
#     return render_template('Project.html')
#
#
# #def main():
# #    app.run(host='0.0.0.0', port=3001, debug=True)
#
#
# if __name__ == '__main__':
#     app.run()