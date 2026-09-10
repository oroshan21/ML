import nflreadpy as nfl

#print(schedules.head(10))

schedules = nfl.load_schedules(range(2015, 2025))

pbp = nfl.load_pbp(range(2015,2025))

player_stats = nfl.load_player_stats(range(2015,2025))

schedules_pandas = schedules.to_pandas()

pbp_pandas = pbp.to_pandas()

player_stats_pandas = player_stats.to_pandas()

#print(pbp.head(5))

team_game_off_epa = pbp_pandas.groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index()
team_game_off_epa = team_game_off_epa.sort_values(['posteam', 'season', 'week'])


team_game_def_epa = pbp_pandas.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index()
team_game_def_epa = team_game_def_epa.sort_values(['defteam', 'season', 'week'])

# print(team_game_def_epa.columns)

#EPA prior to start of current game, based off 8 prior games
team_game_off_epa['rolling_off_epa'] = (    
    team_game_off_epa.groupby(['season','posteam'])['epa']
    .transform(lambda x : x.shift(1).rolling(8, min_periods = 1).mean())
)

team_game_def_epa['rolling_def_epa'] = (
    team_game_def_epa.groupby(['season','defteam'])['epa']
    .transform(lambda x : x.shift(1).rolling(8, min_periods = 1).mean())
)

home_off_epa = team_game_off_epa.rename(
    columns = {
        'posteam' : 'home_off_team',
        'rolling_off_epa': 'home_rolling_off_epa'
    }
)[['season','week','home_off_team','home_rolling_off_epa']]

away_off_epa = team_game_off_epa.rename(
    columns = {
        'posteam' : 'away_off_team',
        'rolling_off_epa': 'away_rolling_off_epa'
    }
)[['season','week','away_off_team','away_rolling_off_epa']]

home_def_epa = team_game_def_epa.rename(
    columns = {
        'defteam' : 'home_def_team',
        'rolling_def_epa': 'home_rolling_def_epa'
    }
)[['season','week','home_def_team','home_rolling_def_epa']]

away_def_epa = team_game_off_epa.rename(
    columns = {
        'defteam' : 'away_def_team',
        'rolling_def_epa': 'away_rolling_def_epa'
    }
)[['season','week','away_def_team','away_rolling_def_epa']]

games = schedules_pandas.merge(
    home_off_epa,
    on=['season','week','home_off_team'],
    how = 'left'
)
games = games.merge(
    away_off_epa,
    on=['season','week','away_off_team'],
    how = 'left'
)
games = games.merge(
    home_def_epa,
    on=['season','week','home_def_team'],
    how = 'left'
)
games = games.merge(
    away_def_epa,
    on=['season','week','away_def_team'],
    how = 'left'
)

games['epa_diff'] = (
    games['home_rolling_epa'] - games['away_rolling_epa'])

games['home_win'] = (
     games['home_score'] > games['away_score']
).astype(int)

features = [
    'home_rolling_off_epa',
    'away_rolling_off_epa',
    'epa_diff',
    'home_rolling_def_epa',
    'away_rolling_def_epa'
]
model_data = games.dropna(
    subset=features + ['home_win']).copy()

train = model_data[
    model_data['season'] <= 2022
]
test = model_data[
    model_data['season'] > 2022
]

Xtrain = train[features]
Ytrain = train["home_win"]

X_test = test[features]
y_test = test["home_win"]

from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators = 300,
    max_depth = 3,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "logloss",
    random_state=42
)
model.fit(Xtrain,Ytrain)

probs = model.predict_proba(X_test)[:,1]
preds = (probs >= 0.5).astype(int)

print("Log loss:", log_loss(y_test, probs))
print("Brier score:", brier_score_loss(y_test, probs))
print("Accuracy:", accuracy_score(y_test, preds))