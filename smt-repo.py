#%%
#neccesary import statements
import sportypy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as pads
import pyarrow as pa
import os
import duckdb 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import shap
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_hastie_10_2
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from xgboost.callback import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sportypy.surfaces import MiLBField
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML
import seaborn as sns
#%%
#
db_path = r'C:\Users\CENSORED\Downloads\smt_2025.db'
con = duckdb.connect(db_path)
#%%
query = """
WITH mapped_players AS (
    -- maps baserunner pos to player ids
    SELECT
        p.game_str,
        p.play_id,
        p.timestamp,
        p.field_x AS x,
        p.field_y AS y,
        CASE p.player_position
            WHEN 11 THEN g.first_baserunner
            WHEN 12 THEN g.second_baserunner
            WHEN 13 THEN g.third_baserunner
        END AS player_id
    FROM player_pos p
    JOIN game_info g ON p.game_str = g.game_str AND p.play_id = g.play_per_game
    WHERE
        p.player_position IN (11, 12, 13)
        AND CASE p.player_position
            WHEN 11 THEN g.first_baserunner
            WHEN 12 THEN g.second_baserunner
            WHEN 13 THEN g.third_baserunner
        END IS NOT NULL
        AND CASE p.player_position
            WHEN 11 THEN g.first_baserunner
            WHEN 12 THEN g.second_baserunner
            WHEN 13 THEN g.third_baserunner
        END <> 'NA'
),
--gets the speed of the players as well as the frame to frame distance
play_distance_and_speed AS (
    SELECT
        game_str,
        player_id,
        play_id,

        -- calculate speed over 1 sec using the data
        (
            SQRT(
                (x - FIRST_VALUE(x) OVER w_1_sec) * (x - FIRST_VALUE(x) OVER w_1_sec) +
                (y - FIRST_VALUE(y) OVER w_1_sec) * (y - FIRST_VALUE(y) OVER w_1_sec)
            ) / NULLIF((timestamp - FIRST_VALUE(timestamp) OVER w_1_sec), 0) * 1000
        ) AS speed_in_window,

        -- filter out speeds from incomplete windows
        (timestamp - FIRST_VALUE(timestamp) OVER w_1_sec) AS time_in_window,

        -- sum the segments for total distance
        SQRT(
            (x - LAG(x, 1, x) OVER w_prev_frame) * (x - LAG(x, 1, x) OVER w_prev_frame) +
            (y - LAG(y, 1, y) OVER w_prev_frame) * (y - LAG(y, 1, y) OVER w_prev_frame)
        ) AS distance_segment

    FROM mapped_players
    -- look back 1 second
    WINDOW w_1_sec AS (
        PARTITION BY game_str, player_id, play_id
        ORDER BY timestamp
        RANGE BETWEEN 1000 PRECEDING AND CURRENT ROW
    ),
    -- looks at previous row
    w_prev_frame AS (
        PARTITION BY game_str, player_id, play_id
        ORDER BY timestamp
    )
),

play_summary AS (
    -- aggregates by play
    SELECT
        game_str,
        player_id,
        play_id,
        -- get the max speed, filtering out for <750ms
        MAX(CASE WHEN time_in_window >= 750 THEN speed_in_window ELSE NULL END) AS max_speed_on_play,
        -- sum the small segmetnws for total distance
        SUM(distance_segment) AS total_distance_on_play
    FROM play_distance_and_speed
    GROUP BY game_str, player_id, play_id
),

filtered_top_speeds AS (
    -- filters speed for unreqalistic speeds
    SELECT
        player_id,
        max_speed_on_play
    FROM play_summary
    WHERE
        total_distance_on_play >= 45 -- makes sure player moved at least 45 feet
        AND max_speed_on_play IS NOT NULL
        AND max_speed_on_play < 40
        AND max_speed_on_play > 20
)

-- finds 80th percentile speed
SELECT
    player_id AS runner_id,
    PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY max_speed_on_play) AS sprint_speed,
    COUNT(*) AS num_plays_measured
FROM filtered_top_speeds
GROUP BY player_id
ORDER BY sprint_speed DESC;
"""
#drops players with less than 3
df_sprint_speeds = con.sql(query).df()
sprint_speed_df = df_sprint_speeds[(df_sprint_speeds['num_plays_measured']>=3)].reset_index(drop=True)
sprint_speed_df
#%%
query="""WITH steal_throws AS (
  --  finds all plays with event code 3
  SELECT DISTINCT
    ge.game_str, 
    ge.play_id, 
    ge.play_per_game 
  FROM game_events ge 
  WHERE 
    ge.event_code = 3 -- throw (ball-in-play) 
), 
steal_throws_with_catcher AS ( 
  -- join info to get the catchers id
  SELECT 
    st.game_str, 
    st.play_id, 
    gi.catcher AS catcher_id 
  FROM steal_throws st 
  JOIN game_info gi 
    ON st.game_str = gi.game_str 
    AND st.play_per_game = gi.play_per_game 
  WHERE
    gi.catcher IS NOT NULL AND gi.catcher != 'NA'
), 
catcher_acquisition AS (
  -- find first moment catcher acquires the ball
  SELECT
    ge.game_str,
    ge.play_id,
    MIN(ge.timestamp) AS catcher_time
  FROM game_events ge
  WHERE
    ge.event_code = 2 -- ball acquired
    AND ge.player_position = 2 -- catcher
  GROUP BY
    ge.game_str, ge.play_id
),
infielder_acquisition AS (
  --find first moment the middle infielder acquires the ball
  SELECT
    ge.game_str,
    ge.play_id,
    MIN(ge.timestamp) AS tag_time
  FROM game_events ge
  WHERE
    ge.event_code = 2 -- ball acquired
    AND ge.player_position IN (4, 6) -- second baseman or shortstop
  GROUP BY
    ge.game_str, ge.play_id
)
--join these events and calculate the pop time
SELECT 
  s.game_str, 
  s.play_id, 
  s.catcher_id, 
  (t.tag_time - c.catcher_time) / 1000.0 AS pop_time_sec 
FROM steal_throws_with_catcher s 
-- join catcher acq
JOIN catcher_acquisition c ON 
  s.game_str = c.game_str 
  AND s.play_id = c.play_id 
-- join infielder acq
JOIN infielder_acquisition t ON 
  s.game_str = t.game_str 
  AND s.play_id = t.play_id 
WHERE 
  t.tag_time > c.catcher_time -- make sure tag happens after the catch
ORDER BY 
  pop_time_sec;
"""
#filters for reasonable times and enough sample
df_pop_times = con.sql(query).df()
df_pop_times = df_pop_times.drop_duplicates(subset=['game_str', 'play_id', 'catcher_id'], keep='first')
df_pop_times = df_pop_times[(df_pop_times['pop_time_sec']>=1.6) & (df_pop_times['pop_time_sec']<=2.5)]
catcher_poptimes_df = df_pop_times.groupby('catcher_id').apply(lambda df: pd.Series({
  'average_pop_time': df['pop_time_sec'].mean(),
  'total_throws': len(df[['game_str', 'play_id']].drop_duplicates())
})).reset_index()
catcher_poptimes_df = catcher_poptimes_df[(catcher_poptimes_df['total_throws'] >= 3)]
catcher_poptimes_df = catcher_poptimes_df.sort_values(by='average_pop_time').reset_index(drop=True)
#%%
catcher_poptimes_df
#%%
query1 = """WITH filtered_game_info AS (
  SELECT *
  FROM game_info
  WHERE play_per_game IS NOT NULL
),
--finds the pickoff attempts with event codes
pickoff_attempts AS (
  SELECT
    ge.game_str,
    ge.play_id,
    gi.first_baserunner AS runner_id
  FROM game_events ge
  JOIN filtered_game_info gi
    ON ge.game_str = gi.game_str AND ge.play_id = gi.play_per_game
  WHERE
    ge.event_code = 6
    AND NOT EXISTS (
      SELECT 1
      FROM game_events ge2
      WHERE ge.game_str = ge2.game_str
        AND ge.play_id = ge2.play_id
        AND ge2.event_code = 1
    )
    AND (gi.first_baserunner != 'NA' AND gi.first_baserunner IS NOT NULL)
    AND (gi.second_baserunner = 'NA' OR gi.second_baserunner IS NULL)
    AND (gi.third_baserunner = 'NA' OR gi.third_baserunner IS NULL)
),
-- gets the pos of the pitcher
pitcher_pos AS (
    SELECT
        game_str,
        play_id,
        timestamp,
        field_x,
        field_y
    FROM player_pos
    WHERE player_position = 1
    AND EXISTS (
        SELECT 1 FROM pickoff_attempts pa
        WHERE pa.game_str = player_pos.game_str AND pa.play_id = player_pos.play_id
    )
),
--gets distance of pitcher and ball
ball_pitcher_distance AS (
    SELECT
        bp.game_str,
        bp.play_id,
        bp.timestamp,
        SQRT(POWER(bp.ball_position_x - pp.field_x, 2) + POWER(bp.ball_position_y - pp.field_y, 2)) as distance_from_pitcher
    FROM ball_pos bp
    JOIN pitcher_pos pp
        ON bp.game_str = pp.game_str
        AND bp.play_id = pp.play_id
        AND bp.timestamp = pp.timestamp
),
--gets teh moment the pitcher released the ball
release_time AS (
    SELECT
        game_str,
        play_id,
        MIN(timestamp) AS release_timestamp
    FROM ball_pitcher_distance
    WHERE distance_from_pitcher >= 1.0
    GROUP BY game_str, play_id
),
--gets the kinematic data of the runner over 500ms, 1000ms, and 1500ms timeframes
runner_kinematics AS (
  SELECT
    pp.game_str,
    pp.play_id,
    pp.timestamp,
    pp.field_x,
    pp.field_y,
    SQRT(POWER(pp.field_x - 45 * SQRT(2), 2) + POWER(pp.field_y - 45 * SQRT(2), 2)) AS lead_distance,
    -- get position and time from 500ms
    (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_500ms_x,
    (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_500ms_y,
    (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_500ms_timestamp,
    -- get position and time from 1000ms
    (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_1000ms_x,
    (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_1000ms_y,
    (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_1000ms_timestamp,
    -- get position and time from 1500ms
    (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_1500ms_x,
    (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_1500ms_y,
    (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = pp.player_position AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_1500ms_timestamp
  FROM player_pos pp
  WHERE EXISTS (
      SELECT 1 FROM pickoff_attempts pa
      WHERE pa.game_str = pp.game_str AND pa.play_id = pp.play_id
  )
  AND pp.player_position = 11
),
--finds the max lead from this sample
max_lead_per_play AS (
  SELECT
    game_str,
    play_id,
    MAX(lead_distance) AS max_lead_distance
  FROM runner_kinematics
  GROUP BY game_str, play_id
),
--calculates the kinematic data
runner_kinematics_calculated AS (
  SELECT
    *,
    -- find lead distance from 500, 1000, and 1500ms
    SQRT(POWER(prev_500ms_x - 45 * SQRT(2), 2) + POWER(prev_500ms_y - 45 * SQRT(2), 2)) AS prev_500ms_lead_distance,
    SQRT(POWER(prev_1000ms_x - 45 * SQRT(2), 2) + POWER(prev_1000ms_y - 45 * SQRT(2), 2)) AS prev_1000ms_lead_distance,
    SQRT(POWER(prev_1500ms_x - 45 * SQRT(2), 2) + POWER(prev_1500ms_y - 45 * SQRT(2), 2)) AS prev_1500ms_lead_distance
  FROM runner_kinematics
),
--finds the velocity based on our position data
velocity_calculated AS (
  SELECT
    *,
    -- last .5 sec
    CASE
      WHEN prev_500ms_timestamp IS NULL OR timestamp = prev_500ms_timestamp THEN NULL
      ELSE (lead_distance - prev_500ms_lead_distance) / ((timestamp - prev_500ms_timestamp) / 1000.0)
    END AS velocity,
    -- .5-1 sec
    CASE
      WHEN prev_1000ms_timestamp IS NULL OR prev_500ms_timestamp = prev_1000ms_timestamp THEN NULL
      ELSE (prev_500ms_lead_distance - prev_1000ms_lead_distance) / ((prev_500ms_timestamp - prev_1000ms_timestamp) / 1000.0)
    END AS prev_velocity,
    -- 1-1.5 sec
    CASE
      WHEN prev_1500ms_timestamp IS NULL OR prev_1000ms_timestamp = prev_1500ms_timestamp THEN NULL
      ELSE (prev_1000ms_lead_distance - prev_1500ms_lead_distance) / ((prev_1000ms_timestamp - prev_1500ms_timestamp) / 1000.0)
    END AS prev_prev_velocity
  FROM runner_kinematics_calculated
),
target_moment AS (
    SELECT
        vc.game_str,
        vc.play_id,
        vc.lead_distance,
        vc.velocity,
        -- calculate accel based on last .5 sec velocity
        CASE
          WHEN vc.velocity IS NULL OR vc.prev_velocity IS NULL OR vc.timestamp = vc.prev_500ms_timestamp THEN NULL
          ELSE (vc.velocity - vc.prev_velocity) / ((vc.timestamp - vc.prev_500ms_timestamp) / 1000.0)
        END AS acceleration,
        -- calculate jerk based off our change in accel
        CASE
          WHEN vc.velocity IS NULL OR vc.prev_velocity IS NULL OR vc.prev_prev_velocity IS NULL OR vc.timestamp = vc.prev_500ms_timestamp OR vc.prev_500ms_timestamp = vc.prev_1000ms_timestamp THEN NULL
          ELSE (
            ((vc.velocity - vc.prev_velocity) / ((vc.timestamp - vc.prev_500ms_timestamp) / 1000.0)) -
            ((vc.prev_velocity - vc.prev_prev_velocity) / ((vc.prev_500ms_timestamp - vc.prev_1000ms_timestamp) / 1000.0))
          ) / ((vc.timestamp - vc.prev_500ms_timestamp) / 1000.0)
        END AS jerk
    FROM velocity_calculated vc
    JOIN release_time rt
        ON vc.game_str = rt.game_str
        AND vc.play_id = rt.play_id
    QUALIFY ROW_NUMBER() OVER(PARTITION BY vc.game_str, vc.play_id ORDER BY ABS(vc.timestamp - (rt.release_timestamp - 725)) ASC, vc.timestamp ASC) = 1
),
--filters for valid pickoff plays
valid_pickoffs AS (
  SELECT
    pa.game_str,
    pa.play_id,
    pa.runner_id,
    tm.lead_distance,
    tm.velocity,
    tm.acceleration,
    tm.jerk
  FROM pickoff_attempts pa
  JOIN target_moment tm
    ON pa.game_str = tm.game_str AND pa.play_id = tm.play_id
  JOIN max_lead_per_play mlpp
    ON pa.game_str = mlpp.game_str AND pa.play_id = mlpp.play_id
)
-- selects data and sets success of pickoffs
SELECT
  vp.game_str,
  vp.play_id,
  vp.runner_id,
  vp.lead_distance,
  vp.velocity,
  vp.acceleration,
  vp.jerk,
  CASE
    WHEN gi_next.play_per_game IS NULL THEN -1
    WHEN (gi_next.first_baserunner != 'NA' AND gi_next.first_baserunner IS NOT NULL)
      OR (gi_next.second_baserunner != 'NA' AND gi_next.second_baserunner IS NOT NULL)
      OR (gi_next.third_baserunner != 'NA' AND gi_next.third_baserunner IS NOT NULL)
    THEN 0
    ELSE 1
  END AS pickoff_result
FROM valid_pickoffs vp
LEFT JOIN filtered_game_info gi_next
  ON vp.game_str = gi_next.game_str
  AND CAST(vp.play_id AS INTEGER) + 1 = gi_next.play_per_game
ORDER BY
  vp.game_str,
  vp.play_id,
  vp.runner_id,
  vp.lead_distance,
  vp.velocity,
  vp.acceleration,
  vp.jerk;
"""
#merges sprint speed
df_pickoff_results = con.sql(query1).df()
df_pickoff_results = df_pickoff_results[df_pickoff_results['pickoff_result'] != -1].drop_duplicates(subset=['game_str', 'play_id'], keep='first')
print(str(len(df_pickoff_results[df_pickoff_results['pickoff_result']==1]))+"/"+str(len(df_pickoff_results)))
sprint_speeds_to_merge = sprint_speed_df[['runner_id', 'sprint_speed']]
df_pickoffs_with_sprint_speed = pd.merge(
    df_pickoff_results,
    sprint_speeds_to_merge,
    on='runner_id',
    how='left'
)
#creates distribution figures
columns_to_drop = ['game_str', 'play_id', 'runner_id']
successful_pickoffs_df = df_pickoffs_with_sprint_speed[df_pickoffs_with_sprint_speed['pickoff_result']==1]
pickoff_chance_input = df_pickoffs_with_sprint_speed.drop(columns=columns_to_drop).reset_index(drop=True)
titles = ['Distribution of Lead Distance', 'Distribution of Velocity', 'Distribution of Acceleration', 'Distribution of Jerk', 'Distribution of Sprint Speed']
x_labels = ['Lead Distance (ft)', 'Velocity (ft/s)', 'Acceleration (ft/s^2)', "Jerk (ft/s^3)", 'Sprint Speed (ft/s)']
#creates graphic
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(['lead_distance', 'velocity', 'acceleration', 'jerk','sprint_speed']):
    data_to_plot = df_pickoffs_with_sprint_speed[col].dropna()
    
    axes[i].hist(data_to_plot, bins=1000, color='skyblue', edgecolor='black')
    axes[i].set_title(titles[i], fontsize=14)
    axes[i].set_xlabel(x_labels[i], fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()
#imputes medians when we lack enough sample
df_model_data = pickoff_chance_input.copy()
features = ['lead_distance', 'velocity', 'acceleration', 'jerk','sprint_speed']
target = 'pickoff_result'
X = df_model_data[features]
y = df_model_data[target]
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
#cross validation
print("Repeated Cross-Validation for Performance Estimation")
n_repeats = 1
n_splits = 5
n_estimators = 1000
early_stopping_rounds_boosting = 100

#storage for models
lgbm_roc_auc_scores_for_this_run = []
xgb_roc_auc_scores_for_this_run = []
rf_roc_auc_scores_for_this_run = []
lasso_roc_auc_scores_for_this_run = []
ridge_roc_auc_scores_for_this_run = []
all_lgbm_best_iterations = []
all_xgb_best_iterations = []

# main cross validation loop
for i in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        # data splitting
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)

        # convert back to df for xgb and lgbm
        X_train_df = pd.DataFrame(X_train_imputed, columns=features)
        X_val_df = pd.DataFrame(X_val_imputed, columns=features)

        # find class weights
        count_negative = np.sum(y_train == 0)
        count_positive = np.sum(y_train == 1)
        scale_pos_weight_value = count_negative / count_positive if count_positive > 0 else 1

        # lgbm
        lgbm_model = lgb.LGBMClassifier(objective='binary', metric='auc', scale_pos_weight=scale_pos_weight_value, n_estimators=n_estimators, verbose=-1, random_state=i)
        lgbm_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds_boosting, verbose=False)])
        all_lgbm_best_iterations.append(lgbm_model.best_iteration_ if lgbm_model.best_iteration_ is not None else n_estimators)
        y_pred_proba_lgbm = lgbm_model.predict_proba(X_val_df)[:, 1]
        lgbm_roc_auc_scores_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lgbm))

        # xgb
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=scale_pos_weight_value, n_estimators=n_estimators, random_state=i, early_stopping_rounds=early_stopping_rounds_boosting)
        xgb_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)],  verbose=False)
        all_xgb_best_iterations.append(xgb_model.best_iteration)
        y_pred_proba_xgb = xgb_model.predict_proba(X_val_df)[:, 1]
        xgb_roc_auc_scores_for_this_run.append(roc_auc_score(y_val, y_pred_proba_xgb))

        # random forest
        rf_model = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=i, n_jobs=-1)
        rf_model.fit(X_train_imputed, y_train)
        y_pred_proba_rf = rf_model.predict_proba(X_val_imputed)[:, 1]
        rf_roc_auc_scores_for_this_run.append(roc_auc_score(y_val, y_pred_proba_rf))

        # lasso regression
        lasso_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LogisticRegression(penalty='l1', solver='liblinear', random_state=i))])
        lasso_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_lasso = lasso_pipeline.predict_proba(X_val_imputed)[:, 1]
        lasso_roc_auc_scores_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lasso))
         
        # ridge regression
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', LogisticRegression(penalty='l2', random_state=i))])
        ridge_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_ridge = ridge_pipeline.predict_proba(X_val_imputed)[:, 1]
        ridge_roc_auc_scores_for_this_run.append(roc_auc_score(y_val, y_pred_proba_ridge))
# final results
print("\n--- Median Pickoff Model Performance Across All Repeats ---")
print(f"Median LightGBM ROC-AUC: {np.median(lgbm_roc_auc_scores_for_this_run):.4f}  (standard deviation {np.std(lgbm_roc_auc_scores_for_this_run):.4f})")
print(f"Median XGBoost ROC-AUC:  {np.median(xgb_roc_auc_scores_for_this_run):.4f} (standard deviation {np.std(xgb_roc_auc_scores_for_this_run):.4f})")
print(f"Median Random Forest ROC-AUC: {np.median(rf_roc_auc_scores_for_this_run):.4f} (standard deviation {np.std(rf_roc_auc_scores_for_this_run):.4f})")
print(f"Median Lasso Regression ROC-AUC: {np.median(lasso_roc_auc_scores_for_this_run):.4f} (standard deviation {np.std(lasso_roc_auc_scores_for_this_run):.4f})")
print(f"Median Ridge Regression ROC-AUC: {np.median(ridge_roc_auc_scores_for_this_run):.4f} (standard deviation {np.std(ridge_roc_auc_scores_for_this_run):.4f})")
print("-" * 50)
print(f"Median LGBM Stopping Point: {np.median(all_lgbm_best_iterations):.0f} estimators")
print(f"Median XGBoost Stopping Point: {np.median(all_xgb_best_iterations):.0f} estimators")
#%%
#creates shap plot
X_percentile = pd.DataFrame(X_imputed, columns=features)
for col in X_percentile.columns:
    X_percentile[col] = X_percentile[col].rank(pct=True) * 100

formatted_feature_names = [name.replace('_', ' ').title() for name in X_percentile.columns]
X_percentile.columns = formatted_feature_names
shap.summary_plot(shap_values_pickoff, features=X_percentile, show=False, cmap='bwr')
plt.title("Factors Influencing the Likelihood of a Pickoff", size=16, pad=20)
plt.xlabel("Model Output Impact\n(Left = Lower chance  |  Right = Higher chance of Pickoff)")
plt.ylabel("Features (Sorted by Overall Importance)", size=10)
fig = plt.gcf()
colorbar_ax = fig.axes[-1]
colorbar_ax.set_ylabel("Feature Value", size=12)
plt.tight_layout()

plt.show()
#%%
query ="""WITH steal_outcomes AS (
    -- identify the steal attempts and their outcomes
    WITH initial_plays AS (
      SELECT DISTINCT
        gi.game_str, gi.play_per_game, ge.play_id, gi.first_baserunner AS runner_id,
        gi.catcher AS catcher_id, gi.top_bottom_inning
      FROM game_info AS gi
      JOIN game_events AS ge ON gi.game_str = ge.game_str AND gi.play_per_game = ge.play_per_game
      WHERE
        gi.first_baserunner IS NOT NULL AND gi.first_baserunner != 'NA'
        AND (gi.second_baserunner IS NULL OR gi.second_baserunner = 'NA')
        AND (gi.third_baserunner IS NULL OR gi.third_baserunner = 'NA')
        AND gi.play_per_game IS NOT NULL
        AND gi.catcher IS NOT NULL AND gi.catcher != 'NA'
    ),
    -- filter the plays to ensure we are only taking steals we want to train n
    filtered_plays AS (
      SELECT
        ip.game_str, ip.play_per_game, ip.play_id,
        ip.runner_id, ip.catcher_id, ip.top_bottom_inning
      FROM initial_plays AS ip
      WHERE EXISTS (
        SELECT 1 FROM game_events ge WHERE ge.game_str = ip.game_str AND ge.play_id = ip.play_id AND ge.event_code = 1 -- pitch
      )
      AND NOT EXISTS (
        SELECT 1 FROM game_events ge WHERE ge.game_str = ip.game_str AND ge.play_id = ip.play_id AND ge.event_code IN (4, 6, 10) -- ball in play, pickoff, deflection
      )
      AND NOT EXISTS (
        SELECT 1 FROM ball_pos bp WHERE bp.game_str = ip.game_str AND bp.play_id = ip.play_id AND bp.ball_position_y < -5 -- passed ball
      )
      -- ensures that the runner actually ran when their was a throw
      AND EXISTS (
          SELECT 1 FROM player_pos pp
          WHERE pp.game_str = ip.game_str
            AND pp.play_id = ip.play_id
            AND pp.player_position = 11 -- runner on first
            -- assuming home plate is (0,0)
            AND SQRT(POWER(field_x - 0, 2) + POWER(field_y - 120, 2)) < 45
      )
    ),
    distinct_play_sequence AS (
      -- sequence plays to compare their before and after states
      SELECT
        game_str, play_per_game,
        LEAD(play_per_game, 1) OVER (PARTITION BY game_str ORDER BY CAST(play_per_game AS INTEGER)) AS next_distinct_play
      FROM (SELECT DISTINCT game_str, play_per_game FROM game_info WHERE play_per_game IS NOT NULL)
    ),
    outcome_analysis AS (
      -- determine steal outcome by comparing runner pos at end of the play
      SELECT
        fp.game_str, fp.play_per_game, fp.play_id, fp.runner_id, fp.catcher_id,
        fp.top_bottom_inning AS initial_inning,
        next_play_info.first_baserunner AS next_first_baserunner,
        next_play_info.second_baserunner AS next_second_baserunner,
        next_play_info.third_baserunner AS next_third_baserunner,
        next_play_info.top_bottom_inning AS next_inning
      FROM filtered_plays AS fp
      JOIN distinct_play_sequence AS dps ON fp.game_str = dps.game_str AND fp.play_per_game = dps.play_per_game
      JOIN game_info AS next_play_info ON dps.game_str = next_play_info.game_str AND dps.next_distinct_play = next_play_info.play_per_game
    )
    SELECT
    --set outcomes
      game_str, play_per_game, play_id, runner_id, catcher_id,
      CASE
        WHEN (next_second_baserunner IS NOT NULL AND next_second_baserunner != 'NA') OR (next_third_baserunner IS NOT NULL AND next_third_baserunner != 'NA')
        THEN 0
        ELSE 1
      END AS outcome
    FROM outcome_analysis
    WHERE
      (next_first_baserunner IS NULL OR next_first_baserunner = 'NA')
      AND (
        initial_inning = next_inning
        OR (
          initial_inning != next_inning
          AND EXISTS (
            SELECT 1 FROM game_events ge
            WHERE ge.game_str = outcome_analysis.game_str
              AND ge.play_id = outcome_analysis.play_id
              AND ge.event_code = 3 -- throw
          )
        )
      )
    GROUP BY game_str, play_per_game, play_id, outcome, runner_id, catcher_id
),
pitcher_pos AS (
    -- get pitcher pos
    SELECT game_str, play_id, timestamp, field_x, field_y
    FROM player_pos
    WHERE player_position = 1 -- pitcher
    AND EXISTS (SELECT 1 FROM steal_outcomes so WHERE so.game_str = player_pos.game_str AND so.play_id = player_pos.play_id)
),
ball_pitcher_distance AS (
    -- calculate distance of ball from pitcher
    SELECT
        bp.game_str, bp.play_id, bp.timestamp,
        SQRT(POWER(bp.ball_position_x - pp.field_x, 2) + POWER(bp.ball_position_y - pp.field_y, 2)) as distance_from_pitcher
    FROM ball_pos bp
    JOIN pitcher_pos pp ON bp.game_str = pp.game_str AND bp.play_id = pp.play_id AND bp.timestamp = pp.timestamp
),
release_time AS (
    --identify the exact timestamp of pitch release
    SELECT game_str, play_id, MIN(timestamp) AS release_timestamp
    FROM ball_pitcher_distance
    WHERE distance_from_pitcher >= 1.0
    GROUP BY game_str, play_id
),
runner_kinematics AS (
  -- gather positional data by looking back at 100ms 200ms and 300ms w2indows
  SELECT
    pp.game_str, pp.play_id, pp.timestamp, pp.field_x, pp.field_y,
    (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_100ms_x,
    (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_100ms_y,
    (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_100ms_timestamp,
    (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_200ms_x,
    (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_200ms_y,
    (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_200ms_timestamp,
    (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_300ms_x,
    (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_300ms_y,
    (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC LIMIT 1) AS prev_300ms_timestamp
  FROM player_pos pp
  WHERE EXISTS (SELECT 1 FROM steal_outcomes so WHERE so.game_str = pp.game_str AND so.play_id = pp.play_id)
  AND pp.player_position = 11 -- runner on first
),
runner_kinematics_calculated AS (
  --calculate lead distancces from 1st
  SELECT *,
    SQRT(POWER(field_x - 45 * SQRT(2), 2) + POWER(field_y - 45 * SQRT(2), 2)) AS lead_distance,
    SQRT(POWER(prev_100ms_x - 45 * SQRT(2), 2) + POWER(prev_100ms_y - 45 * SQRT(2), 2)) AS prev_100ms_lead_distance,
    SQRT(POWER(prev_200ms_x - 45 * SQRT(2), 2) + POWER(prev_200ms_y - 45 * SQRT(2), 2)) AS prev_200ms_lead_distance,
    SQRT(POWER(prev_300ms_x - 45 * SQRT(2), 2) + POWER(prev_300ms_y - 45 * SQRT(2), 2)) AS prev_300ms_lead_distance
  FROM runner_kinematics
),
velocity_calculated AS (
  -- calculate velocities
  SELECT *,
    CASE WHEN prev_100ms_timestamp IS NULL OR timestamp = prev_100ms_timestamp THEN NULL
         ELSE (lead_distance - prev_100ms_lead_distance) / ((timestamp - prev_100ms_timestamp) / 1000.0)
    END AS velocity,
    CASE WHEN prev_200ms_timestamp IS NULL OR prev_100ms_timestamp = prev_200ms_timestamp THEN NULL
         ELSE (prev_100ms_lead_distance - prev_200ms_lead_distance) / ((prev_100ms_timestamp - prev_200ms_timestamp) / 1000.0)
    END AS prev_100ms_velocity,
    CASE WHEN prev_300ms_timestamp IS NULL OR prev_200ms_timestamp = prev_300ms_timestamp THEN NULL
         ELSE (prev_200ms_lead_distance - prev_300ms_lead_distance) / ((prev_200ms_timestamp - prev_300ms_timestamp) / 1000.0)
    END AS prev_200ms_velocity
  FROM runner_kinematics_calculated
),
target_moment AS (
    -- calculate accel based on our velocities
    SELECT
        vc.game_str, vc.play_id, vc.lead_distance, vc.velocity,
        -- accel is rate of change of velocity
        CASE WHEN vc.velocity IS NULL OR vc.prev_100ms_velocity IS NULL OR vc.timestamp = vc.prev_100ms_timestamp THEN NULL
             ELSE (vc.velocity - vc.prev_100ms_velocity) / ((vc.timestamp - vc.prev_100ms_timestamp) / 1000.0)
        END AS acceleration,
        -- calcualtes jerk based on rate of change of accel
        CASE
          WHEN vc.velocity IS NULL OR vc.prev_100ms_velocity IS NULL OR vc.prev_200ms_velocity IS NULL
               OR vc.timestamp = vc.prev_100ms_timestamp OR vc.prev_100ms_timestamp = vc.prev_200ms_timestamp THEN NULL
          ELSE
            -- calculate current and prevous accel
            (
                ((vc.velocity - vc.prev_100ms_velocity) / ((vc.timestamp - vc.prev_100ms_timestamp) / 1000.0)) -
                ((vc.prev_100ms_velocity - vc.prev_200ms_velocity) / ((vc.prev_100ms_timestamp - vc.prev_200ms_timestamp) / 1000.0))
            ) / ((vc.timestamp - vc.prev_100ms_timestamp) / 1000.0) -- Divide by the most recent time delta.
        END AS jerk
    FROM velocity_calculated vc
    JOIN release_time rt ON vc.game_str = rt.game_str AND vc.play_id = rt.play_id
    -- get moment 1.25s before the pitch
    QUALIFY ROW_NUMBER() OVER(PARTITION BY vc.game_str, vc.play_id ORDER BY ABS(vc.timestamp - (rt.release_timestamp - 1250)) ASC, vc.timestamp DESC) = 1
)
-- join outcomes with kinematics
SELECT
  so.game_str,
  so.play_id,
  so.runner_id,
  so.catcher_id,
  so.outcome,
  tm.lead_distance,
  tm.velocity,
  tm.acceleration,
  tm.jerk
FROM steal_outcomes so
LEFT JOIN target_moment tm ON so.game_str = tm.game_str AND so.play_id = tm.play_id
ORDER BY
  so.game_str,
  so.play_id;
"""
#joins sprint speeds
df_steal_attempts = con.sql(query).df()
df_steal_attempts_for_later = con.sql(query).df()
sprint_speeds_to_merge = sprint_speed_df[['runner_id', 'sprint_speed']]
df_steal_attempts = pd.merge(
    df_steal_attempts,
    sprint_speeds_to_merge,
    on='runner_id',
    how='left'
)
#joins poptimes
catcher_poptimes_df = catcher_poptimes_df[['catcher_id', 'average_pop_time']]
df_steal_attempts = pd.merge(
    df_steal_attempts,
    catcher_poptimes_df,
    on='catcher_id',
    how='left'
)
df_steal_attempts = df_steal_attempts[df_steal_attempts['lead_distance'].notna() & ((df_steal_attempts['sprint_speed'].notna()) | (df_steal_attempts['average_pop_time'].notna()))]
print("Steal success rate:")
print(len(df_steal_attempts[df_steal_attempts['outcome']==0])/len(df_steal_attempts))
df_steal_attempts
#%%
#sets up features
columns_to_drop = ['game_str', 'play_id', 'runner_id','catcher_id']
steal_model_data = df_steal_attempts.drop(columns=columns_to_drop).reset_index(drop=True)
features = ['sprint_speed', 'average_pop_time', 'lead_distance','velocity','acceleration']
#makes figures for setals
titles = ['Distribution of Sprint Speed', 'Distribution of Pop Time', 'Distribution of Lead Distance', 'Distribution of Velocity', 'Distribution of Acceleration']
x_labels = ['Sprint Speed (ft/s)', 'Pop time (s)', 'Lead Distance (ft)', 'Velocity (ft/s)', 'Acceleration (ft/s^2)']
'outcome'
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(features):
    data_to_plot = steal_model_data[col].dropna()
    
    axes[i].hist(data_to_plot, bins=10, color='black', edgecolor='black')
    axes[i].set_title(titles[i], fontsize=14)
    axes[i].set_xlabel(x_labels[i], fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()
#%%
#sets up features and what we are predicting
y = steal_model_data['outcome']
X = steal_model_data[features]
n_repeats = 1
n_splits = 5
n_estimators = 1000  #
early_stopping_rounds = 100

# storage for models
lgbm_aucs_for_this_run = []
xgb_aucs_for_this_run = []
rf_aucs_for_this_run = []
lasso_aucs_for_this_run = []
ridge_aucs_for_this_run = []
lr_aucs_for_this_run = []

all_lgbm_best_iterations = []
all_xgb_best_iterations = []
# main cross validation loop
for i in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
    

    # inner loop for processing
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        # data splitting
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        #median impution
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        #sets up the training data
        X_train_df = pd.DataFrame(X_train_imputed, columns=X.columns)
        X_val_df = pd.DataFrame(X_val_imputed, columns=X.columns)
        
        count_negative = np.sum(y_train == 0)
        count_positive = np.sum(y_train == 1)
        scale_pos_weight_value = count_negative / count_positive if count_positive > 0 else 1

        # lgbm
        lgbm_model = lgb.LGBMClassifier(objective='binary', metric='auc', scale_pos_weight=scale_pos_weight_value, n_estimators=n_estimators, random_state=i, verbose=-1)
        lgbm_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)])
        all_lgbm_best_iterations.append(lgbm_model.best_iteration_ if lgbm_model.best_iteration_ is not None else n_estimators)
        y_pred_proba_lgbm = lgbm_model.predict_proba(X_val_df)[:, 1]
        lgbm_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lgbm))

        # xgboost
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=scale_pos_weight_value, n_estimators=n_estimators, random_state=i, early_stopping_rounds=early_stopping_rounds, reg_lambda=0)
        xgb_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], verbose=False)
        all_xgb_best_iterations.append(xgb_model.best_iteration if xgb_model.best_iteration is not None else n_estimators)
        y_pred_proba_xgb = xgb_model.predict_proba(X_val_df)[:, 1]
        xgb_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_xgb))

        # random forest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=i, n_jobs=-1)
        rf_model.fit(X_train_imputed, y_train)
        y_pred_proba_rf = rf_model.predict_proba(X_val_imputed)[:, 1]
        rf_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_rf))

        # lasso regression
        lasso_pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=i))])
        lasso_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_lasso = lasso_pipeline.predict_proba(X_val_imputed)[:, 1]
        lasso_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lasso))
         
        # ridge regression
        ridge_pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', LogisticRegression(penalty='l2', class_weight='balanced', random_state=i))])
        ridge_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_ridge = ridge_pipeline.predict_proba(X_val_imputed)[:, 1]
        ridge_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_ridge))
        # lasso regression
        lr_pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=i))])
        lr_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_lr = lr_pipeline.predict_proba(X_val_imputed)[:, 1]
        lr_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lasso))

# results
print("\n--- Median Steal Model Performance Across All Repeats ---")
print(f"Median LightGBM ROC-AUC: {np.median(lgbm_aucs_for_this_run):.4f}  (standard deviation {np.std(lgbm_aucs_for_this_run):.4f})")
print(f"Median XGBoost ROC-AUC:  {np.median(xgb_aucs_for_this_run):.4f} (standard deviation {np.std(xgb_aucs_for_this_run):.4f})")
print(f"Median Random Forest ROC-AUC: {np.median(rf_aucs_for_this_run):.4f} (standard deviation {np.std(rf_aucs_for_this_run):.4f})")
print(f"Median Lasso Regression ROC-AUC: {np.median(lasso_aucs_for_this_run):.4f} (standard deviation {np.std(lasso_aucs_for_this_run):.4f})")
print(f"Median Ridge Regression ROC-AUC: {np.median(ridge_aucs_for_this_run):.4f} (standard deviation {np.std(ridge_aucs_for_this_run):.4f})")
print("-" * 50)
print(f"Median LGBM Stopping Point: {np.median(all_lgbm_best_iterations):.0f} estimators")
print(f"Median XGBoost Stopping Point: {np.median(all_xgb_best_iterations):.0f} estimators")
#%%
#imputes medians
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(steal_model_data[features])
scale_pos_weight_value = np.sum(y == 0) / np.sum(y == 1) if np.sum(y == 1) > 0 else 1
#%%
#fits for xgb binary
steal_model = xgb.XGBClassifier(
  scale_pos_weight=scale_pos_weight_value,
  objective='binary:logistic',
  n_estimators=int(np.round(np.median(all_xgb_best_iterations))),
  reg_lambda=0
)
steal_model.fit(X, y)

y_pred_proba_final = steal_model.predict_proba(X)
#%%
#creates shap plot
X_percentile[col] = X_percentile[col].rank(pct=True) * 100
formatted_feature_names = [name.replace('_', ' ').title() for name in X_percentile.columns]
X_percentile.columns = formatted_feature_names
shap.summary_plot(shap_values.values, features=X_percentile, show=False, cmap='bwr')
plt.title("Factors Influencing the Likelihood of a Steal", size=16, pad=20)
plt.xlabel("Model Output Impact\n(Left = Lower chance  |  Right = Higher chance of Caught Stealing)")
plt.ylabel("Features (Sorted by Overall Importance)", size=10)
fig = plt.gcf()
colorbar_ax = fig.axes[-1]
colorbar_ax.set_ylabel("Feature Value", size=12)
plt.tight_layout()
plt.show()
#%%
#game count
query = "SELECT COUNT(DISTINCT game_str) AS total_games FROM game_info;"
con.sql(query).df()
#%%
query = """
WITH joined_info AS (
    SELECT
        gi.*,
        ge.event_code,
        ge.play_id
    FROM game_info gi
    JOIN game_events ge
        ON gi.game_str = ge.game_str AND gi.play_per_game = ge.play_id
),
-- deduplicate abs
current_atbat_deduped AS (
    SELECT *
    FROM (
        SELECT
            gi.*,
            ROW_NUMBER() OVER (
                PARTITION BY game_str, top_bottom_inning, at_bat
                -- use tiebreaker with columns
                ORDER BY play_per_game DESC, batter ASC, first_baserunner ASC
            ) AS rn
        FROM joined_info gi
        WHERE event_code = 4
            AND first_baserunner IS NOT NULL AND first_baserunner != 'NA'
            AND (second_baserunner IS NULL OR second_baserunner = 'NA')
            AND (third_baserunner IS NULL OR third_baserunner = 'NA')
    )
    WHERE rn = 1 AND TRY_CAST(at_bat AS INT) IS NOT NULL
),
-deduplicate next at bats
next_atbat_deduped AS (
    SELECT *
    FROM (
        SELECT
            gi.*,
            ROW_NUMBER() OVER (
                PARTITION BY game_str, top_bottom_inning, at_bat
                -- tie breaker with exiting columns
                ORDER BY play_per_game DESC, batter ASC, first_baserunner ASC
            ) AS rn
        FROM joined_info gi
    )
    WHERE rn = 1 AND TRY_CAST(at_bat AS INT) IS NOT NULL
),
-- Join the current and next, both deduped
only_first_base_plays AS (
    SELECT
        gi.game_str,
        gi.play_id,
        gi.top_bottom_inning,
        gi.at_bat,
        gi.play_per_game,
        gi.batter,
        gi.first_baserunner,
        gi2.play_per_game AS next_play,
        gi2.first_baserunner AS next_first,
        gi2.second_baserunner AS next_second,
        gi2.third_baserunner AS next_third
    FROM current_atbat_deduped gi
    INNER JOIN next_atbat_deduped gi2
        ON gi.game_str = gi2.game_str
        AND gi.top_bottom_inning = gi2.top_bottom_inning
        AND CAST(gi.at_bat AS INT) + 1 = CAST(gi2.at_bat AS INT)
),
--classify the results of the balls in play
base_results AS (
    SELECT
        ofbp.game_str,
        ofbp.play_id,
        ofbp.top_bottom_inning,
        ofbp.at_bat,
        ofbp.play_per_game,
        ofbp.batter,
        ofbp.first_baserunner,
        CASE
            WHEN ofbp.batter = ofbp.next_first THEN '1B'
            WHEN ofbp.batter = ofbp.next_second THEN '2B'
            WHEN ofbp.batter = ofbp.next_third THEN '3B'
            ELSE 'Out'
        END AS batter_result,
        CASE
            WHEN ofbp.first_baserunner = ofbp.next_first THEN '1B'
            WHEN ofbp.first_baserunner = ofbp.next_second THEN '2B'
            WHEN ofbp.first_baserunner = ofbp.next_third THEN '3B'
            ELSE 'Inconclusive'
        END AS baserunner_result
    FROM only_first_base_plays ofbp
),
--classify the end base states in 000 form
batter_and_runner_outcomes AS (
    SELECT
        *,
        (
            CASE WHEN batter_result = '1B' OR baserunner_result = '1B' THEN '1' ELSE '0' END ||
            CASE WHEN batter_result = '2B' OR baserunner_result = '2B' THEN '1' ELSE '0' END ||
            CASE WHEN batter_result = '3B' OR baserunner_result = '3B' THEN '1' ELSE '0' END
        ) AS base_state
    FROM base_results
),
-- get final pos of batter at end of play
end_of_play_batter_position AS (
    SELECT
        ge.game_str,
        ge.play_id,
        pp.field_x,
        pp.field_y
    FROM game_events ge
    JOIN player_pos pp
        ON ge.game_str = pp.game_str
        AND ge.play_id = pp.play_id
        AND ge.timestamp = pp.timestamp
    WHERE ge.event_code = 5 AND pp.player_position = 10
    -- tie break timestamp with player pos coords
    QUALIFY ROW_NUMBER() OVER(PARTITION BY ge.game_str, ge.play_id ORDER BY ge.timestamp DESC, pp.player_position ASC, pp.field_x ASC, pp.field_y ASC) = 1
),
-- getws fianl pos of runners on base at end of play
end_of_play_runner_positions AS (
    SELECT
        ge.game_str,
        ge.play_id,
        pp.player_position,
        pp.field_x,
        pp.field_y
    FROM game_events ge
    JOIN player_pos pp
        ON ge.game_str = pp.game_str
        AND ge.play_id = pp.play_id
        AND ge.timestamp = pp.timestamp
    WHERE ge.event_code = 5 AND pp.player_position = 11
    -- tie break timetstamp with pos data
    QUALIFY ROW_NUMBER() OVER(PARTITION BY ge.game_str, ge.play_id, pp.player_position ORDER BY ge.timestamp DESC, pp.field_x ASC, pp.field_y ASC) = 1
),
--join outcomes and end pos
outcomes_with_end_pos AS (
    SELECT
        bro.*,
        batter_pos.field_x AS batter_end_x,
        batter_pos.field_y AS batter_end_y,
        runner_pos.field_x AS runner_end_x,
        runner_pos.field_y AS runner_end_y
    FROM batter_and_runner_outcomes bro
    LEFT JOIN end_of_play_batter_position batter_pos
        ON bro.game_str = batter_pos.game_str AND bro.play_id = batter_pos.play_id
    LEFT JOIN end_of_play_runner_positions runner_pos
        ON bro.game_str = runner_pos.game_str
        AND bro.play_id = runner_pos.play_id
        AND runner_pos.player_position = (
            CASE bro.baserunner_result
                WHEN '1B' THEN 11 WHEN '2B' THEN 12 WHEN '3B' THEN 13
                ELSE -1
            END)
),
--validate these outcomes with the position
validated_outcomes AS (
    SELECT *
    FROM outcomes_with_end_pos
    WHERE
        ( -- batter validation
            batter_result = 'Out' OR
            (batter_end_x IS NOT NULL AND (
                (batter_result = '1B' AND SQRT(POWER(batter_end_x - (45*SQRT(2)), 2) + POWER(batter_end_y - (45*SQRT(2)), 2)) <= 10) OR
                (batter_result = '2B' AND SQRT(POWER(batter_end_x - 0, 2) + POWER(batter_end_y - 120, 2)) <= 10) OR
                (batter_result = '3B' AND SQRT(POWER(batter_end_x - (-45*SQRT(2)), 2) + POWER(batter_end_y - (45*SQRT(2)), 2)) <= 10)
            ))
        )
),
-- kinematic logic
--finds pos of pitcher
pitcher_pos AS (
    SELECT game_str, play_id, timestamp, field_x, field_y
    FROM player_pos
    WHERE player_position = 1
        AND EXISTS (SELECT 1 FROM validated_outcomes bro WHERE bro.game_str = player_pos.game_str AND bro.play_id = player_pos.play_id)
),
--distance of ball from pitcher
ball_pitcher_distance AS (
    SELECT
        bp.game_str, bp.play_id, bp.timestamp,
        SQRT(POWER(bp.ball_position_x - pp.field_x, 2) + POWER(bp.ball_position_y - pp.field_y, 2)) as distance_from_pitcher
    FROM ball_pos bp
    JOIN pitcher_pos pp ON bp.game_str = pp.game_str AND bp.play_id = pp.play_id AND bp.timestamp = pp.timestamp
),
--moment of release
release_time AS (
    SELECT game_str, play_id, MIN(timestamp) AS release_timestamp
    FROM ball_pitcher_distance
    WHERE distance_from_pitcher >= 1.0
    GROUP BY game_str, play_id
),
--finds pos of runner 100, 200 and 300 ms seconds before release
runner_kinematics AS (
    SELECT
        pp.game_str, pp.play_id, pp.timestamp, pp.field_x, pp.field_y,
        (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_100ms_x,
        (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_100ms_y,
        (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 100 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_100ms_timestamp,
        (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_200ms_x,
        (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_200ms_y,
        (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 200 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_200ms_timestamp,
        (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_300ms_x,
        (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_300ms_y,
        (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = pp.game_str AND pp2.play_id = pp.play_id AND pp2.player_position = 11 AND pp2.timestamp <= pp.timestamp - 300 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS prev_300ms_timestamp
    FROM player_pos pp
    WHERE EXISTS (SELECT 1 FROM validated_outcomes bro WHERE bro.game_str = pp.game_str AND bro.play_id = pp.play_id)
        AND pp.player_position = 11
),
--calculates lead in terms of distance from 1st
runner_kinematics_calculated AS (
    SELECT *,
        SQRT(POWER(field_x - 45 * SQRT(2), 2) + POWER(field_y - 45 * SQRT(2), 2)) AS lead_distance,
        SQRT(POWER(prev_100ms_x - 45 * SQRT(2), 2) + POWER(prev_100ms_y - 45 * SQRT(2), 2)) AS prev_100ms_lead_distance,
        SQRT(POWER(prev_200ms_x - 45 * SQRT(2), 2) + POWER(prev_200ms_y - 45 * SQRT(2), 2)) AS prev_200ms_lead_distance,
        SQRT(POWER(prev_300ms_x - 45 * SQRT(2), 2) + POWER(prev_300ms_y - 45 * SQRT(2), 2)) AS prev_300ms_lead_distance
    FROM runner_kinematics
),
--calculate the vellocity based on our position data
velocity_calculated AS (
    SELECT *,
        CASE WHEN prev_100ms_timestamp IS NULL OR timestamp = prev_100ms_timestamp THEN 0
            ELSE (lead_distance - prev_100ms_lead_distance) / ((timestamp - prev_100ms_timestamp) / 1000.0)
        END AS velocity,
        CASE WHEN prev_200ms_timestamp IS NULL OR prev_100ms_timestamp = prev_200ms_timestamp THEN 0
            ELSE (prev_100ms_lead_distance - prev_200ms_lead_distance) / ((prev_100ms_timestamp - prev_200ms_timestamp) / 1000.0)
        END AS prev_100ms_velocity,
        CASE WHEN prev_300ms_timestamp IS NULL OR prev_200ms_timestamp = prev_300ms_timestamp THEN 0
            ELSE (prev_200ms_lead_distance - prev_300ms_lead_distance) / ((prev_200ms_timestamp - prev_300ms_timestamp) / 1000.0)
        END AS prev_200ms_velocity
    FROM runner_kinematics_calculated
),
--finds the accel and jerk based on our velocities
target_moment AS (
    SELECT
        vc.game_str, vc.play_id, vc.lead_distance, vc.velocity,
        CASE WHEN vc.velocity IS NULL OR vc.prev_100ms_velocity IS NULL OR vc.timestamp = vc.prev_100ms_timestamp THEN NULL
            ELSE (vc.velocity - vc.prev_100ms_velocity) / ((vc.timestamp - vc.prev_100ms_timestamp) / 1000.0)
        END AS acceleration,
        CASE
            WHEN vc.velocity IS NULL OR vc.prev_100ms_velocity IS NULL OR vc.prev_200ms_velocity IS NULL
            OR vc.timestamp = vc.prev_100ms_timestamp OR vc.prev_100ms_timestamp = vc.prev_200ms_timestamp THEN NULL
            ELSE
            (
                ((vc.velocity - vc.prev_100ms_velocity) / ((vc.timestamp - vc.prev_100ms_timestamp) / 1000.0)) -
                ((vc.prev_100ms_velocity - vc.prev_200ms_velocity) / ((vc.prev_100ms_timestamp - vc.prev_200ms_timestamp) / 1000.0))
            ) / ((vc.timestamp - vc.prev_100ms_timestamp) / 1000.0)
        END AS jerk
    FROM velocity_calculated vc
    JOIN release_time rt ON vc.game_str = rt.game_str AND vc.play_id = rt.play_id
    --tie breaking using coordinates
    QUALIFY ROW_NUMBER() OVER(PARTITION BY vc.game_str, vc.play_id ORDER BY ABS(vc.timestamp - (rt.release_timestamp - 1250)) ASC, vc.timestamp ASC, vc.field_x ASC, vc.field_y ASC) = 1
)
-- join outcomes and kinematics
SELECT
    bro.*,
    tm.lead_distance,
    tm.velocity,
    tm.acceleration,
    tm.jerk
FROM validated_outcomes bro
LEFT JOIN target_moment tm ON bro.game_str = tm.game_str AND bro.play_id = tm.play_id
ORDER BY bro.game_str, bro.play_id, bro.top_bottom_inning, bro.at_bat;
"""
#drop duplicates
df_batter_outcomes = con.sql(query).df()
df_batter_outcomes = df_batter_outcomes.drop_duplicates(subset=['game_str', 'play_id','at_bat','play_per_game'], keep='first')
df_batter_outcomes = df_batter_outcomes[(df_batter_outcomes['base_state'] != '111')]
df_batter_outcomes.reset_index(drop=True)
#%%
#join sprint speeds
df_batter_outcomes['base_state'] = df_batter_outcomes['base_state'].astype(str).str.zfill(3)
df_batter_outcomes = pd.merge(
    df_batter_outcomes,
    sprint_speeds_to_merge,
    right_on='runner_id', left_on = 'first_baserunner',
    how='left'
)
columns_to_drop = []
df_batter_outcomes = df_batter_outcomes[df_batter_outcomes['lead_distance'].notna()].reset_index(drop=True)
#%%
#make df for outs, singles, and doubles
df_outs = df_batter_outcomes[df_batter_outcomes['batter_result']== "Out"].drop(columns=columns_to_drop).reset_index(drop=True)
df_singles = df_batter_outcomes[df_batter_outcomes['batter_result']== "1B"].drop(columns=columns_to_drop).reset_index(drop=True)
df_doubles = df_batter_outcomes[df_batter_outcomes['batter_result']== "2B"].drop(columns=columns_to_drop).reset_index(drop=True)
#%%
#filter outs so we are only looking at our ideal training set
df_outs = df_outs[(df_outs['base_state'] != '110') & (df_outs['base_state'] != '110') & (df_outs['base_state'] != '101') & (df_outs['base_state'] != '011')]
#%%
#manually filtered doubles types
doubles_011 = [3,4,6,8,9,10,11,14,15,16,19,20,21,22,23,24,26,28,31,32,36,39,40,41,45,46,47,48,54,56,57,59,62,63,66,67,68,70,75,77,80,81,83,88,92,93,95,97,100,106,107]
doubles_010out = [2,34,38,44,60,65,84,109]
doubles_010score = [1,5,12,17,18,25,27,29,30,33,35,37,42,43,49,50,51,52,55,58,71,73,74,76,79,85,87,89,90,91,94,96,98,99,101,102,103,104,108]
df_doubles['base_state'] = np.nan
#abbreviates for these types
df_doubles.loc[(df_doubles.index+1).isin(doubles_011), 'base_state'] = '011'
df_doubles.loc[(df_doubles.index+1).isin(doubles_010out), 'base_state'] = '010(0)'
df_doubles.loc[(df_doubles.index+1).isin(doubles_010score), 'base_state'] = '010(1)'
df_doubles.dropna(subset=['base_state'], inplace=True)
#%%
df_doubles
#%%
#sets up models, model storage, and cross validation
label_encoder_outs = LabelEncoder()
df_outs['base_state_encoded'] = label_encoder_outs.fit_transform(df_outs['base_state'])
features = ['lead_distance',	'velocity',	'acceleration',	'sprint_speed']
X = df_outs[features]
y = df_outs['base_state_encoded']
outs_classes= label_encoder_outs.classes_
all_labels = sorted(y.unique())
num_classes = len(all_labels)
n_repeats = 1
n_splits = 5
n_estimators = 1000
early_stopping_rounds = 100
all_mean_lgbm_aucs = []
all_mean_xgb_aucs = []
all_mean_rf_aucs = []
all_mean_lasso_aucs = []
all_mean_ridge_aucs = []

all_lgbm_best_iterations = []
all_xgb_best_iterations = []

# main cross validation loop
for i in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
    #storage for aucs
    lgbm_aucs_for_this_run = []
    xgb_aucs_for_this_run = []
    rf_aucs_for_this_run = []
    lasso_aucs_for_this_run = []
    ridge_aucs_for_this_run = []

    print(f"--- Starting Repeat {i+1}/{n_repeats} ---")
    #inner fold
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        #imputes medians
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)

        # converts imputed arrays back into dfs
        X_train_df = pd.DataFrame(X_train_imputed, columns=X.columns)
        X_val_df = pd.DataFrame(X_val_imputed, columns=X.columns)

        # lgbm
        lgbm_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, metric='multi_logloss', class_weight='balanced', n_estimators=n_estimators, random_state=i, verbose=-1)
        lgbm_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)])
        y_pred_proba_lgbm = lgbm_model.predict_proba(X_val_df)
        lgbm_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lgbm, multi_class='ovr', average='weighted', labels=all_labels))
        all_lgbm_best_iterations.append(lgbm_model.best_iteration_ if lgbm_model.best_iteration_ is not None else n_estimators)

        # xgboost
        xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, eval_metric='mlogloss', n_estimators=n_estimators, random_state=i, early_stopping_rounds=early_stopping_rounds)
        xgb_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], verbose=False)
        y_pred_proba_xgb = xgb_model.predict_proba(X_val_df)
        xgb_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_xgb, multi_class='ovr', average='weighted', labels=all_labels))
        all_xgb_best_iterations.append(xgb_model.best_iteration)

        # random forest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=i, n_jobs=-1)
        rf_model.fit(X_train_imputed, y_train)
        y_pred_proba_rf = rf_model.predict_proba(X_val_imputed)
        rf_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_rf, multi_class='ovr', average='weighted', labels=all_labels))

        # lasso regression
        lasso_pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=i))])
        lasso_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_lasso = lasso_pipeline.predict_proba(X_val_imputed)
        lasso_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lasso, multi_class='ovr', average='weighted', labels=all_labels))

        # logistic regression
        ridge_pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', LogisticRegression(penalty='l2', class_weight='balanced', random_state=i))])
        ridge_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_ridge = ridge_pipeline.predict_proba(X_val_imputed)
        ridge_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_ridge, multi_class='ovr', average='weighted', labels=all_labels))
    #append aucs
    all_mean_lgbm_aucs.append(lgbm_aucs_for_this_run)
    all_mean_xgb_aucs.append(xgb_aucs_for_this_run)
    all_mean_rf_aucs.append(rf_aucs_for_this_run)
    all_mean_lasso_aucs.append(lasso_aucs_for_this_run)
    all_mean_ridge_aucs.append(ridge_aucs_for_this_run)

# results for outs
print("\n" + "="*55)
print("--- Outs Model Performance (Weighted ROC-AUC) ---")
print(f"Median LightGBM ROC-AUC:            {np.median(all_mean_lgbm_aucs):.4f}")
print(f"Median XGBoost ROC-AUC:             {np.median(all_mean_xgb_aucs):.4f}")
print(f"Median Random Forest ROC-AUC:       {np.median(all_mean_rf_aucs):.4f}")
print(f"Median Lasso (L1) LogReg ROC-AUC:   {np.median(all_mean_lasso_aucs):.4f}")
print(f"Median Ridge (L2) LogReg ROC-AUC:   {np.median(all_mean_ridge_aucs):.4f}")
print("-" * 55)
print(f"Median LGBM Stopping Point:         {np.median(all_lgbm_best_iterations):.0f} estimators")
print(f"Median XGBoost Stopping Point:      {np.median(all_xgb_best_iterations):.0f} estimators")
print("="*55)
#impute medians and fit imputeds
outs_model = Pipeline([('scaler', StandardScaler()), ('lasso', LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=i))])
imputer = SimpleImputer(strategy='median')
X_imputed =  imputer.fit_transform(X)
outs_model.fit(X_imputed, y)
#%%
logistic_model = outs_model.named_steps['lasso']
coefficients = logistic_model.coef_
avg_importance = np.mean(np.abs(coefficients), axis=0)
total_importance = np.sum(avg_importance)
percentage_importance = (avg_importance / total_importance) * 100
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': percentage_importance
}).sort_values(by='Importance', ascending=False)
feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('_', ' ').str.title()
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(False)
bars = ax.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
ax.bar_label(bars, fmt='%.2f%%')
ax.set_title('Outs Model Feature Importance', fontsize=16, pad=20)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance (%)', fontsize=12)
ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

#%%
#sets up singles models, storage, and cross validation
label_encoder_singles = LabelEncoder()
df_singles['base_state_encoded'] = label_encoder_singles.fit_transform(df_singles['base_state'])
features = ['lead_distance',	'velocity',	'acceleration',	'sprint_speed']
X = df_singles[features]
y = df_singles['base_state_encoded']
singles_classes= label_encoder_singles.classes_
all_labels = sorted(y.unique())
num_classes = len(all_labels)
n_repeats = 1
n_splits = 5
n_estimators = 1000
early_stopping_rounds = 100
all_mean_lgbm_aucs = []
all_mean_xgb_aucs = []
all_mean_rf_aucs = []
all_mean_lasso_aucs = []
all_mean_ridge_aucs = []

all_lgbm_best_iterations = []
all_xgb_best_iterations = []

# main cross validation loop
for i in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
    #auc storage
    lgbm_aucs_for_this_run = []
    xgb_aucs_for_this_run = []
    rf_aucs_for_this_run = []
    lasso_aucs_for_this_run = []
    ridge_aucs_for_this_run = []

    print(f"--- Starting Repeat {i+1}/{n_repeats} ---")
    #inner loop
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        #impute medians
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)

        # convert the imputed arrays back into dfs
        X_train_df = pd.DataFrame(X_train_imputed, columns=X.columns)
        X_val_df = pd.DataFrame(X_val_imputed, columns=X.columns)

        # lgbm
        lgbm_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, metric='multi_logloss', class_weight='balanced', n_estimators=n_estimators, random_state=i, verbose=-1)
        lgbm_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)])
        y_pred_proba_lgbm = lgbm_model.predict_proba(X_val_df)
        lgbm_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lgbm, multi_class='ovr', average='weighted', labels=all_labels))
        all_lgbm_best_iterations.append(lgbm_model.best_iteration_ if lgbm_model.best_iteration_ is not None else n_estimators)

        # xgboost
        xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, eval_metric='mlogloss', n_estimators=n_estimators, random_state=i, early_stopping_rounds=early_stopping_rounds)
        xgb_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], verbose=False)
        y_pred_proba_xgb = xgb_model.predict_proba(X_val_df)
        xgb_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_xgb, multi_class='ovr', average='weighted', labels=all_labels))
        all_xgb_best_iterations.append(xgb_model.best_iteration)

        # random forest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=i, n_jobs=-1)
        rf_model.fit(X_train_imputed, y_train)
        y_pred_proba_rf = rf_model.predict_proba(X_val_imputed)
        rf_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_rf, multi_class='ovr', average='weighted', labels=all_labels))

        # lasso regression
        lasso_pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=i))])
        lasso_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_lasso = lasso_pipeline.predict_proba(X_val_imputed)
        lasso_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lasso, multi_class='ovr', average='weighted', labels=all_labels))

        # ridge regression
        ridge_pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', LogisticRegression(penalty='l2', class_weight='balanced', random_state=i))])
        ridge_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_ridge = ridge_pipeline.predict_proba(X_val_imputed)
        ridge_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_ridge, multi_class='ovr', average='weighted', labels=all_labels))
    #append aucs
    all_mean_lgbm_aucs.append(lgbm_aucs_for_this_run)
    all_mean_xgb_aucs.append(xgb_aucs_for_this_run)
    all_mean_rf_aucs.append(rf_aucs_for_this_run)
    all_mean_lasso_aucs.append(lasso_aucs_for_this_run)
    all_mean_ridge_aucs.append(ridge_aucs_for_this_run)

# results
print("\n" + "="*55)
print("--- Singles Model Performance (Weighted ROC-AUC) ---")
print(f"Median LightGBM ROC-AUC:            {np.median(all_mean_lgbm_aucs):.4f}")
print(f"Median XGBoost ROC-AUC:             {np.median(all_mean_xgb_aucs):.4f}")
print(f"Median Random Forest ROC-AUC:       {np.median(all_mean_rf_aucs):.4f}")
print(f"Median Lasso (L1) LogReg ROC-AUC:   {np.median(all_mean_lasso_aucs):.4f}")
print(f"Median Ridge (L2) LogReg ROC-AUC:   {np.median(all_mean_ridge_aucs):.4f}")
print("-" * 55)
print(f"Median LGBM Stopping Point:         {np.median(all_lgbm_best_iterations):.0f} estimators")
print(f"Median XGBoost Stopping Point:      {np.median(all_xgb_best_iterations):.0f} estimators")
print("="*55)
#impute and fit medians
singles_model = Pipeline([('scaler', StandardScaler()), ('ridge', LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', random_state=i))])
imputer = SimpleImputer(strategy='median')
X_imputed =  imputer.fit_transform(X)
singles_model.fit(X_imputed, y)
#%%
logistic_model = singles_model.named_steps['ridge']
coefficients = logistic_model.coef_
avg_importance = np.mean(np.abs(coefficients), axis=0)
total_importance = np.sum(avg_importance)
percentage_importance = (avg_importance / total_importance) * 100
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': percentage_importance
}).sort_values(by='Importance', ascending=False)
feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('_', ' ').str.title()
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(False)
bars = ax.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
ax.bar_label(bars, fmt='%.2f%%')
ax.set_title('Singles Model Feature Importance', fontsize=16, pad=20)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance (%)', fontsize=12)
ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
#%%
#sets up doubles model, storage, and cross validation
label_encoder_doubles = LabelEncoder()
df_doubles['base_state_encoded'] = label_encoder_doubles.fit_transform(df_doubles['base_state'])
features = ['lead_distance',	'velocity',	'acceleration',	'sprint_speed']
X = df_doubles[features]
y = df_doubles['base_state_encoded']
all_labels = sorted(y.unique())
num_classes = len(all_labels)
doubles_classes= label_encoder_doubles.classes_
n_repeats = 1
n_splits = 5
n_estimators = 1000
early_stopping_rounds = 100
all_mean_lgbm_aucs = []
all_mean_xgb_aucs = []
all_mean_rf_aucs = []
all_mean_lasso_aucs = []
all_mean_ridge_aucs = []

all_lgbm_best_iterations = []
all_xgb_best_iterations = []

# main cross validation
for i in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
    
    lgbm_aucs_for_this_run = []
    xgb_aucs_for_this_run = []
    rf_aucs_for_this_run = []
    lasso_aucs_for_this_run = []
    ridge_aucs_for_this_run = []

    print(f"--- Starting Repeat {i+1}/{n_repeats} ---")
    #inner loop
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        #impute medians
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)

        # convert out imputed arrays back to dfs
        X_train_df = pd.DataFrame(X_train_imputed, columns=X.columns)
        X_val_df = pd.DataFrame(X_val_imputed, columns=X.columns)

        # lgbm
        lgbm_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, metric='multi_logloss', class_weight='balanced', n_estimators=n_estimators, random_state=i, verbose=-1)
        lgbm_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)])
        y_pred_proba_lgbm = lgbm_model.predict_proba(X_val_df)
        lgbm_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lgbm, multi_class='ovr', average='weighted', labels=all_labels))
        all_lgbm_best_iterations.append(lgbm_model.best_iteration_ if lgbm_model.best_iteration_ is not None else n_estimators)

        # xgb
        xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, eval_metric='mlogloss', n_estimators=n_estimators, random_state=i, early_stopping_rounds=early_stopping_rounds)
        xgb_model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], verbose=False)
        y_pred_proba_xgb = xgb_model.predict_proba(X_val_df)
        xgb_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_xgb, multi_class='ovr', average='weighted', labels=all_labels))
        all_xgb_best_iterations.append(xgb_model.best_iteration)

        # random forest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=i, n_jobs=-1)
        rf_model.fit(X_train_imputed, y_train)
        y_pred_proba_rf = rf_model.predict_proba(X_val_imputed)
        rf_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_rf, multi_class='ovr', average='weighted', labels=all_labels))

        # lasso regression
        lasso_pipeline = Pipeline([('scaler', StandardScaler()), ('lasso', LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=i))])
        lasso_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_lasso = lasso_pipeline.predict_proba(X_val_imputed)
        lasso_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_lasso, multi_class='ovr', average='weighted', labels=all_labels))

        # ridge regression
        ridge_pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', LogisticRegression(penalty='l2', class_weight='balanced', random_state=i))])
        ridge_pipeline.fit(X_train_imputed, y_train)
        y_pred_proba_ridge = ridge_pipeline.predict_proba(X_val_imputed)
        ridge_aucs_for_this_run.append(roc_auc_score(y_val, y_pred_proba_ridge, multi_class='ovr', average='weighted', labels=all_labels))

    all_mean_lgbm_aucs.append(lgbm_aucs_for_this_run)
    all_mean_xgb_aucs.append(xgb_aucs_for_this_run)
    all_mean_rf_aucs.append(rf_aucs_for_this_run)
    all_mean_lasso_aucs.append(lasso_aucs_for_this_run)
    all_mean_ridge_aucs.append(ridge_aucs_for_this_run)

#final results
print("\n" + "="*55)
print("--- Doubles Model Performance (Weighted ROC-AUC) ---")
print(f"Median LightGBM ROC-AUC:            {np.median(all_mean_lgbm_aucs):.4f}")
print(f"Median XGBoost ROC-AUC:             {np.median(all_mean_xgb_aucs):.4f}")
print(f"Median Random Forest ROC-AUC:       {np.median(all_mean_rf_aucs):.4f}")
print(f"Median Lasso (L1) LogReg ROC-AUC:   {np.median(all_mean_lasso_aucs):.4f}")
print(f"Median Ridge (L2) LogReg ROC-AUC:   {np.median(all_mean_ridge_aucs):.4f}")
print("-" * 55)
print(f"Median LGBM Stopping Point:         {np.median(all_lgbm_best_iterations):.0f} estimators")
print(f"Median XGBoost Stopping Point:      {np.median(all_xgb_best_iterations):.0f} estimators")
print("="*55)
#impute nad fit medians
imputer = SimpleImputer(strategy='median')
X_imputed =  imputer.fit_transform(X)
doubles_model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=i, n_jobs=-1)
doubles_model.fit(X_imputed, y)
#%%
importances = doubles_model.feature_importances_
percentage_importance = importances * 100
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': percentage_importance
}).sort_values(by='Importance', ascending=False)
feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('_', ' ').str.title()
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(False)
bars = ax.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
ax.bar_label(bars, fmt='%.2f%%')
ax.set_title('Doubles Model Feature Importance', fontsize=16, pad=20)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance (%)', fontsize=12)
ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
#%%
#Animation code adapted from code by David Awosoga
def _plot_single_play_animation(player_position_df: pd.DataFrame,
                               ball_position_df: pd.DataFrame,
                               game_str: str,
                               play_id: str,
                               filepath: str) -> None:
    """
    generates and saves an animation for a single baseball play
    """
    if not isinstance(play_id, str):
        raise ValueError("Play ID must be a String. This function only handles one Play ID.")

    if len(player_position_df['game_str'].unique()) > 1 or len(ball_position_df['game_str'].unique()) > 1:
        raise ValueError("Player or Ball Position DataFrame has multiple games. Please filter for one game.")
    
    player_pos = player_position_df.query(f'play_id == "{play_id}"')
    ball_pos = ball_position_df.query(f'play_id == "{play_id}"')

    if player_pos.empty:
        print(f"Warning: No player data found for play_id '{play_id}'. Cannot generate animation.")
        return None

    merged_df = pd.merge(player_pos, ball_pos, on=['timestamp', 'play_id', 'game_str'], how='left')
    merged_df['player_position'] = pd.to_numeric(merged_df['player_position'], errors='coerce')
    merged_df = merged_df[merged_df['player_position'] < 14]

    field = MiLBField()
    field.draw(display_range='full')

    fig = plt.gcf()
    ax = plt.gca()

    p = field.scatter([], [], c='white', s=100) # player markers
    b = field.scatter([], [], c='red', s=50)   # ball marker

    game_id = merged_df['game_str'].unique()[0]
    ax.text(0, 400, f'Game ID: {game_id}', c='white', ha='center', fontsize=12)
    ax.text(120, 0, f'Play: {play_id}', c='white', ha='center', fontsize=12)

    # animation update func
    def update(frame):
        frame_data = merged_df[merged_df['timestamp'] <= frame]

        players = frame_data.sort_values('timestamp').drop_duplicates(subset=['player_position'], keep='last')

        balls = frame_data[['ball_position_x', 'ball_position_y', 'ball_position_z']].dropna().iloc[-1:]

        if not players.empty:
            players_colors = ['yellow' if 10 <= pos <= 13 else 'white' for pos in players['player_position']]
            p.set_offsets(np.c_[players['field_x'], players['field_y']])
            p.set_color(players_colors)

        if not balls.empty:
            z_pos = balls['ball_position_z'].values[0]
            ball_size = max(10, z_pos * 8) # Use max() for a clean minimum size check

            b.set_sizes([ball_size])
            b.set_offsets(np.c_[balls['ball_position_x'], balls['ball_position_y']])
        else:
            b.set_sizes([0])

        return p, b

    # create and save the animation
    ani = FuncAnimation(fig, update,
                        frames=np.linspace(merged_df['timestamp'].min(), merged_df['timestamp'].max(), num=100),
                        blit=False)

    ani.save(filepath, writer='imagemagick', fps=25)
    plt.close(fig)

    return None
#%%
def animate_plays(plays_to_animate: pd.DataFrame, con, output_dir: str):
    """
    Generates and saves animations for a given list of baseball plays.

    Args:
        plays_to_animate (pd.DataFrame): DataFrame with plays to animate. 
                                         Must contain 'game_str' and 'play_id' columns.
        con: A database connection object with a .sql() method (e.g., DuckDB).
        output_dir (str): The directory where the output GIF files will be saved.
    """
    required_columns = ['game_str', 'play_id']
    if not all(col in plays_to_animate.columns for col in required_columns):
        raise ValueError(f"The input DataFrame must contain the columns: {required_columns}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")


    for index, row in plays_to_animate.iterrows():
        game_str = row['game_str']
        play_id = str(row['play_id']
        
        print(f"\nProcessing: Game '{game_str}', Play ID '{play_id}'...")

        try:
            player_position_df = con.sql(f"SELECT * FROM player_pos WHERE game_str = '{game_str}' AND play_id = '{play_id}'").df()
            ball_position_df = con.sql(f"SELECT * FROM ball_pos WHERE game_str = '{game_str}' AND play_id = '{play_id}'").df()

            if player_position_df.empty or ball_position_df.empty:
                print(f"  - No data found in database for this play. Skipping.")
                continue
            
            output_filepath = os.path.join(output_dir, f"{game_str}-{play_id}.gif")

            _plot_single_play_animation(
                player_position_df=player_position_df,
                ball_position_df=ball_position_df,
                game_str=game_str,
                play_id=play_id,
                filepath=output_filepath
            )

        except Exception as e:
            print(f"  - ERROR processing {game_str}, Play ID: {play_id}. Reason: {e}")

    print("\nAll plays processed.")
# %%

#%%
# %%
df_doubles
# %%
doubles_classes

# %%
query = """--find plays with a runner on first and other bases empty
WITH PlaysWithRunnerOnFirst AS (
    SELECT
        game_str,
        play_per_game
    FROM game_info
    WHERE
        first_baserunner IS NOT NULL AND first_baserunner != 'NA'
        AND (second_baserunner IS NULL OR second_baserunner = 'NA')
        AND (third_baserunner IS NULL OR third_baserunner = 'NA')
    -- only analyze each play once
    QUALIFY ROW_NUMBER() OVER(PARTITION BY game_str, play_per_game ORDER BY at_bat ASC) = 1
),

-- determine the play type
PlayTypes AS (
    SELECT
        ge.game_str,
        ge.play_id,
        MIN(CASE WHEN ge.event_code = 1 THEN 'pitch' WHEN ge.event_code = 6 THEN 'pickoff' END) as play_type
    FROM game_events ge
    --join to our filtered plays
    WHERE EXISTS (SELECT 1 FROM PlaysWithRunnerOnFirst p WHERE p.game_str = ge.game_str AND p.play_per_game = ge.play_id)
      AND ge.event_code IN (1, 6) -- 1 is pitch, 6 is pickoff throw [cite: 89]
    GROUP BY ge.game_str, ge.play_id
),

-- find the release times by seeing when the ball leaves the pitcher
ReleaseTimes AS (
    SELECT
        bp.game_str,
        bp.play_id,
        MIN(bp.timestamp) AS release_timestamp
    FROM ball_pos bp
    JOIN player_pos pp
        ON bp.game_str = pp.game_str
        AND bp.play_id = pp.play_id
        AND bp.timestamp = pp.timestamp
    WHERE pp.player_position = 1
      AND SQRT(POWER(bp.ball_position_x - pp.field_x, 2) + POWER(bp.ball_position_y - pp.field_y, 2)) >= 1.0
      -- filter for relevant plays
      AND EXISTS (SELECT 1 FROM PlaysWithRunnerOnFirst p WHERE p.game_str = bp.game_str AND p.play_per_game = bp.play_id)
    GROUP BY bp.game_str, bp.play_id
),

-- find the specific target timestamp
TargetMoments AS (
    SELECT
        rt.game_str,
        rt.play_id,
        CASE
            WHEN pt.play_type = 'pitch'   THEN rt.release_timestamp - 1250
            WHEN pt.play_type = 'pickoff' THEN rt.release_timestamp - 725
        END AS target_timestamp
    FROM ReleaseTimes rt
    JOIN PlayTypes pt ON rt.game_str = pt.game_str AND rt.play_id = pt.play_id
),

-- find the closest player position record to our timestamp
RunnerPositionAtTarget AS (
    SELECT
        tm.game_str,
        tm.play_id,
        pp.timestamp,
        pp.field_x,
        pp.field_y
    FROM player_pos pp
    JOIN TargetMoments tm
      ON pp.game_str = tm.game_str AND pp.play_id = tm.play_id
    WHERE pp.player_position = 11 AND tm.target_timestamp IS NOT NULL
    QUALIFY ROW_NUMBER() OVER(PARTITION BY pp.game_str, pp.play_id ORDER BY ABS(pp.timestamp - tm.target_timestamp) ASC, pp.timestamp ASC) = 1
),

-- gather the position, velo, accel, and jerk based on kinematic data
RunnerKinematics AS (
    SELECT
        rpat.game_str, rpat.play_id,
        rpat.timestamp as timestamp_0, rpat.field_x as field_x_0, rpat.field_y as field_y_0,
        (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 100 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS timestamp_1,
        (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 100 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS field_x_1,
        (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 100 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS field_y_1,
        (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 200 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS timestamp_2,
        (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 200 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS field_x_2,
        (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 200 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS field_y_2,
        (SELECT pp2.timestamp FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 300 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS timestamp_3,
        (SELECT pp2.field_x FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 300 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS field_x_3,
        (SELECT pp2.field_y FROM player_pos pp2 WHERE pp2.game_str = rpat.game_str AND pp2.play_id = rpat.play_id AND pp2.player_position = 11 AND pp2.timestamp <= rpat.timestamp - 300 ORDER BY pp2.timestamp DESC, pp2.field_x DESC, pp2.field_y DESC LIMIT 1) AS field_y_3
    FROM RunnerPositionAtTarget rpat
),

-- finds the kineamtic data
Final_Calculations AS (
    SELECT
        game_str, play_id,
        SQRT(POWER(field_x_0 - (45*SQRT(2)), 2) + POWER(field_y_0 - (45*SQRT(2)), 2)) as lead_distance_0,
        SQRT(POWER(field_x_1 - (45*SQRT(2)), 2) + POWER(field_y_1 - (45*SQRT(2)), 2)) as lead_distance_1,
        SQRT(POWER(field_x_2 - (45*SQRT(2)), 2) + POWER(field_y_2 - (45*SQRT(2)), 2)) as lead_distance_2,
        SQRT(POWER(field_x_3 - (45*SQRT(2)), 2) + POWER(field_y_3 - (45*SQRT(2)), 2)) as lead_distance_3,
        (lead_distance_0 - lead_distance_1) / NULLIF((timestamp_0 - timestamp_1), 0) * 1000.0 as velocity_0,
        (lead_distance_1 - lead_distance_2) / NULLIF((timestamp_1 - timestamp_2), 0) * 1000.0 as velocity_1,
        (lead_distance_2 - lead_distance_3) / NULLIF((timestamp_2 - timestamp_3), 0) * 1000.0 as velocity_2,
        (velocity_0 - velocity_1) / NULLIF(((timestamp_0 + timestamp_1)/2.0 - (timestamp_1 + timestamp_2)/2.0), 0) * 1000.0 as acceleration_0,
        (velocity_1 - velocity_2) / NULLIF(((timestamp_1 + timestamp_2)/2.0 - (timestamp_2 + timestamp_3)/2.0), 0) * 1000.0 as acceleration_1,
        timestamp_0, timestamp_1, timestamp_2
    FROM RunnerKinematics
    WHERE timestamp_0 IS NOT NULL AND timestamp_1 IS NOT NULL AND timestamp_2 IS NOT NULL AND timestamp_3 IS NOT NULL
)

-- present the results and filter into safe and aggersive leads
SELECT
    game_str,
    play_id,
    lead_distance_0 AS lead_distance,
    velocity_0 AS velocity,
    acceleration_0 AS acceleration,
    (acceleration_0 - acceleration_1) / NULLIF(((timestamp_0 + timestamp_1)/2.0 - (timestamp_1 + timestamp_2)/2.0), 0) * 1000.0 AS jerk
FROM Final_Calculations
ORDER BY game_str, play_id;"""
#filter leads into safe or agg
df_leads= con.sql(query).df()
kinematic_cols = ['lead_distance', 'velocity', 'acceleration','jerk']
q1_values = df_leads[kinematic_cols].quantile(0.25)
q3_values = df_leads[kinematic_cols].quantile(0.75)
std_devs = df_leads[kinematic_cols].std()
agg_lead = (q3_values).to_frame().T
safe_lead = (q1_values).to_frame().T
# %%
#calculate payoffs based on real run expectancy changes from a run expectancy matrix
def calculate_payoffs(kinematic_data, sprint_speed = sprint_speed_df['sprint_speed'].median(), pop_time = catcher_poptimes_df['average_pop_time'].median()):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #cols for the various outcomes
        pickoff_cols = ['lead_distance', 'velocity', 'acceleration', 'jerk', 'sprint_speed']
        steal_cols = ['sprint_speed', 'average_pop_time', 'lead_distance', 'velocity', 'acceleration']
        bip_cols = ['lead_distance', 'velocity', 'acceleration', 'sprint_speed']

        lead_data = kinematic_data.copy()

        lead_data['sprint_speed'] = sprint_speed
        lead_data['average_pop_time'] = pop_time
        #gets base state probabilties for balls in pay and pickoffs/steals
        pickoff_prob = pickoff_model.predict_proba(lead_data[pickoff_cols])[:, 1][0]
        steal_prob = steal_model.predict_proba(lead_data[steal_cols])[:, 0][0]
  
        out_probs = outs_model.predict_proba(lead_data[bip_cols])[0]
        single_probs = singles_model.predict_proba(lead_data[bip_cols])[0]
        double_probs = doubles_model.predict_proba(lead_data[bip_cols])[0]
        #pickoff payoffs
        pickoff_payoff = 0.0016296 - pickoff_prob * 0.596
        #out payoff
        bip_out_payoff = (out_probs[0] * (0.0961 - 0.836) +
                       out_probs[3] * (0.307 - 0.836) +
                       out_probs[2] * (0.536 - 0.836) +
                       out_probs[1] * (0.829 - 0.836))
        #single payoff
        bip_single_payoff = (single_probs[0] * (0.407 - 0.836) +
                       single_probs[2] * (1.33 - 0.836) +
                       single_probs[1] * (1.64 - 0.836))
        #double payoff
        bip_double_payoff = (double_probs[0] * (0.536 - 0.836) +
                       double_probs[1] * (2.03 - 0.836) +
                       double_probs[2] * (1.83 - 0.836))
        #pitch stay payoff
        pitch_stay_payoff = (
            0.1125 *  bip_out_payoff+
            0.0354 * bip_single_payoff +
            0.0166 * bip_double_payoff

        )+0.021329803603828152
        #pitch and steal payoff
        pitch_steal_payoff = (
            0.6486 * (steal_prob * (1.03 - 0.836) +
                      (1 - steal_prob) * (0.24 - 0.836)) +
            pitch_stay_payoff
        )
          #organizes these payoffs
        payoffs = {
            'pickoff_prob': round(pickoff_prob,5),
            'steal_prob' : round(steal_prob, 5),
            'pickoff_payoff': round(pickoff_payoff,5),
            'pitch_stay_payoff': round(pitch_stay_payoff,5),
            'pitch_steal_payoff': round(pitch_steal_payoff,5),
            'bip_out_payoff': round(bip_out_payoff,5),
            'bip_single_payoff': round(bip_single_payoff,5),
            'bip_double_payoff': round(bip_double_payoff,5)
        }

        return payoffs
#%%
#print the payoffs
def print_payoffs(lead):
  print("Pickoff payoff:")
  print(calculate_payoffs(lead)['pickoff_payoff'])
  print("Pitch stay payoff:")
  print(calculate_payoffs(lead)['pitch_stay_payoff'])
  print("Pitch steal payoff:")
  print(calculate_payoffs(lead)['pitch_steal_payoff'])
# %%
## 4 branch arbitrary example printed out
very_safe_lead = df_steal_attempts[kinematic_cols].quantile(0.25).to_frame().T
safe_lead = df_steal_attempts[kinematic_cols].quantile(0.27).to_frame().T
agg_lead = df_steal_attempts[kinematic_cols].quantile(0.55).to_frame().T
very_agg_lead = df_steal_attempts[kinematic_cols].quantile(0.6).to_frame().T
print('Very Safe Lead Payoffs:')
print_payoffs(very_safe_lead)
print("*********************************")
print('Safe Lead Payoffs')
print_payoffs(safe_lead)
print("*********************************")
print('Aggressive Lead Payoffs')
print_payoffs(agg_lead)
print("*********************************")
print('Very Aggressive Lead Payoffs')
print_payoffs(very_agg_lead)
print("*********************************")
# %%
# %%
# 2 branch arbitrary example payoffs
safe_lead = df_steal_attempts[kinematic_cols].quantile(0.27).to_frame().T
agg_lead = df_steal_attempts[kinematic_cols].quantile(0.55).to_frame().T
print('Safe Lead Payoffs')
print_payoffs(safe_lead)
print("*********************************")
print('Aggressive Lead Payoffs')
print_payoffs(agg_lead)
print("*********************************")
# %%
print_payoffs( df_leads[kinematic_cols].quantile(0.5).to_frame().T)
# %%
# 2-branch rickey henderson example
safe_lead = df_steal_attempts[kinematic_cols].quantile(0.25).to_frame().T
agg_lead = df_steal_attempts[kinematic_cols].quantile(0.55).to_frame().T
print('Safe Lead Payoffs')
print_payoffs(safe_lead, sprint_speed=31.5, pop_time=1.9)
print("*********************************")
print('Aggressive Lead Payoffs')
print_payoffs(agg_lead, sprint_speed=31.5, pop_time=1.9)
print("*********************************")
#%%
#2-branch pujols 
very_safe_lead = df_leads[kinematic_cols].quantile(0.11).to_frame().T
safe_lead = df_steal_attempts[kinematic_cols].quantile(0.25).to_frame().T
print('Very Safe Lead Payoffs')
print_payoffs(very_safe_lead)
print("*********************************")
print('Safe Lead Payoffs')
print_payoffs(safe_lead)
print("*********************************")
#%%
#3-branch arbitrary example payoffs
safe_lead = df_steal_attempts[kinematic_cols].quantile(0.25).to_frame().T
med_lead = df_leads[kinematic_cols].quantile(0.5).to_frame().T
agg_lead = df_steal_attempts[kinematic_cols].quantile(0.60).to_frame().T
print('Safe Lead Payoffs:')
print_payoffs(safe_lead)
print("*********************************")
print('Median Lead Payoffs')
print_payoffs(med_lead)
print("*********************************")
print('Aggressive Lead Payoffs')
print_payoffs(agg_lead)
print("*********************************")
# %%