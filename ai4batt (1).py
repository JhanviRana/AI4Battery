import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import seaborn as sns

def load_data(mat_filename, batch_number):
    print(f"Loading data from {mat_filename}")
    f = h5py.File(mat_filename, 'r')
    batch = f['batch']
    num_cells = batch['summary'].shape[0]
    print(f"Number of cells: {num_cells}")
    bat_dict = {}
    for i in range(num_cells):
        print(f"Processing cell {i}")
        Vdlin = f[batch['Vdlin'][i,0]][()]
        cl = f[batch['cycle_life'][i,0]][()]
        policy = f[batch['policy_readable'][i,0]][()].tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i,0]]['IR'][0,:].tolist())
        summary_QC = np.hstack(f[batch['summary'][i,0]]['QCharge'][0,:].tolist())
        summary_QD = np.hstack(f[batch['summary'][i,0]]['QDischarge'][0,:].tolist())
        summary_TA = np.hstack(f[batch['summary'][i,0]]['Tavg'][0,:].tolist())
        summary_TM = np.hstack(f[batch['summary'][i,0]]['Tmin'][0,:].tolist())
        summary_TX = np.hstack(f[batch['summary'][i,0]]['Tmax'][0,:].tolist())
        summary_CT = np.hstack(f[batch['summary'][i,0]]['chargetime'][0,:].tolist())
        summary_CY = np.hstack(f[batch['summary'][i,0]]['cycle'][0,:].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg': summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT, 'cycle': summary_CY}
        cycles = f[batch['cycles'][i,0]]
        cycle_dict = {}
        for j in range(cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j,0]][()]))
            Qc = np.hstack((f[cycles['Qc'][j,0]][()]))
            Qd = np.hstack((f[cycles['Qd'][j,0]][()]))
            Qdlin = np.hstack((f[cycles['Qdlin'][j,0]][()]))
            T = np.hstack((f[cycles['T'][j,0]][()]))
            Tdlin = np.hstack((f[cycles['Tdlin'][j,0]][()]))
            V = np.hstack((f[cycles['V'][j,0]][()]))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j,0]][()]))
            t = np.hstack((f[cycles['t'][j,0]][()]))
            cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V':V, 'dQdV': dQdV, 't':t}
            cycle_dict[str(j)] = cd

        cell_dict = {'Vdlin': Vdlin, 'cycle_life': cl, 'charge_policy':policy, 'summary': summary, 'cycles': cycle_dict}
        key = f'b{batch_number}c{i}'
        bat_dict[key] = cell_dict

    print("Data loaded successfully.")
    return bat_dict


# Deriving Energy_Discharge and Energy_Charge features in cycles data as well as summary data
def derive_energy_features(bat_dict):
    print("Deriving energy features...")
    for cell in bat_dict.keys():
        print(f"Processing cell {cell} for energy features")
        bat_dict[cell]['summary']['ED'] = np.zeros(shape = bat_dict[cell]['summary']['cycle'].shape)
        bat_dict[cell]['summary']['EC'] = np.zeros(shape = bat_dict[cell]['summary']['cycle'].shape)
        deltaQ = bat_dict[cell]['cycles']['99']['Qdlin'] - bat_dict[cell]['cycles']['9']['Qdlin']
        data = np.array([deltaQ])
        deltaQ_var = np.var(data, axis=1)
        bat_dict[cell]['deltaQ_var'] = deltaQ_var
        for cycle_no in bat_dict[cell]['cycles'].keys():
            bat_dict[cell]['cycles'][cycle_no]['Ed'] = (bat_dict[cell]['cycles'][cycle_no]['Qd'])*(bat_dict[cell]['cycles'][cycle_no]['V'])
            bat_dict[cell]['summary']['ED'][int(cycle_no)] = sum(bat_dict[cell]['cycles'][cycle_no]['Ed'])
            
            bat_dict[cell]['cycles'][cycle_no]['Ec'] = (bat_dict[cell]['cycles'][cycle_no]['Qc'])*(bat_dict[cell]['cycles'][cycle_no]['V'])
            bat_dict[cell]['summary']['EC'][int(cycle_no)] = sum(bat_dict[cell]['cycles'][cycle_no]['Ec'])

    print("Energy features derived successfully.")
    return bat_dict

# Function to find the minimum dQ/dV in a specified voltage range
def find_min_dQdV_in_range(dQdV_data, V_data, V_min, V_max):
    # Mask for values within the voltage range
    mask = (V_data >= V_min) & (V_data <= V_max)
    # Apply mask to get dQ/dV values within the range
    dQdV_in_range = dQdV_data[mask]
    # Find and return the minimum dQdV in the range
    min_dQdV = np.min(dQdV_in_range)
    return min_dQdV

# Loading data from all three .mat files
print("Loading batch 1 data")
batch1 = load_data('D://2017-05-12_batchdata_updated_struct_errorcorrect.mat',1)
print("Loading batch 2 data")
batch2 = load_data('D://2017-06-30_batchdata_updated_struct_errorcorrect.mat',2)
print("Loading batch 3 data")
batch3 = load_data('D://2018-04-12_batchdata_updated_struct_errorcorrect.mat',3)

# Merging all batches
print("Merging all batches")
bat_dict = {**batch1, **batch2, **batch3}



# Deriving energy features
print("Deriving energy features")
bat_dict = derive_energy_features(bat_dict)

# Extracting features and target values
print("Extracting features and target values")
X = []
y = []

for cell in bat_dict.keys():
    print(f"Processing cell {cell} for features and target extraction")
    available_cycles=list(map(int,bat_dict[cell]['cycles'].keys()))
    if 199 in available_cycles:
        selected_cycle=199
    else:
        selected_cycle=max(available_cycles)
    dQdV_data = np.array(bat_dict[cell]['cycles'][str(selected_cycle)]['dQdV'])
    V_data = np.array(bat_dict[cell]['Vdlin'][0])
    deltaQ_var=bat_dict[cell]['deltaQ_var']
    RUL = int(bat_dict[cell]['summary']['cycle'].max() - (selected_cycle+1))
    QD_in = int(bat_dict[cell]['summary']['QD'][0])
    QD_200 = int(bat_dict[cell]['summary']['QD'][selected_cycle])
    QD_delta = int(bat_dict[cell]['summary']['QD'][0]) - int(bat_dict[cell]['summary']['QD'][selected_cycle])
    IR_in = int(bat_dict[cell]['summary']['IR'][0])
    IR_200 = int(bat_dict[cell]['summary']['IR'][selected_cycle])
    charge_time=int(bat_dict[cell]['summary']['chargetime'][0])
    dQdV_1 = find_min_dQdV_in_range(dQdV_data, V_data, 2.2, 2.6)
    dQdV_2 = find_min_dQdV_in_range(dQdV_data, V_data, 3.0, 3.3)
    
    dis_cap = []
    cycle_index = []
    IR = []
    for a in bat_dict[cell]['cycles'].keys():
        if int(a) in range(0,selected_cycle+1):
            dis_cap.append(bat_dict[cell]['summary']['QD'][int(a)])
            IR.append(bat_dict[cell]['summary']['IR'][int(a)])
            cycle_index.append(int(a))
    
    QD_slope, QD_intercept = np.polyfit(cycle_index, dis_cap, 1)
    IR_slope, IR_intercept = np.polyfit(cycle_index, IR, 1)
    
    features = [QD_in, QD_200, QD_delta, IR_in, IR_200, dQdV_1, dQdV_2, QD_slope, QD_intercept, IR_slope, IR_intercept,charge_time, deltaQ_var[0]]
    X.append(features)
    y.append(RUL)

import pandas as pd

print("Features and target values extracted")
X = np.array(X)
y = np.array(y)
feature_names = ['QD_in', 'QD_200', 'QD_delta', 'IR_in', 'IR_200', 'dQdV_1', 'dQdV_2', 'QD_slope', 'QD_intercept', 'IR_slope', 'IR_intercept', 'charge_time','deltaQ_var[0]']

X_df = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y, name='Target')  # Ensure y is a pandas Series with a name

# Ensure X and y have the same number of rows
assert X_df.shape[0] == y.shape[0], "X and y must have the same number of rows."

# Concatenate X and y into a single DataFrame
df_combined = pd.concat([X_df, y], axis=1)

# Calculate the correlation matrix
corr_matrix = df_combined.corr()

# Extract correlations with the target variable
target_corr = corr_matrix[['Target']].drop(['Target'], axis=0)

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Adjust the size as needed
sns.heatmap(target_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
plt.title('Pearson Correlation Heatmap Between Features and Target')
plt.show()

# Prepare data with selected features
X_selected = X_df[selected_features].values


# Splitting data into training and testing sets
print("Splitting data into training and testing sets")
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb

from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    print(f"Evaluating model: {model}")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_pred_train = np.round(y_pred_train).astype(int)
    y_pred_test = np.round(y_pred_test).astype(int)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"Train MSE: {mse_train}, Train RMSE: {rmse_train}, Train MAPE: {mape_train}, Train R^2: {r2_train}")
    print(f"Test MSE: {mse_test}, Test RMSE: {rmse_test}, Test MAPE: {mape_test}, Test R^2: {r2_test}")
    
    return model, y_pred_test, y_test

# Linear Regression
print("Linear Regression:")
lr_model = LinearRegression()
evaluate_model(lr_model, X_train, X_test, y_train, y_test)

#Lasso Regression
print("\n Lasso Regression")
las_model = Lasso(alpha=1.0)
evaluate_model(las_model, X_train, X_test, y_train, y_test)

# Elastic Net
print("\nElastic Net:")
en_model = ElasticNet(random_state=42)
evaluate_model(en_model, X_train, X_test, y_train, y_test)


param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3.5]
}

# Step 4: Initialize XGBoost regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Step 5: Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xg_reg,
    param_distributions=param_dist,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

# Step 6: Fit the random search model
random_search.fit(X_train, y_train)

# Step 7: Output the best parameters and the corresponding score
best_params = random_search.best_params_

print("\nXGBoost:")
xgb_model = XGBRegressor(subsample= 1.0, reg_lambda= 2, reg_alpha= 0, n_estimators=200, min_child_weight= 1,max_depth= 3, learning_rate= 0.3, gamma= 0.2, colsample_bytree=0.6,random_state=42)
evaluate_model(xgb_model, X_train, X_test, y_train, y_test)