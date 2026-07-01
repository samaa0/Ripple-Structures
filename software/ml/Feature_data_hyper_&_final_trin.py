# feature_data_hyperparameter_optimization_gpu.py

# Import necessary libraries
import pandas as pd
import numpy as np
import optuna
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json

# Set Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.INFO)

# GPU Management Code
# Initialize a lock for GPU assignment
gpu_lock = threading.Lock()

# Automatically detect available GPUs
available_gpus = list(range(torch.cuda.device_count()))

def get_assigned_gpu_id():
    with gpu_lock:
        if available_gpus:
            gpu_id = available_gpus.pop()
            print(f"Assigned GPU {gpu_id}")
            return gpu_id
        else:
            print("No GPU available currently. Will wait for one to become available.")
            return None  # Return None if no GPU is available

def release_gpu_id(gpu_id):
    with gpu_lock:
        available_gpus.append(gpu_id)
        print(f"Released GPU {gpu_id}")

# Step 1: Load and Prepare the Data
# Load normalized feature datasets
train_features_normalized = pd.read_csv('train_features_normalized.csv')
validation_features_normalized = pd.read_csv('validation_features_normalized.csv')
test_features_normalized = pd.read_csv('test_features_normalized.csv')

# Combine training and validation sets
train_val_data = pd.concat([train_features_normalized, validation_features_normalized], ignore_index=True)

# Prepare Features and Labels
# Separate features and labels
feature_columns = [col for col in train_val_data.columns if col not in ['label', 'sequence']]

X = train_val_data[feature_columns]
y = train_val_data['label']

# For test set
X_test = test_features_normalized[feature_columns]
y_test = test_features_normalized['label']

# If labels are categorical strings, encode them as integers
# Uncomment and adjust the following lines if needed
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
# y_test = le.transform(y_test)

# Step 2: Set Up Stratified K-Fold Cross-Validation
# Set up Stratified K-Fold cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Step 3: Hyperparameter Optimization with Optuna
def objective(trial):
    # Get an available GPU
    gpu_id = get_assigned_gpu_id()
    try:
        device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
        print(f"Using device: {device}")

        model_type = trial.suggest_categorical('model_type', ['LightGBM', 'XGBoost', 'RandomForest', 'MLP'])

        if model_type == 'LightGBM':
            return objective_lgbm(trial, device)
        elif model_type == 'XGBoost':
            return objective_xgb(trial, device)
        elif model_type == 'RandomForest':
            return objective_rf(trial)
        elif model_type == 'MLP':
            return objective_mlp(trial, device)
    finally:
        if gpu_id is not None:
            release_gpu_id(gpu_id)

def objective_lgbm(trial, device):
    import lightgbm as lgb

    # Extract the device (e.g., 'cuda:0' or 'cpu')
    gpu_id = int(device.split(':')[-1]) if device != 'cpu' else -1

    # Hyperparameter space with unique parameter names
    param = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'num_leaves': trial.suggest_categorical('lgb_num_leaves', [31, 63, 127, 255]),
        'max_depth': trial.suggest_categorical('lgb_max_depth', [-1, 5, 10, 15]),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 1e-5, 1e-1, log=True),
        'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.6, 1.0, step=0.1),
        'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.6, 1.0, step=0.1),
        'bagging_freq': trial.suggest_categorical('lgb_bagging_freq', [0, 5, 10]),
        'min_data_in_leaf': trial.suggest_categorical('lgb_min_data_in_leaf', [20, 40, 60, 80, 100]),
        'lambda_l1': trial.suggest_categorical('lgb_lambda_l1', [0.0, 0.1, 0.5, 1.0]),
        'lambda_l2': trial.suggest_categorical('lgb_lambda_l2', [0.0, 0.1, 0.5, 1.0]),
        'verbosity': -1,  # Suppress warnings
    }

    if device != 'cpu':
        param['device'] = 'gpu'
        param['gpu_device_id'] = gpu_id

    num_boost_round = 1000
    early_stopping_rounds = 30

    fold_f1_scores = []

    for train_index, valid_index in skf.split(X, y):
        X_train_fold = X.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        X_valid_fold = X.iloc[valid_index]
        y_valid_fold = y.iloc[valid_index]

        # Compute class weights
        classes = np.unique(y_train_fold)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_fold)
        class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
        weights = y_train_fold.map(class_weights_dict)

        lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, weight=weights)
        lgb_valid = lgb.Dataset(X_valid_fold, label=y_valid_fold)

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=10)
        ]

        # Train the model
        gbm = lgb.train(
            param,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_valid],
            callbacks=callbacks
        )

        # Predict
        y_pred_probs = gbm.predict(X_valid_fold, num_iteration=gbm.best_iteration)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)

        # Compute F1-Score
        f1 = f1_score(y_valid_fold, y_pred_labels, average='macro')
        fold_f1_scores.append(f1)

    avg_f1 = np.mean(fold_f1_scores)

    # Save the model and hyperparameters
    # Retrain the model on the entire training data
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    weights = y.map(class_weights_dict)

    lgb_full_train = lgb.Dataset(X, label=y, weight=weights)

    gbm_full = lgb.train(
        param,
        lgb_full_train,
        num_boost_round=num_boost_round
    )

    # Save the model
    model_filename = f"lgbm_model_trial_{trial.number}.txt"
    gbm_full.save_model(model_filename)

    # Save hyperparameters and performance
    trial.set_user_attr("model_filename", model_filename)
    trial.set_user_attr("hyperparameters", param)
    trial.set_user_attr("avg_f1", avg_f1)

    return -avg_f1

def objective_xgb(trial, device):
    import xgboost as xgb

    # Extract the device (e.g., 'cuda:0' or 'cpu')
    xgb_device = device  # Ensure this is a string like 'cuda:0' or 'cpu'

    # Hyperparameter space with unique parameter names
    param = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'max_depth': trial.suggest_categorical('xgb_max_depth', [3, 5, 7, 9]),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 1e-5, 1e-1, log=True),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0, step=0.1),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0, step=0.1),
        'lambda': trial.suggest_categorical('xgb_lambda', [0.0, 0.1, 0.5, 1.0]),
        'alpha': trial.suggest_categorical('xgb_alpha', [0.0, 0.1, 0.5, 1.0]),
        'verbosity': 0,
        'tree_method': 'gpu_hist' if 'cuda' in xgb_device else 'hist',
        'device': xgb_device,  # Use 'device' instead of 'gpu_id'
    }

    # Training parameters
    num_rounds = 1000
    early_stopping_rounds = 30

    fold_f1_scores = []

    for train_index, valid_index in skf.split(X, y):
        X_train_fold = X.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        X_valid_fold = X.iloc[valid_index]
        y_valid_fold = y.iloc[valid_index]

        # Compute class weights
        classes = np.unique(y_train_fold)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_fold)
        sample_weights = np.array([class_weights[np.where(classes == label)[0][0]] for label in y_train_fold])

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, weight=sample_weights)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        # Train the model
        bst = xgb.train(
            param,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dvalid, 'eval')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )

        # Predict
        y_pred_probs = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
        y_pred_labels = np.argmax(y_pred_probs, axis=1)

        # Compute F1-Score
        f1 = f1_score(y_valid_fold, y_pred_labels, average='macro')
        fold_f1_scores.append(f1)

    avg_f1 = np.mean(fold_f1_scores)

    # Save the model and hyperparameters
    # Retrain the model on the entire training data
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    sample_weights = np.array([class_weights[np.where(classes == label)[0][0]] for label in y])

    dtrain_full = xgb.DMatrix(X, label=y, weight=sample_weights)

    bst_full = xgb.train(
        param,
        dtrain_full,
        num_boost_round=num_rounds
    )

    # Save the model
    model_filename = f"xgb_model_trial_{trial.number}.model"
    bst_full.save_model(model_filename)

    # Save hyperparameters and performance
    trial.set_user_attr("model_filename", model_filename)
    trial.set_user_attr("hyperparameters", param)
    trial.set_user_attr("avg_f1", avg_f1)

    return -avg_f1

def objective_rf(trial):
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    # Hyperparameter space with unique parameter names
    param = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 50, step=5),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('rf_bootstrap', [True, False]),
        'class_weight': 'balanced',
            'n_jobs': -1,
    }

    fold_f1_scores = []

    for train_index, valid_index in skf.split(X, y):
        X_train_fold = X.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        X_valid_fold = X.iloc[valid_index]
        y_valid_fold = y.iloc[valid_index]

        # Train the model
        clf = RandomForestClassifier(**param)
        clf.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred_labels = clf.predict(X_valid_fold)

        # Compute F1-Score
        f1 = f1_score(y_valid_fold, y_pred_labels, average='macro')
        fold_f1_scores.append(f1)

    avg_f1 = np.mean(fold_f1_scores)

    # Retrain the model on the entire training data
    clf_full = RandomForestClassifier(**param)
    clf_full.fit(X, y)

    # Save the model
    model_filename = f"rf_model_trial_{trial.number}.joblib"
    joblib.dump(clf_full, model_filename)

    # Save hyperparameters and performance
    trial.set_user_attr("model_filename", model_filename)
    trial.set_user_attr("hyperparameters", param)
    trial.set_user_attr("avg_f1", avg_f1)

    return -avg_f1

def objective_mlp(trial, device):
    # Serialize the choices into strings to avoid warnings
    hidden_layer_sizes_choices = [
        str((units,) * layers)
        for layers in [1, 2, 3]
        for units in [64, 128, 256]
    ]
    hidden_layer_sizes_str = trial.suggest_categorical('mlp_hidden_layer_sizes', hidden_layer_sizes_choices)
    hidden_layer_sizes = eval(hidden_layer_sizes_str)

    activation = trial.suggest_categorical('mlp_activation', ['relu', 'tanh'])
    learning_rate_init = trial.suggest_float('mlp_learning_rate_init', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('mlp_batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_float('mlp_dropout_rate', 0.0, 0.5)
    weight_decay = trial.suggest_float('mlp_weight_decay', 1e-5, 1e-2, log=True)

    max_epochs = 200
    early_stopping_patience = 10

    num_classes = len(np.unique(y))

    fold_f1_scores = []

    for train_index, valid_index in skf.split(X, y):
        X_train_fold = torch.tensor(X.iloc[train_index].values, dtype=torch.float32)
        y_train_fold = torch.tensor(y.iloc[train_index].values, dtype=torch.long)
        X_valid_fold = torch.tensor(X.iloc[valid_index].values, dtype=torch.float32)
        y_valid_fold = torch.tensor(y.iloc[valid_index].values, dtype=torch.long)

        # Compute class weights
        classes = np.unique(y_train_fold)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_fold.numpy())
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        valid_dataset = TensorDataset(X_valid_fold, y_valid_fold)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Define the model
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_layers, num_classes, activation_fn, dropout_rate):
                super(MLP, self).__init__()
                layers = []
                prev_size = input_size
                for hidden_size in hidden_layers:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(activation_fn)
                    if dropout_rate > 0.0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev_size = hidden_size
                layers.append(nn.Linear(prev_size, num_classes))
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        activation_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()

        model = MLP(
            input_size=X.shape[1],
            hidden_layers=hidden_layer_sizes,
            num_classes=num_classes,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_init, weight_decay=weight_decay)

        best_val_loss = np.inf
        epochs_no_improve = 0

        # Training loop
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # Validation loop
            model.eval()
            val_losses = []
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch_X, batch_y in valid_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    targets = batch_y.cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(targets)
            val_loss = np.mean(val_losses)
            val_f1 = f1_score(all_targets, all_preds, average='macro')

            # Print progress
            print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    break  # Early stopping

        # Compute F1-Score for this fold
        f1 = val_f1  # Using the last validation F1-Score
        fold_f1_scores.append(f1)

    avg_f1 = np.mean(fold_f1_scores)

    # Retrain the model on the entire training data
    X_train_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y.values, dtype=torch.long).to(device)

    # Compute class weights
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    model_full = MLP(
        input_size=X.shape[1],
        hidden_layers=hidden_layer_sizes,
        num_classes=num_classes,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model_full.parameters(), lr=learning_rate_init, weight_decay=weight_decay)

    # Train the model
    model_full.train()
    for epoch in range(max_epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model_full(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Full Training Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_loss:.4f}')

    # Save the model
    model_filename = f"mlp_model_trial_{trial.number}.pth"
    torch.save(model_full.state_dict(), model_filename)

    # Save hyperparameters and performance
    trial.set_user_attr("model_filename", model_filename)
    trial.set_user_attr("hyperparameters", {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'learning_rate_init': learning_rate_init,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'weight_decay': weight_decay
    })
    trial.set_user_attr("avg_f1", avg_f1)

    return -avg_f1

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)


# Get the top 3 best trials
top_3_trials = study.best_trials[:3]

# Function to retrain and evaluate top models
def retrain_and_save_top_models(trials, X_train, y_train, X_test, y_test):
    for idx, trial in enumerate(trials):
        print(f"\nRetraining model ranked {idx+1} with trial number {trial.number}")
        retrain_best_model(trial, X_train, y_train, X_test, y_test, rank=idx+1)


# Retrain the best model on the entire training data and evaluate on the test set
def retrain_best_model(trial, X_train, y_train, X_test, y_test, rank=1):
    model_type = trial.params['model_type']
    gpu_id = get_assigned_gpu_id()
    try:
        device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
        print(f"Using device: {device}")

        if model_type == 'LightGBM':
            import lightgbm as lgb

            # Load hyperparameters
            param = trial.user_attrs['hyperparameters']
            param['objective'] = 'multiclass'
            param['num_class'] = len(np.unique(y_train))
            param['verbosity'] = -1

            if device != 'cpu':
                param['device'] = 'gpu'
                param['gpu_device_id'] = gpu_id

            num_boost_round = 1000

            # Compute class weights
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
            weights = y_train.map(class_weights_dict)

            lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights)

            # Retrain the model
            gbm = lgb.train(
                param,
                lgb_train,
                num_boost_round=num_boost_round
            )

            # Save the model
            model_filename = f"best_lgbm_model_rank_{rank}.txt"
            gbm.save_model(model_filename)

            # Predict
            y_pred_probs = gbm.predict(X_test)
            y_pred_labels = np.argmax(y_pred_probs, axis=1)

        elif model_type == 'XGBoost':
            import xgboost as xgb

            xgb_device = device
            param = trial.user_attrs['hyperparameters']
            param['objective'] = 'multi:softprob'
            param['num_class'] = len(np.unique(y_train))
            param['verbosity'] = 0
            param['tree_method'] = 'gpu_hist' if 'cuda' in xgb_device else 'hist'
            param['device'] = xgb_device

            num_rounds = 1000

            # Compute class weights
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([class_weights[np.where(classes == label)[0][0]] for label in y_train])

            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Retrain the model
            bst = xgb.train(
                param,
                dtrain,
                num_boost_round=num_rounds
            )

            # Save the model
            model_filename = f"best_xgb_model_rank_{rank}.model"
            bst.save_model(model_filename)

            # Predict
            y_pred_probs = bst.predict(dtest)
            y_pred_labels = np.argmax(y_pred_probs, axis=1)

        elif model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            import joblib

            param = trial.user_attrs['hyperparameters']

            # Retrain the model
            clf = RandomForestClassifier(**param)
            clf.fit(X_train, y_train)

            # Save the model
            model_filename = f"best_rf_model_rank_{rank}.joblib"
            joblib.dump(clf, model_filename)

            # Predict
            y_pred_labels = clf.predict(X_test)

        elif model_type == 'MLP':
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader

            params = trial.user_attrs['hyperparameters']
            hidden_layer_sizes = params['hidden_layer_sizes']
            activation = params['activation']
            learning_rate_init = params['learning_rate_init']
            batch_size = params['batch_size']
            dropout_rate = params['dropout_rate']
            weight_decay = params['weight_decay']

            max_epochs = 200

            num_classes = len(np.unique(y_train))

            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

            # Compute class weights
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Define the model
            class MLP(nn.Module):
                def __init__(self, input_size, hidden_layers, num_classes, activation_fn, dropout_rate):
                    super(MLP, self).__init__()
                    layers = []
                    prev_size = input_size
                    for hidden_size in hidden_layers:
                        layers.append(nn.Linear(prev_size, hidden_size))
                        layers.append(activation_fn)
                        if dropout_rate > 0.0:
                            layers.append(nn.Dropout(dropout_rate))
                        prev_size = hidden_size
                    layers.append(nn.Linear(prev_size, num_classes))
                    self.model = nn.Sequential(*layers)

                def forward(self, x):
                    return self.model(x)

            activation_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()

            model = MLP(
                input_size=X_train.shape[1],
                hidden_layers=hidden_layer_sizes,
                num_classes=num_classes,
                activation_fn=activation_fn,
                dropout_rate=dropout_rate
            ).to(device)

            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate_init, weight_decay=weight_decay)

            # Initialize early stopping parameters
            best_val_loss = np.inf
            epochs_no_improve = 0
            early_stopping_patience = 10  # You can adjust this value
            
            # Split a validation set from the training data
            from sklearn.model_selection import train_test_split
            X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
            
            # Convert validation set to tensors
            X_valid_tensor = torch.tensor(X_valid_fold.values, dtype=torch.float32).to(device)
            y_valid_tensor = torch.tensor(y_valid_fold.values, dtype=torch.long).to(device)
            valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            
            # Update train_dataset and train_loader to use X_train_fold
            X_train_tensor = torch.tensor(X_train_fold.values, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.long).to(device)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Train the model with early stopping
            model.train()
            for epoch in range(max_epochs):
                running_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * batch_X.size(0)

                epoch_loss = running_loss / len(train_loader.dataset)
                print(f'Full Training Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_loss:.4f}')
                
                # Validation loop
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_X, batch_y in valid_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save the best model state
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        print("Early stopping triggered.")
                        # Load the best model state before breaking
                        model.load_state_dict(best_model_state)
                        break  # Early stopping
                
                model.train()
            
            # Save the model
            model_filename = f"best_mlp_model_rank_{rank}.pth"
            torch.save(model.state_dict(), model_filename)

            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                y_pred_probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                y_pred_labels = np.argmax(y_pred_probs, axis=1)

        # Compute performance metrics
        macro_f1 = f1_score(y_test, y_pred_labels, average='macro')
        accuracy = accuracy_score(y_test, y_pred_labels)
        kappa = cohen_kappa_score(y_test, y_pred_labels)
        conf_mat = confusion_matrix(y_test, y_pred_labels)

        # Save performance metrics and hyperparameters
        performance = {
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': conf_mat.tolist(),
            'hyperparameters': trial.user_attrs['hyperparameters']
        }

        log_filename = f"model_rank_{rank}_performance.json"
        with open(log_filename, 'w') as f:
            json.dump(performance, f, indent=4)

        print(f"Model saved as {model_filename}")
        print(f"Performance metrics saved in {log_filename}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print("Confusion Matrix:")
        print(conf_mat)

    finally:
        if gpu_id is not None:
            release_gpu_id(gpu_id)

# Retrain and save the top 3 models
retrain_and_save_top_models(top_3_trials, X, y, X_test, y_test)



