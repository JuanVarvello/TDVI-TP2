import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from hyperopt import fmin, tpe, hp
from xgboost import XGBClassifier  # Import XGBClassifier


# Load the competition data
comp_data = pd.read_csv("competition_data.csv")

# Separa el archivo en train y eval dependiendo de la columna ROW_ID
train_data = comp_data[comp_data["ROW_ID"].isna()]
eval_data = comp_data[comp_data["ROW_ID"].notna()]

# Separamos el train set en train y validation set. 
#  - El validation set es el 20% del train set
#  - El random_state es para que siempre se divida de la misma forma
#  - El train set es el 80% del train set original

train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Preparamos la data de training 
y_train = train_data["conversion"]
X_train = train_data.drop(columns=["conversion", "ROW_ID"])
X_train = X_train.select_dtypes(include='number')

# Preparamos la data de validación
y_val = val_data["conversion"]
X_val = val_data.drop(columns=["conversion", "ROW_ID"])
X_val = X_val.select_dtypes(include='number')

# Preparamos la data de evaluación
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')

# Entrenamos el modelo con un pipeline que primero imputa los valores faltantes y luego entrena un árbol de decisión
cls = make_pipeline(SimpleImputer(), XGBClassifier(max_depth=8, random_state=2345))  # Usamos XGBClassifier en lugar de DecisionTreeClassifier
cls.fit(X_train, y_train)

# Evaluamos el modelo con el validation set
val_score = cls.score(X_val, y_val)
print(f"Validation Accuracy: {val_score:.4f}")

# Me quedo solo con los atributos que tienen una importancia mayor a 0.05
feature_importances = cls.named_steps['xgbclassifier'].feature_importances_
feature_names = X_train.columns
selected_feature_names = [feature_name for feature_name, importance in zip(feature_names, feature_importances) if importance >= 0.02]
X_train_selected = X_train[selected_feature_names]
X_val_selected = X_val[selected_feature_names]


# Hacemos busqueda de hyperparámetros con hyperopt
def objective(params):
    max_depth = int(params['max_depth'])
    random_state = params['random_state']
    
    cls = make_pipeline(SimpleImputer(), XGBClassifier(max_depth=max_depth, random_state=random_state))  # Usamos XGBClassifier en lugar de DecisionTreeClassifier
    cls.fit(X_train_selected, y_train)
    val_score = cls.score(X_val_selected, y_val)
    return -val_score  # Hyperopt minimizes the objective, so we negate the score

# Definimos el espacio de búsqueda de hyperparámetros
space = {
    'max_depth': hp.quniform('max_depth', 1, 40, 1),
    'random_state': hp.choice('random_state', [10, 42, 123, 234, 345])
}


# Corremos la búsqueda de hyperparámetros
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)
best_max_depth = int(best['max_depth'])
best_random_state = [42, 123, 234, 345][best['random_state']]

# Entrenamos el modelo con los mejores hyperparámetros
eval_data_selected = eval_data[selected_feature_names]
best_cls = make_pipeline(SimpleImputer(), XGBClassifier(max_depth=best_max_depth, random_state=best_random_state))  # Usamos XGBClassifier en lugar de DecisionTreeClassifier
best_cls.fit(X_train_selected, y_train)
y_preds = best_cls.predict_proba(eval_data_selected)[:, best_cls.classes_ == 1].squeeze()
score = best_cls.score(X_val_selected, y_val)
print(f"Validation Accuracy (best parametros): {score:.4f}")

# Hacemos el subimssion file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)

