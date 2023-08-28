import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

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
cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
cls.fit(X_train, y_train)

# Evaluamos el modelo con el validation set
val_score = cls.score(X_val, y_val)
print(f"Validation Accuracy: {val_score:.4f}")

# Si la importancia del atributo es menor a 0.05, se elimina
feature_importances = cls.named_steps['decisiontreeclassifier'].feature_importances_
feature_names = X_train.columns
for feature_name, importance in zip(feature_names, feature_importances):
    if importance < 0.05:
        X_train = train_data.drop(columns=feature_name)

# Predecimos la probabilidad de conversión para el eval set
y_preds = cls.predict_proba(eval_data.drop(columns=["ROW_ID"]))[:, cls.classes_ == 1].squeeze()

# Hacemos el subimssion file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)

