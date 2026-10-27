import mlflow
import mlflow.sklearn
import pandas as pd

# Retrouver le meilleur run selon le score F1
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

run_id = runs.iloc[0]["run_id"]

# Recharger le modèle enregistré dans ce run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Modèle rechargé :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)

# Créer un exemple de client
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# Garder les colonnes utilisées par le modèle, dans le bon ordre
client = client[list(model.feature_names_in_)]

prediction = model.predict(client)[0]
print("Le client va-t-il partir ?", "OUI" if prediction == 1 else "NON")