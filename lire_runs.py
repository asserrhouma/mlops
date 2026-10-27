import mlflow

# Lire les runs et les trier par score F1
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

print(runs[colonnes].to_string(index=False))

meilleur = runs.iloc[0]
print("\nMeilleur run :", meilleur["tags.mlflow.runName"])
print("Run ID :", meilleur["run_id"])