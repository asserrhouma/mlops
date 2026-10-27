import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Charger les données
df = pd.read_csv("churn_data.csv")
y = df["churn"]

# Créer ou sélectionner l'expérience
mlflow.set_experiment("churn-prediction")


def lancer_essai(nom, n_estimators, max_depth, colonnes):
    # Préparer les données
    X = pd.get_dummies(df[colonnes], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entraîner et enregistrer un essai
    with mlflow.start_run(run_name=nom):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
        mlflow.log_metric("precision", precision_score(y_test, predictions, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, predictions, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, predictions, zero_division=0))

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        print(
            nom,
            "| accuracy :", round(accuracy_score(y_test, predictions), 3),
            "| f1 :", round(f1_score(y_test, predictions, zero_division=0), 3)
        )


colonnes = ["age", "salaire", "abonnement", "anciennete", "consommation"]

lancer_essai("essai-1", 100, 5, colonnes)
lancer_essai("essai-2", 50, 3, colonnes)
lancer_essai(
    "essai-3", 100, 5,
    ["age", "abonnement", "anciennete", "consommation"]
)