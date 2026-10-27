# Réflexion – TP3 : MLflow

## Quel est le meilleur modèle ?

Les essais 1 et 3 ont obtenu le même score F1 : environ 0,889.
L'essai 3 utilise une variable de moins, car il n'utilise pas le salaire.
Je peux donc le choisir pour sa simplicité. 

## Comment retrouver exactement le modèle d'un essai ?

MLflow enregistre un modèle pour chaque run. Je peux retrouver un run
grâce à son identifiant et recharger son modèle avec
`mlflow.sklearn.load_model("runs:/<run_id>/model")`.

## Un camarade peut-il reproduire le travail ?

Il peut récupérer le code, les données et les versions des bibliothèques
depuis GitHub, puis relancer les scripts. Les résultats MLflow enregistrés
sur mon ordinateur ne sont pas dans le dépôt : il devra créer ses propres
runs. Selon son environnement et les versions installées, les résultats
peuvent varier.