Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>



Cette partie explique dans quel ordre tester les codes si vous souhaitez visualiser les codes. On rappelle que ce projet est un problème de classification. Pendant ce projet, deux datasets ont été utilisés. Lorsque l'on évoquera le "premier dataset" ou "le dataset initial", nous faisons référence aux données récupérées sur ce site : "https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/#_". Lorsque l'on évoque "le second dataset", on fait référence aux données récupérées sur ce site : "https://co2cars.apps.eea.europa.eu/?source". Ces deux datasets sont fusionnés pour fournir un autre dataset que l'on nommera le "dataset mergé. Dans nos études, nous appliquons des modèles sur le "premier dataset" et le "dataset mergé". Le "second dataset" ne sera jamais utilisé ou il sera très rapidement exploré. Il sert juste à faire la fusion pour acquérir quelques nouvelles variables intéressentes.

Ordre pour tester les codes :
Data exploration et data visulisation:
- "Exploration_1er_dataset.ipynb" : ce premier fichier explore le premier dataset. Il y a également de la data visualisation sur le premier dataset.
- "1.0-ugo-data-exploration.py" : explore également le premier dataset. C'est un brouillon avec peu de nouvelles informations, on ne vous conseille pas de le regarder mais si vous êtes curieux vous pouvez jeter un coup d'oeil.
- "3.0-ugo-data-exploration.py" : explore le second dataset. C'est un brouillon, on ne vous conseille pas de le regarder mais si vous êtes curieux vous pouvez jeter un coup d'oeil.

Merge :
- "2.0-ugo-data-exploration.py" : C'est un brouillon qui pour trouver un moyen de fusionner les deux datasets. On ne vous conseille pas de le regarder car c'est un brouillon mais si vous êtes curieux, vous pouvez jeter un oeil.
- "4.0-ugo-merge.py" : Le merge se fait en deux parties à cause d'un problème de fichier trop lourd à télécharger. L'explication de cette décomposition du code en deux fichiers se trouve en commentaire au début de ce fichier. Ce fichier consiste juste à merger les deux datasets.
- "4.1-ugo-merge.ipynb" : deuxième partie du merge. On nettoie le dataset mergé. Puis on fait de la data visualisation sur les données mergées.

Modélisation:
- "Modelisation_1er_dataset.ipynb" : Ce fichier effectue plusieurs modélisations en utilisant le premier dataset.
- "5.0-ugo-first-models" : Ce fichier effectue plusieurs modélisations sur le dataset mergé.

