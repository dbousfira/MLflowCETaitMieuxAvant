<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">C'était mieux avant</h3>

  <p align="center">
    La société "Réfractaire", fleuron de la recherche médicale, utilise des algorithmes de Deep Learning pour ses projets d'analyses d'images (principalement autour des cancers). Aujourd'hui, ses projets fonctionnent mais la société fait parfois face à des soucis de régressions dans leurs applications. Après quelques recherche il s'avère, que une fois sur deux, l'algorithme de Deep Learning fait des erreurs. Elle vous sollicite pour l'aider.
  </p>
</p>

Contexte du projet

La société "Réfractaire" est une spécialiste de l'intelligence artificielle autour de la recherche médicale. Elle possède des Data Scientists et a déjà réalisée de nombreux algorithmes.

Elle n'est cependant pas très mature sur les outils dits "MLOps", qui permettent entre autre, de monitorer les résultats des algorithmes qu'elle conçoit. Algorithmes qui aujourd'hui, peuvent lui poser des problèmes lorsque l'entrainement dégrade les performances de ces derniers (pour diverses raisons).

Elle vous demande de lui présenter l'outil MLFlow à l'aide d'un projet (brief) sur lequel vous avez déjà travaillé. De lui montrer les principales fonctionnalités :

Création d'un fichier MLProject

Versionning avec Git

Gestion des paramètres entre les runs

Exécuter sur un fichier python et/ou un notebook

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo

    ```sh
    git clone https://github.com/dbousfira/c-etait-mieux-avant
    ```

2. Install mlflow:

```python
pip install mlflow
```

<!-- USAGE EXAMPLES -->
## Usage

```python
mlflow run mlflow_cifar
mlflow ui
```
