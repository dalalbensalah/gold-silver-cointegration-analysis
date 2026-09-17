# L'or et l'argent bougent-ils vraiment ensemble ? Pas toujours.

Analyse de cointégration, modélisation VAR/VECM et fonctions de réponse impulsionnelle sur 24 ans de données (2000-2024), pour tester une idée reçue du marché plutôt que la confirmer par défaut.

## Ce que ce projet montre en 30 secondes

- **La relation "évidente" entre or et argent ne l'est pas** : cointégration détectée mais épisodique, pas structurelle, elle apparaît et disparaît selon les périodes (analyse en fenêtre glissante)
- **Prévision hors échantillon sous 1,5 % de MAPE** sur les deux actifs avec les modèles VAR/VECM
- **Le contexte macro change la lecture, sans la trancher totalement.** Ajouter l'indice dollar et le taux 10 ans US fait apparaître un signal de cointégration plus net sur certaines périodes (Engle-Granger), mais ce signal n'est pas confirmé par tous les tests (Johansen) — la relation reste conditionnelle au régime macro, pas définitivement restaurée

Ce résultat illustre un enjeu classique en gestion des risques : une relation historique entre deux actifs peut se rompre sans prévenir, ce qui limite la fiabilité des stratégies qui supposent une corrélation stable dans le temps.

## Pourquoi ce sujet

L'or et l'argent sont classés dans la même case par la plupart des investisseurs : métaux précieux, valeurs refuges, censés bouger ensemble. Cette hypothèse est rarement testée sérieusement. Ce projet la met à l'épreuve avec des outils économétriques rigoureux, plutôt que de la prendre pour acquise.

## Structure du repo

​```
├── data/                              # Données brutes et transformées (prix or/argent, variables macro)
├── figures/                           # Graphiques générés (exploration, résultats, diagnostics)
├── notebooks/
│   ├── 01_Data_Exploration.ipynb                # Exploration des données et statistiques descriptives
│   ├── 02_Stationarity_and_Cointegration.ipynb  # Tests de stationnarité (ADF / KPSS) et cointégration
│   ├── 03_Bivariate_VAR_VECM.ipynb              # VAR/VECM bivarié (gold, silver)
│   ├── 04_Multivariate_VAR_VECM.ipynb           # VAR/VECM multivarié (+ dollar, taux 10 ans)
│   └── 05_Impulse_Response_and_Rolling.ipynb    # Causalité, IRF/GIRF et robustesse par régime macro
├── report/                            # Rapport détaillé du projet
└── src/
    ├── cointegration.py               # Fonctions de tests de cointégration
    ├── data_loader.py                 # Chargement et récupération des données (yfinance)
    ├── plots.py                       # Fonctions de visualisation
    ├── preprocessing.py               # Nettoyage et préparation des données
    ├── stationarity.py                # Fonctions de tests de stationnarité
    ├── utils.py                       # Fonctions utilitaires
    └── var_vecm.py                    # Fonctions de modélisation VAR/VECM
​```

## Méthodologie

1. **Tests de stationnarité** (ADF, KPSS) sur les séries de prix et leurs différences
2. **Tests de cointégration** : Engle-Granger et Johansen, avec analyse en fenêtre glissante pour capter l'instabilité temporelle de la relation
3. **Modélisation VAR/VECM** selon la présence ou non de cointégration, pour capturer les dynamiques de court et long terme
4. **Causalité de Granger (avec test de robustesse sur la longueur des retards), décomposition de Cholesky et fonctions de réponse impulsionnelle** : IRF orthogonalisées pour les VAR, IRF généralisées (Pesaran & Shin, 1998) pour les VECM, afin de s'affranchir de l'ordre arbitraire imposé par Cholesky

## Stack technique

Python · `pandas` · `numpy` · `statsmodels` · `yfinance` · `matplotlib`

## Reproduire le projet

​```bash
git clone https://github.com/dalalbensalah/gold-silver-cointegration-analysis.git
cd gold-silver-cointegration-analysis
pip install -r requirements.txt
jupyter notebook
​```

Lancer les notebooks dans l'ordre (01 → 05) pour reproduire l'analyse complète.

## Limites et pistes d'amélioration

- La relation étant instable dans le temps, un modèle à changement de régime (Markov-Switching) permettrait de mieux capturer les phases de convergence/divergence
- L'ajout d'autres variables macro-financières (inflation, taux réels, volatilité VIX) pourrait affiner la compréhension des facteurs communs
- Croiser avec les positions spéculatives (données COT - Commitment of Traders) pour voir si les phases de convergence/divergence coïncident avec des mouvements de positionnement des investisseurs institutionnels