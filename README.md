# SentinelX AI — Maintenance prédictive (S2 + S3)

SentinelX AI vise à estimer la **durée de vie utile restante (RUL)** de moteurs d'avion à partir de leurs capteurs, sur les données publiques NASA C-MAPSS. Le projet complet (modèle séquentiel, anomalies, Health Score, API, interface…) est décrit dans le [cahier des charges](docs/Cahier_des_charges_SentinelX_AI.pdf).

**Ce dépôt ne contient que les deux premières étapes réalisées :**

| Étape | Contenu | État |
|---|---|---|
| **S2** — données et EDA | qualité des données, analyse des capteurs, sélection de 14 capteurs, cible RUL, séparation 80 / 20 moteurs | terminé, notebook exécuté |
| **S3** — baseline | features d'historique causales, régression linéaire, Ridge et forêt aléatoire, validation croisée par moteur, test NASA | terminé, notebook exécuté |

Aucune autre fonctionnalité du cahier des charges n'est encore développée.

## Structure

```text
sentinelx-ai/
├── data/
│   ├── README.md                 # téléchargement et empreintes des fichiers NASA
│   └── raw/                      # train_FD001.txt, test_FD001.txt, RUL_FD001.txt (non versionnés)
├── notebooks/
│   ├── 01_eda_fd001.ipynb        # S2
│   └── 02_baseline_fd001.ipynb   # S3
├── ml/src/
│   ├── data.py                   # chargement, RUL, dernier cycle, séparation par moteur
│   ├── features.py               # moyenne, pente et écart au début de vie (passé uniquement)
│   ├── evaluation.py             # MAE / RMSE / R², zone d'évaluation, plis par moteur
│   └── models.py                 # 3 modèles, entraînement, validation croisée
├── tests/                        # 39 tests pytest
├── artifacts/
│   ├── eda_fd001/                # 3 JSON de décisions + 8 figures
│   └── baseline_fd001/           # métriques JSON, prédictions test CSV, modèle .joblib + 8 figures
├── docs/Cahier_des_charges_SentinelX_AI.pdf
├── requirements.txt
└── pytest.ini
```

Les notebooks importent les fonctions de `ml/src/` : le code testé est exactement celui qui produit les résultats.

## Installation (Windows, PowerShell)

Prérequis : Python 3.11 ou plus récent (testé avec 3.13).

```powershell
cd sentinelx-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloque l'activation : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis relancer l'activation.

## Données

Télécharger NASA C-MAPSS et copier `train_FD001.txt`, `test_FD001.txt` et `RUL_FD001.txt` dans `data/raw/` (détails et empreintes SHA-256 : [`data/README.md`](data/README.md)).

## Exécution

Ouvrir les notebooks dans VS Code ou Jupyter avec le noyau de `.venv`, et les exécuter **dans l'ordre** :

1. `notebooks/01_eda_fd001.ipynb` — environ 15 secondes ; écrit `artifacts/eda_fd001/`.
2. `notebooks/02_baseline_fd001.ipynb` — environ 5 minutes (validation croisée) ; relit les décisions de l'EDA et écrit `artifacts/baseline_fd001/`.

Chaque notebook se termine par une cellule qui revérifie les chiffres cités dans son texte.

## Tests

```powershell
python -m pytest -v
```

34 tests unitaires sur de petits exemples déterministes (RUL, séparation sans chevauchement de moteurs, features, absence de fuite du futur, métriques, cas limites) et 5 tests de cohérence sur les vraies données et les artefacts. Ces 5 derniers sont **ignorés** (`skipped`) si `data/raw/` est vide ou si les notebooks n'ont pas encore été exécutés.

## Méthode

- **Séparation par moteur** : 80 moteurs d'entraînement, 20 de validation (graine 42), jamais par ligne.
- **Features causales** : pour 14 capteurs, moyenne et pente sur les 30 derniers cycles, écart à la moyenne des 20 premiers cycles → 42 features, calculées moteur par moteur avec le passé uniquement.
- **Choix** (features, plafond du RUL, `max_features`, `alpha`) par validation croisée `GroupKFold(5)` sur les 80 moteurs, règle fixée à l'avance : plus petit RMSE moyen sur la zone RUL réel ≤ 127.
- **Validation** : le modèle choisi est vérifié une seule fois sur les 20 moteurs mis de côté.
- **Test NASA** : réentraînement sur les 100 moteurs, prédiction au dernier cycle observé, `RUL_FD001.txt` lu une seule fois.

## Résultats obtenus

Plafond du RUL retenu : **110 cycles** pour les trois modèles. RMSE en cycles.

| Modèle | RMSE validation croisée (80 moteurs) | RMSE validation (20 moteurs) | RMSE test | MAE test | R² test |
|---|---|---|---|---|---|
| Régression linéaire | 16,8 | 15,7 | 19,1 | 15,1 | 0,79 |
| Ridge (`alpha` = 1000) | 16,6 | 15,7 | 19,2 | 15,2 | 0,79 |
| **Forêt aléatoire (retenue)** | **12,8** | **11,4** | **14,4** | **10,5** | **0,88** |

- Les features d'historique font passer la forêt de 20,7 à 13,8 en validation croisée (plafond 125) ; sans plafond, elle monte à 23,7.
- Près de la panne (RUL ≤ 25), la forêt se trompe en moyenne de 2,8 cycles sur les moteurs de validation, avec un léger biais optimiste.
- Ridge n'améliore pas significativement la régression linéaire : la limite vient de la non-linéarité de la dégradation, pas de la redondance des capteurs.

**Validation et test ne sont pas directement comparables** : la validation évalue toutes les lignes de RUL ≤ 127 de moteurs suivis jusqu'à la panne ; le test évalue un seul point par moteur, avec des RUL réels jusqu'à 145 (28 moteurs dépassent le plafond).

**Indépendance du test.** Dans l'ancien projet, `RUL_FD001.txt` avait déjà été lu par plusieurs versions de la baseline (RMSE 22,69 ; 14,23 ; 17,05 ; 14,4) et les réglages actuels en héritent. Le test n'a jamais servi directement à un choix, mais **son score est probablement un peu optimiste**. Les scores de validation croisée et de validation sont les références les plus propres.

## Limites

Données simulées, une seule condition de fonctionnement et un seul mode de panne (FD001) ; 20 moteurs de validation seulement ; aucune distinction au-delà de 110 cycles restants ; pas d'intervalle de confiance.
