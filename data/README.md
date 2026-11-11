# Données NASA C-MAPSS FD001

Placer ces trois fichiers, **sans les modifier**, dans `data/raw/` :

| Fichier | Contenu | SHA-256 |
|---|---|---|
| `train_FD001.txt` | 100 moteurs suivis jusqu'à la panne (20 631 lignes) | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | 100 moteurs arrêtés avant la panne (13 096 lignes) | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | vrai RUL des moteurs de test (évaluation finale uniquement) | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

Source officielle : <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data> (archive `CMAPSSData.zip`).

Vérifier une empreinte sous Windows (PowerShell) :

```powershell
Get-FileHash data\raw\train_FD001.txt -Algorithm SHA256
```

Le notebook de baseline vérifie automatiquement les empreintes de `train_FD001.txt` et `test_FD001.txt`.
