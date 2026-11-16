# Homophily Research

Étiquetage non supervisé des arêtes par similarité textuelle pour GNN hétérophiles.

## Installer les données

```bash
pip install -r requirements.txt
python scripts/download_data.py              # par défaut : cornell texas wisconsin actor
python scripts/download_data.py amazon       # un dataset précis
python scripts/download_data.py cornell actor twitter   # plusieurs d'un coup
```

Les données vont dans `data/` (ignoré par git). Les fichiers déjà présents ne sont pas retéléchargés.

## Datasets disponibles

| Nom (commande) | Source | Fichier | Taille | Par défaut |
|---|---|---|---|---|
| `cornell` | HeTGB | `data/hetgb/Cornell.npz` | 9 Mo | oui |
| `texas` | HeTGB | `data/hetgb/Texas.npz` | 12 Mo | oui |
| `wisconsin` | HeTGB | `data/hetgb/Wisconsin.npz` | 29 Mo | oui |
| `actor` | HeTGB | `data/hetgb/Actor.npz` | 61 Mo | oui |
| `amazon` | HeTGB | `data/hetgb/Amazon.npz` | 255 Mo | non (lourd) |
| `twitter` | TEG-DB | `data/teg_db/twitter/` | ? | non |

Cornell, Texas, Wisconsin : pages web de départements d'informatique (5 classes : student, course, project, staff, faculty).
Actor : graphe d'acteurs. Amazon : produits. Twitter : réseau social avec texte sur les nœuds et les arêtes.

## Contenu d'un fichier HeTGB (`.npz`)

Vérifié sur Cornell, Texas, Wisconsin et Actor ; Amazon devrait avoir le même format (non vérifié).

| Clé | Contenu | Forme |
|---|---|---|
| `edges` | arêtes du graphe (source, destination) | `[2, nb_arêtes]` |
| `node_labels` | classe de chaque nœud | `[nb_nœuds]` |
| `node_features` | embeddings du texte (4096 dimensions) | `[nb_nœuds, 4096]` |
| `node_texts` | texte brut de chaque nœud | `[nb_nœuds]` |
| `label_texts` | nom des classes | `[nb_classes]` |
| `train_masks`, `val_masks`, `test_masks` | split train / validation / test | `[nb_nœuds]` |

```python
import numpy as np
d = np.load("data/hetgb/Cornell.npz")
d["edges"], d["node_texts"]
```

## Contenu du dataset TEG-DB (`twitter`)

Format différent (non vérifié, d'après le README de TEG-DB) : un objet PyTorch Geometric avec
`edge_index`, `text_nodes`, `text_edges` et `node_labels`.
