# Behavioral Cloning

Le Behavioral Cloning (BC) est une méthode simple d'apprentissage par imitation où l'on cherche à reproduire directement le comportement d'un expert en traitant le problème comme une tâche de classification supervisée.

## Principe

L'objectif est d'apprendre une politique π qui imite au mieux les actions prises par l'expert. Pour cela, on :
1. Collecte des données de démonstration (état, action) de l'expert
2. Entraîne un réseau de neurones à prédire l'action de l'expert pour chaque état
3. Minimise la cross-entropy entre les prédictions et les vraies actions de l'expert

## Formulation mathématique

La loss à minimiser est :
$L_{BC} = -\mathbb{E}_{(s,a)\sim \mathcal{D}} [\log \pi_\theta(a|s)]$

où :
- $\mathcal{D}$ est le dataset de démonstrations expertes
- $\pi_\theta$ est notre politique paramétrée par θ
- $s$ est l'état
- $a$ est l'action prise par l'expert

## Avantages et Limitations

**Avantages** :
- Simple à implémenter
- Apprentissage stable (supervision directe)
- Pas besoin d'interaction avec l'environnement pendant l'entraînement

**Limitations** :
- Problème de distribution shift
- Nécessite beaucoup de données d'expert
- Pas de généralisation aux situations non vues dans les démonstrations

