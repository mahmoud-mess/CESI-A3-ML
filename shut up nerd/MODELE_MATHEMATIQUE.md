# Modélisation Mathématique : Prédiction de l'Attrition

Ce document formalise le problème d'attrition des employés sous la forme d'un problème d'apprentissage statistique supervisé.

## 1. Définition du Problème

Soit un ensemble de données $D = \{(x_i, y_i)\}_{i=1}^N$ représentant $N$ employés.

*   $x_i \in \mathbb{R}^d$ est le vecteur des caractéristiques (features) de l'employé $i$.
*   $y_i \in \{0, 1\}$ est la variable cible (target), où :
    *   $y_i = 1$ signifie que l'employé a quitté l'entreprise (Attrition = Yes).
    *   $y_i = 0$ signifie que l'employé est resté (Attrition = No).

L'objectif est d'apprendre une fonction de prédiction $f: \mathbb{R}^d \to [0, 1]$ qui estime la probabilité conditionnelle de départ :

$$ \hat{y}_i = P(Y=1 | X=x_i) = f(x_i; \theta) $$

où $\theta$ représente les paramètres du modèle appris.

## 2. Espace des Caractéristiques (Feature Space)

Le vecteur $x_i$ est composé de différents types de variables transformées.

### 2.1 Variables Numériques ($X_{num}$)
Les variables continues ou discrètes ordonnées sont normalisées ou utilisées telles quelles (selon l'algorithme).
Exemples : `Age`, `MonthlyIncome`, `DistanceFromHome`, `NumCompaniesWorked`.

Transformation dérivée des séries temporelles ($T_{in}, T_{out}$) :
Soit $t_{in}^{(d)}$ et $t_{out}^{(d)}$ les heures d'entrée et de sortie pour le jour $d$.
La durée de travail journalière est $\Delta t^{(d)} = t_{out}^{(d)} - t_{in}^{(d)}$.

Nous définissons la caractéristique "Temps de travail moyen" :
$$ x_{\text{mean\_time}} = \frac{1}{|D_{worked}|} \sum_{d \in D_{worked}} \Delta t^{(d)} $$
où $D_{worked}$ est l'ensemble des jours où l'employé était présent.

### 2.2 Variables Catégorielles ($X_{cat}$)
Les variables qualitatives (ex: `Department` $\in$ {Sales, R&D, HR}) sont encodées.

Pour une variable catégorielle $C$ avec $K$ modalités uniques $\{m_1, ..., m_K\}$ :
*   **One-Hot Encoding :** Création de $K$ variables binaires.
    $x_{C=m_k} = 1$ si la catégorie est $m_k$, sinon $0$.

## 3. Le Modèle : Gradient Boosting (XGBoost / Random Forest)

Nous privilégions les méthodes d'ensemble basées sur les arbres de décision.

### 3.1 Arbre de Décision (CART)
Un arbre partitionne l'espace des caractéristiques en régions rectangulaires $R_j$.
La prédiction pour une entrée $x$ est une constante $w_j$ si $x \in R_j$.

$$ h(x) = \sum_{j=1}^J w_j \mathbb{I}(x \in R_j) $$

### 3.2 Ensemble Learning
Le modèle final est une somme pondérée de $M$ arbres.

$$ f_M(x) = \sum_{m=1}^M \gamma_m h_m(x) $$

Dans le cas du **Gradient Boosting**, chaque nouvel arbre $h_m$ est entraîné pour prédire le résidu (l'erreur) des arbres précédents $f_{m-1}(x)$, en minimisant une fonction de perte différentiable $L$.

## 4. Fonction de Coût et Optimisation

Pour une classification binaire, nous minimisons la **Log-Loss** (Entropie Croisée Binaire) :

$$ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

Le modèle cherche $\theta^*$ tel que :
$$ \theta^* = \arg \min_{\theta} \sum_{i=1}^N L(y_i, f(x_i; \theta)) + \Omega(\theta) $$
où $\Omega(\theta)$ est un terme de régularisation pour éviter le sur-apprentissage (contrôle de la complexité des arbres).

## 5. Gestion du Déséquilibre (Imbalanced Classes)

Étant donné que $P(Y=1) \approx 0.15$, le modèle tendrait naturellement à toujours prédire $0$.
Pour contrer cela, nous introduisons un poids $w_1$ pour la classe positive dans la fonction de perte :

$$ L_{weighted}(y, \hat{y}) = - \left[ w_1 \cdot y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right] $$

Généralement, $w_1 \approx \frac{N}{2 \cdot N_{positives}}$.

## 6. Interprétabilité : Valeurs de SHAP (Shapley Additive exPlanations)

Pour expliquer une prédiction individuelle $f(x)$, nous utilisons les valeurs de Shapley issues de la théorie des jeux coopératifs.
La prédiction est décomposée comme la somme de l'effet de chaque caractéristique :

$$ f(x) = \phi_0 + \sum_{j=1}^d \phi_j(x) $$

*   $\phi_0$ est la prédiction moyenne du modèle.
*   $\phi_j(x)$ est la contribution de la caractéristique $j$ à la déviation par rapport à la moyenne.

Cela permet de répondre à la question utilisateur : *"Quels facteurs affectent le plus la prédiction ?"* en classant les $|\phi_j(x)|$ par ordre décroissant.
