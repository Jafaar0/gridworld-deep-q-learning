# Projet de résolution du problème de grid world par deep-q learning

Le but de ce court projet était de mettre en oeuvre les notions de q learning vues dans mon cours de fondamentaux d'IA, en essayant d'aller plus loin en remplaçant la fonction q classique par un réseau de neurone qui apprendrait au fur et à mesure.

Le projet est inspiré dans sa structure par [cet exemple](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) tutoriel de reinforcement learning de la documentation de pytorch, et dans l'idée par mon cours de fondamentaux d'IA par M. Althoff Matthias.

## Description
Un gridworld est un monde en 2 dimensions composé de cases "vide", de cases obstacles, d'un point de départ et d'un point d'arrivée. L'agent part du point de départ et chaque action a un fait coût en utilité, le point d'arrivée possède une grande récompense en utilité qui va diriger l'apprentissage.

La fonction Q, ici remplacée par un réseau de neurones, est la fonction qui à chaque couple (état, action) associe une valeur. On définit donc la politique de notre modèle en choisissant à chaque étape l'action avec la plus haute valeur de Q. Lors de la phase d'entraînement, on autorise cependant l'agent à choisir des actions qui dérogent à la politique (epsilon-greedy), avec une probabilité décroissante pour explorer l'environnement.

## Prérequis
- Python 3.10+
- PyTorch (1.13+) : `pip install torch`

## Exécution
- lancer l'entraînement et affichage :
  - `python main.py`

### Paramètres manuels (à changer dans le code)
- Soit GridWorld(dimensions, départ, arrivée, "fixed", [liste des obstacles]) comme ceci : `GridWorld((5,5), (0,0), (4,4), "fixed", [(1,1),(2,2),(3,3)])`
- Soit GridWorld(dimensions, départ, arrivée, "random", nb_obstacles=n) comme ceci : `GridWorld((5,5), (0,0), (4,4), "random", nb_obstacles=3)`
- modifier `num_episodes` dans `train.py` selon GPU/CPU, et `max_steps` dans `train.py` aussi selon la taille de l'environnement.

## Structure du projet
- `main.py` : définition de l'environnement `GridWorld`, affichage, et point d'entrée.
- `DQN.py` : définition du modèle `deep_q_network` (réseau entièrement connecté).
- `train.py` : boucle DQN, mémoire de replay, sélection d'action epsilon-greedy, optimisation, mise à jour target network.

## Résultat attendu
- pendant l'entraînement : affichage `Episode x/n, Reward y, Epsilon z`
- après entraînement : sauvegarde / affichage des valeurs Q pour l'état courant et la politique par flèches.

## Améliorations possibles
- gérer explicitement collisions et bordures dans `move` (actuellement ce sont des cases accessibles avec une grosse pénalité d'utilité).
- marquer épisode terminé en cas de rencontre d'obstacle.
- ajouter tests unitaires pour `GridWorld` et `optimize_model`.

## Remarque
Du fait de la simplicité de l'environnement, un algorithme de q-learning classique fonctionnerai très bien, même mieux dans de nombreux cas, que ce modèle. Ce projet avait simplement pour ambition d'appliquer des connaissances théoriques de mes cours.
