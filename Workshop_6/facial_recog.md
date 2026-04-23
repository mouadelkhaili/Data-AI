# Reconnaissance Faciale avec le Réseau de Neurones Profond

Ce guide explique comment adapter les notebooks de ce workshop pour construire un classificateur binaire de reconnaissance faciale : **est la personne** (1) vs **n'est pas la personne** (0).

---

## 1. Collecte des données

Créez deux dossiers d'images :

```
data/
  positive/   ← photos du visage cible (classe 1)
  negative/   ← photos d'autres personnes ou non-visages (classe 0)
```

**Recommandations :**
- Au moins 100–200 images par classe
- Variez les conditions : éclairage, angle, expression, fond
- Recadrez les images autour du visage (avec OpenCV ou un outil en ligne)
- Redimensionnez toutes les images à une taille fixe carrée (ex. 64×64 pixels)

---

## 2. Chargement des données

Remplacez l'appel à `load_data()` dans `Application - Réseau Profond.ipynb` par la fonction suivante :

```python
from PIL import Image
import numpy as np
import os

def load_face_data(pos_dir, neg_dir, img_size=64, test_split=0.2):
    def load_dir(path, label):
        X, Y = [], []
        for f in os.listdir(path):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = Image.open(os.path.join(path, f)).convert('RGB')
            img = img.resize((img_size, img_size))
            X.append(np.array(img))
            Y.append(label)
        return X, Y

    X1, Y1 = load_dir(pos_dir, 1)
    X0, Y0 = load_dir(neg_dir, 0)

    X = np.array(X1 + X0)        # (m, h, w, 3)
    Y = np.array(Y1 + Y0)

    # Mélange aléatoire
    idx = np.random.permutation(len(Y))
    X, Y = X[idx], Y[idx]

    # Séparation train/test
    split = int(len(Y) * (1 - test_split))
    return X[:split], Y[:split].reshape(1, -1), X[split:], Y[split:].reshape(1, -1)

train_x_orig, train_y, test_x_orig, test_y = load_face_data(
    pos_dir='data/positive',
    neg_dir='data/negative',
    img_size=64
)
```

Le reste du pipeline (aplatissement, normalisation) reste identique :

```python
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten  = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255.
test_x  = test_x_flatten  / 255.
```

---

## 3. Architecture du modèle

### Réseau à L couches (recommandé)

Pour des images 64×64, l'entrée est de taille `64 × 64 × 3 = 12 288`. Utilisez la configuration suivante comme point de départ :

```python
layers_dims = [12288, 20, 7, 5, 1]
```

Si vous utilisez des images plus grandes (ex. 128×128), ajustez la première dimension :

```python
num_px = 128
layers_dims = [num_px * num_px * 3, 32, 16, 8, 1]
```

### Réseau à 2 couches (plus simple)

```python
n_x = 12288
n_h = 20
n_y = 1
layers_dims = (n_x, n_h, n_y)
```

---

## 4. Entraînement

Lancez le modèle avec les mêmes fonctions que dans le notebook :

```python
parameters, costs = L_layer_model(
    train_x, train_y,
    layers_dims,
    learning_rate=0.005,
    num_iterations=2500,
    print_cost=True
)
plot_costs(costs)
```

**Conseils :**
- Diminuez le taux d'apprentissage (`0.001`–`0.005`) si le coût diverge
- Augmentez `num_iterations` si le modèle n'a pas convergé
- Si la précision sur l'ensemble d'entraînement est bonne mais pas sur le test → surapprentissage (voir section 5)

---

## 5. Améliorer la robustesse

### Augmentation des données

Pour réduire le surapprentissage avec peu d'images, générez des variations artificielles :

```python
from PIL import Image, ImageEnhance
import random

def augment(img_array):
    img = Image.fromarray(img_array)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    return np.array(img)
```

Appliquez cette fonction sur les images positives avant d'entraîner pour doubler ou tripler le jeu de données.

### Transfer learning (recommandé pour la production)

Pour une reconnaissance robuste (angles variés, occultation, etc.), utilisez un encodeur de visage pré-entraîné qui extrait des embeddings significatifs plutôt que des pixels bruts :

```bash
pip install face-recognition  # basé sur dlib + ResNet
```

```python
import face_recognition

def get_face_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        return None
    return encodings[0]  # vecteur de 128 dimensions
```

Remplacez alors les pixels par ces vecteurs de 128 dimensions comme entrée du classificateur :

```python
layers_dims = [128, 16, 1]
```

---

## 6. Prédiction sur une nouvelle image

```python
def predict_face(image_path, parameters, num_px=64):
    img = Image.open(image_path).convert('RGB').resize((num_px, num_px))
    x = np.array(img).reshape(-1, 1) / 255.
    AL, _ = L_model_forward(x, parameters)
    pred = int(AL[0, 0] > 0.5)
    label = "C'est la personne" if pred == 1 else "Ce n'est pas la personne"
    print(f"Prédiction : {label} (score : {AL[0, 0]:.3f})")
    return pred
```

---

## Récapitulatif des étapes

| Étape | Action |
|-------|--------|
| 1 | Collecter ~100–200 images par classe dans `data/positive/` et `data/negative/` |
| 2 | Remplacer `load_data()` par `load_face_data()` |
| 3 | Adapter `layers_dims` à la taille d'entrée |
| 4 | Entraîner avec `L_layer_model()` |
| 5 | Évaluer avec `predict()` et ajuster hyperparamètres |
| 6 | (Optionnel) Ajouter de l'augmentation ou passer au transfer learning |
