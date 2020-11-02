import sys
sys.path.append(r"C:\Users\CreeP\OneDrive\Bureau\DeepLearningNato\AnalyseChiffre\bdd")

import gzip
import numpy as np

file_image_train = gzip.open('bdd/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 1000

file_image_train.read(16)
buf = file_image_train.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
vdata = data.reshape(num_images, image_size*image_size, 1)
data = data.reshape(num_images, image_size, image_size, 1)

file_label_train = gzip.open('bdd/train-labels-idx1-ubyte.gz','r')
file_label_train.read(8)
labels = []
for i in range(0,num_images):
    buf = file_label_train.read(1)
    label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    labels.append(label)

file_image_verif = gzip.open('bdd/t10k-images-idx3-ubyte.gz','r')

image_size_verif = 28
num_images_verif = 10000

file_image_verif.read(16)
buf = file_image_verif.read(image_size * image_size * num_images)
data_verif = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
vdata_verif = data.reshape(num_images, image_size*image_size, 1)
data_verif = data.reshape(num_images, image_size, image_size, 1)

file_label_verif = gzip.open('bdd/t10k-labels-idx1-ubyte.gz','r')
file_label_verif.read(8)
labels_verif = []
for i in range(0,num_images_verif):
    buf = file_label_verif.read(1)
    label_v = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    labels_verif.append(label_v)

import numpy as np
import random

num_layers = 3  #3 couches
sizes = [784, 15, 10] #taille de chaque couche

biases = [np.random.randn(y, 1) for y in sizes[1:]]
#génération des biais initiés à une valeur aléatoire suivant une loi gaussienne de moyenne 0 et variance 1
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
#génération des poids initiés à une valeur aléatoire suivant une loi gaussienne de moyenne 0 et variance 1

#On commence par utiliser la fonction sigmoide comment fonction d'activation f(z) = 1/(1+e^-z)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#Retourne la sortie du réseau avec 'a' comme vecteur d'entrée
def feedforward(a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(biases, weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

#Cette fonction permet de redéfinir les labels en tant que vecteur correspondant on nombre de sortie du système,
#Chacune des 10 sorties représente un chiffre possible (0,1,2,3...,9)
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#On commence par l'ensemble d'entrainement
#training_data = np.concatenate((data, labels),0)

training_data = []
training_data_verif = []
for i in range(0,num_images):
    training_data.append((vdata[i],vectorized_result(int(labels[i]))))
    training_data_verif.append((vdata_verif[i],int(labels_verif[i])))

def cost_derivative(output_activations, y):
        "Retourne la derivée partielle du coût par rapport aux activations"
        return (output_activations-y)

def backprop(x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # en sortie : le gradient de la fonction de cout, pour les poids et biais
        # pour chaque couche
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        # on calcul les activations du réseau (notamment pour avoir la dernier couche d'activation)
        activation = x
        activations = [x] # on enregistre toutes les activations couche par couche
        zs = [] # on enregistre tout les valeurs de neurone (sans fonction d'activation) couche par couche
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # on calcul l'erreur delta pour la dernière couche en multipliant sur les composantes la derivée partielle de
        # cout par rapport à l'activation et la derivée d'activation de la dernière couche (formule backpropagation n°1)
        delta = cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # on utilise la retropropagation pour trouver les erreurs des autres couches
        for l in range(2, num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

def update_mini_batch(mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        weights2 = [w-(eta/len(mini_batch))*nw for w, nw in zip(weights, nabla_w)]
        biases2 = [b-(eta/len(mini_batch))*nb for b, nb in zip(biases, nabla_b)]
        return (weights2, biases2)

def evaluate(test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def SGD(training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                weights, biases = update_mini_batch(mini_batch, eta)
            print(biases)
            print("Epoch n°" + str(j) + ", " + str(evaluate(training_data_verif)) + "/" + str(n_test))

SGD(training_data, 30, 10, 3.0, test_data=training_data_verif)