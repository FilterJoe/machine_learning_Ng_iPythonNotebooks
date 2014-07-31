import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, ndimage
from IPython.display import display, clear_output


def display_data(X):
    # Clear and close all figures
    clear_output()  #iPython Notebook command
    plt.clf()

    m, n = X.shape
    width = round(np.sqrt(n))

    cm = plt.cm.get_cmap('Greys_r')
    fig, ax_array = plt.subplots(10, 10, figsize=(8, 8))
    plt.subplots_adjust(left=.12, bottom=None, right=None, top=None, wspace=0, hspace=0)

    for i in range(10):
        for j in range(10):
            ax_array[i, j].axis('off')
            rotated_image = ndimage.rotate(X[i*10 + j, :].reshape(width, width), 90)
            ax_array[i, j].imshow(rotated_image, cmap=cm, origin='lower')

def sigmoid(z):
    '''
    Return sigmoid(z)
    Args:
        Z (scalar or array)
    '''
    return 1.0 / (1.0 + np.exp(-z))

def h_of_theta(theta, X):
    if theta.ndim == 1:
        transposed_theta = theta[:, np.newaxis]
    return sigmoid(X.dot(transposed_theta))

def cost_function_reg(theta, X, y, lamda):
    '''
    Return cost of using theta for logistic regression
    Args:
        theta (1D numpy array): scipy optimization requires theta parameter to be first, and passed as 1D array
        X (m x n numpy array): first column is 1 (intercept), any number of columns to follow
        y (m x 1 numpy array): each number represents the answer to the corresponding row in X
        lamda (float): regularization parameter purposely misspelled to avoid conflict with python's lambda keyword
    '''
    m = len(y) * 1.0
    h_theta = h_of_theta(theta, X)

    # change values of 0 or 1 slightly to avoid log(0)
    tol = .00000000000000000000001  # tolerance must be no higher than .00000000000000000000001 to match octave results
    h_theta[h_theta < tol] = tol  # close-to-zero values get set to tol
    h_theta[(h_theta < 1 + tol) & (h_theta > 1 - tol)] = 1 - tol  # close-to-1 values get set to 1 - tol

    regularization_term = (float(lamda)/2) * theta**2

    cost_vector = y * np.log(h_theta) + (-y + 1) * np.log(-h_theta + 1)

    J = -sum(cost_vector)/m + sum(regularization_term[1:])/m

    return J[0]


def gradient_reg(theta, X, y, lamda):
    '''
    Return gradient of theta for logistic regression
    Args:
        theta (1D numpy array): theta is first parameter and a 1D array because scipy optimization requires it
        X (m x n numpy array): first column is intercept, any number of columns to follow
        y (m x 1 numpy array): each number represents the answer to the corresponding row in X
        lamda (float): regularization parameter purposely misspelled to avoid conflict with python's lambda keyword
    '''
    m = len(y)
    h_theta = h_of_theta(theta, X)
    derivative_regularization_term = float(lamda) * theta/m
    grad = (h_theta - y).T.dot(X)/m + derivative_regularization_term.T
    grad[0][0] -= derivative_regularization_term[0]  # theta_0 is the only term that does not get regularized, so back out
    return np.ndarray.flatten(grad)  # scipy optimization requires gradient to be a 1D array

def cost_and_gradient(theta, X, y, lamda):
    J = cost_function_reg(theta, X, y, lamda)
    grad = gradient_reg(theta, X, y, lamda)
    return J, grad

def predict(Theta1, Theta2, X):
    '''
    Return a 1D array of ints between 1 and 10, which is
    the digit predicted by neural network (Theta1, Theta2) for each image in X.
    Each int represents 1 predicted digit (10 means 0) for a corresponding training example (row) from X.
    Args:
        Theta1 (2D numpy array): parameters for connections between input and hidden layers of neural network
        Theta2 (2D numpy array): parameters for connections between hidden and output layers of neural network
        X (m x n numpy array): each row represents a 400 bit (20 x 20) training example. X has no bias unit (1).
    '''
    m = X.shape[0]
    p = np.zeros(m)

    for i in range(m):
        x_row = np.insert(X[i, :], 0, 1.0, axis=0)
        a2 = sigmoid(Theta1.dot(x_row))

        a2_row = np.insert(a2, 0, 1.0, axis=0)
        a3 = sigmoid(Theta2.dot(a2_row))

        p[i] = np.argmax(a3) + 1  # octave data assumes index starts at 1
    return p
