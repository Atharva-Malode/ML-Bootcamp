import numpy as np 
import pandas as pd 
import os
import math
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from ipywidgets import interact
from google.colab import output
output.enable_custom_widget_manager()
from mpl_toolkits.mplot3d import Axes3D

def visualize_data(X, Y):
    plt.close()
    # we used the 'marker' and 'c' parameters
    plt.scatter(X,Y, marker='x', c='r') 

    # Set the title
    plt.title("Data Visualization")
    plt.show()


def plot_real_cost_function(X, Y, compute_cost):
    w_range = np.linspace(-2, 4, 100)
    b_range = np.linspace(-4, 2, 100)
    # Create a meshgrid of w and b values
    W, B = np.meshgrid(w_range, b_range)

    # Compute the cost for each combination of w and b using the compute_cost function
    Z = np.zeros_like(W)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            Z[j, i] = compute_cost(X, Y, W[j, i], B[j, i])
    @interact(w=(-2, 4, 0.1), b=(-4, 2, 0.1))
    def plot_cost(w, b):
        plt.close()  # Close any existing figures
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(W, B, Z, cmap='viridis')

        # Set the current values of w and b
        ax.scatter(w, b, compute_cost(X, Y, w, b), color='red', s=50)

        # Set labels and title
        ax.set_xlabel('w')
        ax.set_ylabel('b')
        ax.set_zlabel('Cost')
        ax.set_title('Cost Function')

        plt.show()
    plot_cost(W, B)


def soup_bowl_3D():
    plt.close()
    """ Create figure and plot with a 3D projection"""
    fig = plt.figure(figsize=(8, 8))

    # Plot configuration
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(45, -120)

    # Useful linearspaces to give values to the parameters w and b
    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)

    # Get the z value for a bowl-shaped cost function
    z = np.zeros((len(w), len(b)))
    for i in range(len(w)):
        for j in range(len(b)):
            z[i, j] = w[i]**2 + b[j]**2

    # Meshgrid used for plotting 3D functions
    W, B = np.meshgrid(w, b)

    # Create the 3D surface plot of the bowl-shaped cost function
    ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$J(w,b)$", rotation=90)
    ax.set_title("$J(w,b)$\n [You can rotate this figure]", size=15)

    plt.show()


def soup_bowl_2D():
    plt.close()
    """ Create figure and plot with a 2D contour plot"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Useful linearspaces to give values to the parameters w and b
    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)

    # Get the z value for a bowl-shaped cost function
    z = np.zeros((len(w), len(b)))
    for i in range(len(w)):
        for j in range(len(b)):
            z[i, j] = w[i]**2 + b[j]**2

    # Create the 2D contour plot of the bowl-shaped cost function
    c = ax.contour(w, b, z, cmap="Spectral_r")
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_title("$J(w,b)$")

    # Add the initial w and b points
    w_points = [-10, 0, 10]
    b_points = [-10, 0, 10]
    ax.scatter(w_points, b_points, color='red', s=50)

    # Define the update function for interact
    def update(w1, b1, w2, b2, w3, b3):
        w_points = [w1, w2, w3]
        b_points = [b1, b2, b3]

        # Update the scatter plot data
        ax.cla()
        c = ax.contour(w, b, z, cmap="Spectral_r")
        ax.scatter(w_points, b_points, color='red', s=50)

        # Update the plot labels
        ax.set_xlabel("$w$")
        ax.set_ylabel("$b$")
        ax.set_title("$J(w,b)$")

        fig.canvas.draw()

    # Use interact to dynamically update the plot
    interact(update, w1=(-20, 20, 1), b1=(-20, 20, 1), w2=(-20, 20, 1), b2=(-20, 20, 1), w3=(-20, 20, 1), b3=(-20, 20, 1))

    plt.show()


def soup_bowl_3D_interactive():
    plt.close()
    """ Create figure and plot with a 3D projection"""
    fig = plt.figure(figsize=(8, 8))

    # Plot configuration
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(45, -120)

    # Useful linearspaces to give values to the parameters w and b
    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)

    # Get the z value for a bowl-shaped cost function
    z = np.zeros((len(w), len(b)))
    for i in range(len(w)):
        for j in range(len(b)):
            z[i, j] = w[i]**2 + b[j]**2

    # Meshgrid used for plotting 3D functions
    W, B = np.meshgrid(w, b)

    # Create the 3D surface plot of the bowl-shaped cost function
    ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$J(w,b)$", rotation=90)
    ax.set_title("$J(w,b)$\n [You can rotate this figure]", size=15)

    # Add the initial w and b points
    w_points = [-10, 0, 10]
    b_points = [-10, 0, 10]
    j_points = [w_point**2 + b_point**2 for w_point, b_point in zip(w_points, b_points)]
    points = ax.scatter(w_points, b_points, j_points, color='red', s=50)

    # Define the update function for interact
    def update(w1, b1, w2, b2, w3, b3):
        w_points = [w1, w2, w3]
        b_points = [b1, b2, b3]
        j_points = [w_point**2 + b_point**2 for w_point, b_point in zip(w_points, b_points)]

        # Update the scatter plot data
        points._offsets3d = (w_points, b_points, j_points)

        # Update the lines
        ax.cla()
        ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
        ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
        ax.scatter(w_points, b_points, j_points, color='red', s=50)
        for i in range(len(w_points) - 1):
            ax.plot([w_points[i], w_points[i+1]], [b_points[i], b_points[i+1]], [j_points[i], j_points[i+1]], color='red')

        # Update the plot labels
        ax.set_xlabel("$w$")
        ax.set_ylabel("$b$")
        ax.set_zlabel("$J(w,b)$", rotation=90)
        ax.set_title("$J(w,b)$\n [You can rotate this figure]", size=15)

        fig.canvas.draw()

    # Use interact to dynamically update the plot
    interact(update, w1=(-20, 20, 1), b1=(-20, 20, 1), w2=(-20, 20, 1), b2=(-20, 20, 1), w3=(-20, 20, 1), b3=(-20, 20, 1))

    plt.show()


"""
  Compute gradient
"""
def visualize_gradient(x, y, w, b, compute_cost, compute_gradient):
    plt.close()
    """
    Visualizes the gradient computed by the compute_gradient() function.
    
    Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): Target values
        w, b (scalar): Model parameters
    """
    # Compute the gradient
    dw, db = compute_gradient(x, y, w, b)
    
    # Create a grid of w and b values
    w_vals = np.linspace(-1, 3, 100)
    b_vals = np.linspace(-1, 3, 100)
    W, B = np.meshgrid(w_vals, b_vals)
    
    # Compute the cost for each combination of w and b
    cost_vals = np.zeros_like(W)
    for i in range(len(w_vals)):
        for j in range(len(b_vals)):
            cost_vals[i, j] = compute_cost(x, y, W[i, j], B[i, j])
    
    # Plot the cost surface and gradient vector
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W, B, cost_vals, cmap='viridis', alpha=0.8)
    ax.quiver(w, b, compute_cost(x, y, w, b), -dw, -db, 0, color='red', label='Gradient')
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Cost')
    ax.set_title('Gradient Visualization')
    ax.legend()
    plt.show()


def visualize_gradient_2D(x, y, w, b, compute_cost, compute_gradient):
    plt.close()
    """
    Visualizes the gradient computed by the compute_gradient() function.
    
    Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): Target values
        w, b (scalar): Model parameters
    """
    # Compute the gradient
    dw, db = compute_gradient(x, y, w, b)
    
    # Create a grid of w and b values
    w_vals = np.linspace(-1, 3, 100)
    b_vals = np.linspace(-1, 3, 100)
    W, B = np.meshgrid(w_vals, b_vals)
    
    # Compute the cost for each combination of w and b
    cost_vals = np.zeros_like(W)
    for i in range(len(w_vals)):
        for j in range(len(b_vals)):
            cost_vals[i, j] = compute_cost(x, y, W[i, j], B[i, j])
    
    # Plot the 2D cost contour plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contour(W, B, cost_vals, levels=20, cmap='viridis')
    ax.quiver(w, b, -dw, -db, color='red', angles='xy', scale_units='xy', scale=1, label='Gradient')
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('Gradient Visualization')
    ax.legend()
    plt.show()

def plot_regression_line(X, Y, model):
    plt.close()
    # Plot scatter plot
    plt.scatter(X, Y, color='b', label='Data Points')

    # Plot line traced by linear regression
    plt.plot(X, model.predict(X), color='r', label='Linear Regression')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()