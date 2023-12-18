import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def diff():
    w1, w2 = sp.symbols('w1 w2')
    x1, x2, y_tar = 1.0, 1.5, 2.0

    # Definiere die Fehlerfunktion
    error = 0.5 * (sp.sin(w1 * x1) + sp.cos(w2 * x2) + w2 - y_tar)**2

    # Berechne die partiellen Ableitungen
    derivative_w1 = sp.diff(error, w1)
    derivative_w2 = sp.diff(error, w2)

    print("Partial derivative with respect to w1:", derivative_w1)
    print("Partial derivative with respect to w2:", derivative_w2)


# Definition der Fehlerfunktion
def loss_function(w1, w2, x1=1.0, x2=1.5, y_tar=2.0):
    return 0.5 * (np.sin(w1 * x1) + np.cos(w2 * x2) + w2 - y_tar)**2



# Visualisierung der Fehlerfunktion
def visualize_error_surface(current_w1, current_w2, current_loss):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.7)

    ax.plot(current_w1, current_w2, current_loss, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="white")
    
    ax.set_title("Fehlergebirge", fontsize=13)
    ax.set_xlabel('w1', fontsize=11)
    ax.set_ylabel('w2', fontsize=11)
    ax.set_zlabel('E', fontsize=11)

    plt.show()

diff()

# Beispiel: Initialisierung mit zufälligen Werten
initial_w1 = np.random.uniform(-10, 10) 
initial_w2 = np.random.uniform(-10, 10)
#initial_loss = loss_function(initial_w1, initial_w2)

# Visualisierung der Fehlerfunktion mit initialen Gewichten
#visualize_error_surface(initial_w1, initial_w2, initial_loss)
