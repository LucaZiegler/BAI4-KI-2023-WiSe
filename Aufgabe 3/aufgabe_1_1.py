import matplotlib.pyplot as plt
import numpy as np


def loss_function(_w1, _w2, _x1, _x2, _y_tar):
    # Fehlerfunktion E aus Aufgabenzettel (Mittlere quadratische Abweichung/Fehler)
    return (1 / 2) * (np.sin(_w1 * _x1) + np.cos(_w2 * _x2) + _w2 - _y_tar) ** 2


def visualize(_current_w1, _current_w2, _x1, _x2, _y_tar, title):
    _x = np.linspace(
        -10, 10, 100
    )  # Der fuer diese Aufgabe interessante Definitionsbereich liegt zwischen -10 und 10
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(_x, y)

    # X und Y auf Z der abbilden -> Fehlerfunktion wird visualiziert.
    Z = loss_function(X, Y, _x1, _x2, _y_tar)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap="jet", edgecolor="none"
    )  # Fehlerfunktion im 3D Raum darstellen

    _current_loss = loss_function(
        float(_current_w1), float(_current_w2), _x1, _x2, _y_tar
    )

    if title == "Fehlergebirge - Start":
        print("Fehlerwert Start: " + str(round(_current_loss, 1)) + "\r\n")

    ax.plot(
        _current_w1,
        _current_w2,
        _current_loss,
        marker="o",
        markersize=10,
        markeredgecolor="black",
        markerfacecolor="white",
    )  #

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("w1", fontsize=11)
    ax.set_ylabel("w2", fontsize=11)
    ax.set_zlabel("E", fontsize=11)

    plt.show()

    return _current_w1, _current_w2, _current_loss


if __name__ == "__main__":
    # # Datensatz 1
    # w1 = -6.5
    # w2 = -9.5
    # alpha = 0.05

    # Datensatz 2
    w1 = 0.0
    w2 = -0.5
    alpha = 0.05

    w1_start = w1
    w2_start = w2

    # konstant, definiert durch Datensample aus Aufgabe
    x1 = 1.0
    x2 = 1.5
    y_tar = 2.0

    visualize(
        _current_w1=w1,
        _current_w2=w2,
        _x1=x1,
        _x2=x2,
        _y_tar=y_tar,
        title="Fehlergebirge - Start",
    )

    # Schrittweise Näherung des/eines Minimums
    for x in range(1000):
        w1 = w1 - alpha * (
            x1 * np.cos(x1 * w1) * (np.sin(x1 * w1) - y_tar + np.cos(w2 * x2) + w2)
        )
        w2 = w2 - alpha * (np.cos(x2 * w2) + w2 - y_tar + np.sin(w1 * x1)) * (
            1 - x2 * np.sin(x2 * w2)
        )

    print("w1 Start (Vorhersage): " + str(w1_start) + "; w1 Ende: " + str(round(w1, 1)))
    print(
        "w2 Start (Vorhersage): "
        + str(w2_start)
        + "; w2 Ende: "
        + str(round(w2, 1))
        + "\r\n"
    )

    current_w1, current_w2, current_loss = visualize(
        _current_w1=w1,
        _current_w2=w2,
        _x1=x1,
        _x2=x2,
        _y_tar=y_tar,
        title="Fehlergebirge - Ende",
    )
    print(
        "x: "
        + str(round(current_w1, 1))
        + " y: "
        + str(round(current_w2, 1))
        + " z: "
        + str(round(current_loss, 1))
        + " (Fehlerwert)"
    )

# Um schrittweise das Minimum der Funktion zu ermitteln, sind die partiellen Ableitungen
# nach den lernbaren Gewichten notwendig (warum?).
# → Um das lokale Minimum zu ermitteln, müssen wir die Fehlerfunktion entlang der stärksten Steigung nach unten
# 'fahren'. Anhand der ersten Ableitung kann die Steigung ermittelt werden.

# Wie unterscheiden sich Fehlerwert, Gewichtswerte und Vorhersagen?
# → Der Fehlerwert wird kleiner (nähert sich gegen 0)
# → Die Gewichtswerte passen sich an das lokale Minimum an
# → Siehe Console

# → Auffällig ist, dass obwohl das gleiche Datensample genutzt wird, verschiedene lokale Minima gefunden werden
