import random
import numpy as np
from sympy import true
from Filters import HMMFilter
from models.StateModel import StateModel
from models.TransitionModel import TransitionModel
from models.ObservationModel_NUF import ObservationModelNUF
from IPython.display import display
from models.RobotSim import RobotSim

# In view_control.Dashboard, there is simply the handling of all the thread based visualisation provided,
# no changes needed, but feel free...
from view_control.Dashboard import Dashboard
from view_control.Localizer import Localizer
import matplotlib.pyplot as plt


ROWS = 4
COLS = 4
REPS = 500
sensorType = 0  # 1 for UF, 0 for NUF


def ManhattanDistance(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    ManhattanDistance = dx + dy
    return ManhattanDistance


def main():
    # Create an instance of StateModel, TransitionModel, and ObservationModel
    # You can replace these with your actual model instances
    sm = StateModel(rows=ROWS, cols=COLS)
    tm = TransitionModel(stateModel=sm)
    om = ObservationModelNUF(stateModel=sm)
    correctGuesses = 0
    manhattanData = []

    # Initialize the Localizer with the models and a flag indicating the type of observation model
    # Replace uniformF with your desired value (0 or 1)
    localizer = Localizer(sm, uniformF=0)

    for foo in range(REPS):
        # Call the update method to perform one cycle of the localization process
        # This will update the true state, sense, probabilities, and estimate
        (
            ret,
            trueX,
            trueY,
            trueH,
            sensorX,
            sensorY,
            estimateX,
            estimateY,
            error,
            fPositions,
        ) = localizer.update()
        manhattanData.append(ManhattanDistance(trueX, trueY, estimateX, estimateY))

    # Use the returned values as needed
    print("Returned values:")
    print("Sensor reading not 'nothing':", ret)
    print("True pose (x, y, h):", trueX, trueY, trueH)
    print("Sensor reading position (x, y):", sensorX, sensorY)
    print("Estimated position (x, y):", estimateX, estimateY)
    print("Error:", error)
    print("Probability dist:", fPositions)

    # Print correct guesses and accuracy
    print("Correct guesses: ", correctGuesses)
    print("Accuracy: ", (correctGuesses / len(manhattanData)) * 100, "%")

    # Plot Manhattan distance
    plt.plot(manhattanData)
    plt.title("Manhattan Distance")
    plt.xlabel("Repetition")
    plt.ylabel("Distance")
    plt.axvline(x=correctGuesses, color="r", linestyle="--")
    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
