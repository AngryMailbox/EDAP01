import numpy as np
from Filters import HMMFilter
from models.StateModel import StateModel
from models.TransitionModel import TransitionModel
from models.ObservationModel_NUF import ObservationModelNUF
from models.ObservationModel_UF import ObservationModelUF
from models.RobotSim import RobotSim
import matplotlib.pyplot as plt


def ManhattanDistance(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    distance = dx + dy
    return distance


def displayManhattanDistanceOverTime(data, title):
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Time")
    plt.yticks([0, 1, 2, 3, 4, 5])
    plt.ylim(0, 5)
    plt.ylabel("Average Manhattan Distance")
    plt.show()


REPS = 500
ROWS = 10
COLS = 10


def main():
    manhattanData1 = []  # scenario 1: FF
    manhattanData2 = []  # scenario 2: FF with Smooth

    cumulativeErrorFilter = 0
    cumulativeErrorSmooth = 0

    # Inits
    SM = StateModel(ROWS, COLS)
    TM = TransitionModel(SM)
    NUF = ObservationModelNUF(SM)

    # Fill the probs with equal probabilities for each state
    numStates = SM.get_num_of_states()
    probsFilter = np.ones(numStates) / numStates
    probsSmooth = np.ones(numStates) / numStates

    truePos = 0
    FILTER = HMMFilter(probsFilter, TM, NUF, SM)
    SMOOTH = HMMFilter(probsSmooth, TM, NUF, SM)

    ROBOT = RobotSim(truePos, SM)

    lastFive = []

    for step in range(1, REPS, + 1):
        truePos = ROBOT.move_once(TM)
        sensorPos = ROBOT.sense_in_current_state(NUF)
        probsFilter = FILTER.fwd_filter(sensorPos)

        lastFive.append([truePos, sensorPos])
        truePos, sensorPos = lastFive[0]
        lastFive = lastFive[1:]

        probsSmooth = SMOOTH.fwd_bwd_smoothing(lastFive, sensorPos)

        filterPositions = probsFilter.copy()
        smoothPositions = probsSmooth.copy()

        state = 0
        while state < SM.get_num_of_states():
            smoothPositions[state:state + 4] = sum(smoothPositions[state:state + 4])
            state += 4

        estimateFilter = SM.state_to_position(int(np.argmax(filterPositions)))
        estimateSmooth = SM.state_to_position(int(np.argmax(smoothPositions)))

        tsX, tsY = SM.state_to_position(truePos)
        eX1, eY1 = estimateFilter
        eX2, eY2 = estimateSmooth

        error1 = ManhattanDistance(tsX, tsY, eX1, eY1)
        error2 = ManhattanDistance(tsX, tsY, eX2, eY2)

        cumulativeErrorFilter += error1
        cumulativeErrorSmooth += error2

        avgErrorFilter = cumulativeErrorFilter / (step + 1)
        avgErrorSmooth = cumulativeErrorSmooth / (step + 1)

        manhattanData1.append(avgErrorFilter)
        manhattanData2.append(avgErrorSmooth)

    print(f"Step {step+1}/{REPS}:")
    print("Average Error for Filter:", avgErrorFilter)
    print("Average Error for Smooth:", avgErrorSmooth)

    displayManhattanDistanceOverTime(manhattanData1, "Avg Manhattan Distance (Forward Filtering)")
    displayManhattanDistanceOverTime(manhattanData2, "Avg Manhattan Distance (Forward Filtering+Smoothing)")


if __name__ == "__main__":
    main()
