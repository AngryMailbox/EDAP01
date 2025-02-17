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
    plt.ylabel("Expected Manhattan Distance")
    plt.show()


REPS = 500
ROWS = 4
COLS = 4


def main():
    manhattanData1 = []  # scenario 1: FF with NUF
    manhattanData2 = []  # scenario 2: FF with UF

    correctGuessNUF = 0
    correctGuessUF = 0
    cumulativeErrorNUF = 0
    cumulativeErrorUF = 0

    SM = StateModel(ROWS, COLS)
    TM = TransitionModel(SM)
    NUF = ObservationModelNUF(SM)
    UF = ObservationModelUF(SM)

    numStates = SM.get_num_of_states()
    probsNUF = np.ones(numStates) / numStates
    probsUF = np.ones(numStates) / numStates

    truePos = 0
    FILTERNUF = HMMFilter(probsNUF, TM, NUF, SM)
    FILTERUF = HMMFilter(probsUF, TM, UF, SM)
    ROBOT = RobotSim(truePos, SM)

    for step in range(1, REPS + 1):
        truePos = ROBOT.move_once(TM)
        tsX, tsY = SM.state_to_position(truePos)

        sensorPosNUF = ROBOT.sense_in_current_state(NUF)
        sensorPosUF = ROBOT.sense_in_current_state(UF)
        probsNUF = FILTERNUF.fwd_filter(sensorPosNUF)
        probsUF = FILTERUF.fwd_filter(sensorPosUF)

        fPositionsNUF = probsNUF.copy()
        fPositionsUF = probsUF.copy()

        estimateFilterNUF = SM.state_to_position(int(np.argmax(fPositionsNUF)))
        estimateFilterUF = SM.state_to_position(int(np.argmax(fPositionsUF)))

        eX1, eY1 = estimateFilterNUF
        eX2, eY2 = estimateFilterUF

        if eX1 == tsX and eY1 == tsY:
            correctGuessNUF += 1
        if eX2 == tsX and eY2 == tsY:
            correctGuessUF += 1

        errorNUF = ManhattanDistance(tsX, tsY, eX1, eY1)
        errorUF = ManhattanDistance(tsX, tsY, eX2, eY2)

        cumulativeErrorNUF += errorNUF
        cumulativeErrorUF += errorUF

        avgErrorNUF = cumulativeErrorNUF / step
        avgErrorUF = cumulativeErrorUF / step

        manhattanData1.append(avgErrorNUF)
        manhattanData2.append(avgErrorUF)

    displayManhattanDistanceOverTime(manhattanData1, "Avg Manhattan Distance (NUF)")
    displayManhattanDistanceOverTime(manhattanData2, "Avg Manhattan Distance (UF)")


if __name__ == "__main__":
    main()
