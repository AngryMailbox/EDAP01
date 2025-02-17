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
ROWS = 8
COLS = 8


def main():
    manhattanData1 = []  # scenario 1: FF with NUF
    manhattanData2 = []  # scenario 2: Sensor output only

    cumulativeErrorNUF = 0
    cumulativeErrorSensor = 0

    SM = StateModel(ROWS, COLS)
    TM = TransitionModel(SM)
    NUF = ObservationModelNUF(SM)

    numStates = SM.get_num_of_states()
    probs = np.ones(numStates) / numStates

    truePos = 0
    FILTER = HMMFilter(probs, TM, NUF, SM)
    ROBOT = RobotSim(truePos, SM)

    for step in range(1, REPS + 1):
        truePos = ROBOT.move_once(TM)
        tsX, tsY = SM.state_to_position(truePos)

        sensorPos = ROBOT.sense_in_current_state(NUF)
        probs = FILTER.fwd_filter(sensorPos)

        fPositions = probs.copy()

        estimateFilter = SM.state_to_position(int(np.argmax(fPositions)))

        if sensorPos is not None:
            estimateSensor = SM.state_to_position(sensorPos)
            eX2, eY2 = estimateSensor
            error2 = ManhattanDistance(tsX, tsY, eX2, eY2)
        else:
            error2 = 1

        cumulativeErrorSensor += error2
        avgErrorSensor = cumulativeErrorSensor / step
        manhattanData2.append(avgErrorSensor)

        eX1, eY1 = estimateFilter
        error1 = ManhattanDistance(tsX, tsY, eX1, eY1)
        cumulativeErrorNUF += error1
        avgErrorFilter = cumulativeErrorNUF / step
        manhattanData1.append(avgErrorFilter)

        print(f"Step {step}:")
        print("Avg Error for Filter 1:", avgErrorFilter)
        print("Avg Error for Sensor:", avgErrorSensor)

    displayManhattanDistanceOverTime(
        manhattanData1, "Avg Manhattan Distance (Forward Filtering with NUF)"
    )
    displayManhattanDistanceOverTime(
        manhattanData2, "Avg Manhattan Distance (Sensor Output Only)"
    )


if __name__ == "__main__":
    main()
