# In this cell, you can write your own "main" to run and evaluate your
# implementation without using the visualisation above (should be considerably faster!)

import random
import numpy as np
from sympy import true
from Filters import HMMFilter
from models.ObservationModel_UF import ObservationModelUF
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


def main():
    # init innit
    FWD = True
    BWD = True
    NUF = True
    nbr_correct1 = 0
    nbr_correct2 = 0
    avg_error1 = 0
    avg_error2 = 0
    total_error1 = 0
    total_error2 = 0
    error1 = 0
    error2 = 0
    error_array1 = []
    error_array2 = []
    last_five = []

    state_model = StateModel(10, 10)
    transition_model = TransitionModel(state_model)
    if NUF:
        observation_model1 = ObservationModelNUF(state_model)
    else:
        observation_model1 = ObservationModelUF(state_model)

    observation_model2 = ObservationModelUF(state_model)

    nbr_of_states = state_model.get_num_of_states()
    probs1 = np.ones(nbr_of_states) / nbr_of_states
    probs2 = np.ones(nbr_of_states) / nbr_of_states
    true_pos = 0
    sensor_pos1 = None
    sensor_pos2 = None
    estimate1 = state_model.state_to_position(np.argmax(probs1))
    estimate2 = state_model.state_to_position(np.argmax(probs2))

    HMMF1 = HMMFilter(probs1, transition_model, observation_model1, state_model)
    HMMF2 = HMMFilter(probs2, transition_model, observation_model1, state_model)
    robot_sim = RobotSim(true_pos, state_model)

    if BWD:
        last_five = []
        for _ in range(5):
            true_pos_smoothing = robot_sim.move_once(transition_model)
            sensor_pos1 = robot_sim.sense_in_current_state(observation_model1)
            sensor_pos2 = robot_sim.sense_in_current_state(observation_model2)
            last_five.append([true_pos_smoothing, sensor_pos1])

    for m in range(500):

        # if m < 5:
        #     for _ in range(5):
        #         true_pos = robot_sim.move_once(transition_model)
        #         sensor_pos1 = robot_sim.sense_in_current_state(observation_model1)
        #         last_five.append([true_pos, sensor_pos1])

        true_pos = robot_sim.move_once(transition_model)
        sensor_pos1 = robot_sim.sense_in_current_state(observation_model1)
        probs1 = HMMF1.fwd_filter(sensor_pos1)

        last_five.append([true_pos, sensor_pos1])
        true_pos_smoothing, sensor_pos1 = last_five[0]
        last_five = last_five[1:]
        probs2 = HMMF2.fwd_bwd_smoothing(last_five, sensor_pos1)
        # if FWD:
        #     # true_pos = robot_sim.move_once(transition_model)
        #     # sensor_pos1 = robot_sim.sense_in_current_state(observation_model1)
        #     #sensor_pos2 = robot_sim.sense_in_current_state(observation_model2)
        #     probs1 = HMMF1.filter(sensor_pos1)
        #     #probs2 = HMMF1.filter(sensor_pos2)
        # if BWD:
        #     # true_pos = robot_sim.move_once(transition_model)
        #     # sensor_pos1 = robot_sim.sense_in_current_state(observation_model1)
        #     # sensor_pos2 = robot_sim.sense_in_current_state(observation_model2)

        #     last_five.append([true_pos, sensor_pos1])
        #     true_pos, sensor_pos1 = last_five[0]
        #     last_five = last_five[1:]

        #     probs2 = HMMF1.smoothing(last_five, sensor_pos1)

        fPositions1 = probs1.copy()
        fPositions2 = probs2.copy()

        for state in range(0, state_model.get_num_of_states(), 4):
            fPositions1[state : state + 4] = sum(fPositions1[state : state + 4])
            fPositions2[state : state + 4] = sum(fPositions2[state : state + 4])

        estimate1 = state_model.state_to_position(np.argmax(fPositions1))
        estimate2 = state_model.state_to_position(np.argmax(fPositions2))

        tsX1, tsY1 = state_model.state_to_position(true_pos)
        tsX2, tsY2 = state_model.state_to_position(true_pos_smoothing)
        eX1, eY1 = estimate1
        eX2, eY2 = estimate2

        if eX1 == tsX1 and eY1 == tsY1:
            nbr_correct1 += 1

        if eX2 == tsX2 and eY2 == tsY2:
            nbr_correct2 += 1

        error1 = abs(tsX1 - eX1) + abs(tsY1 - eY1)
        error2 = abs(tsX2 - eX2) + abs(tsY2 - eY2)
        total_error1 += error1
        total_error2 += error2
        error_array1.append(error1)
        error_array2.append(error2)

        avg_error1 = total_error1 / m
        avg_error2 = total_error2 / m

        # ret = False
        # srX = -1
        # srY = -1
        # if sensor_pos1 != None:
        #     srX, srY = state_model.reading_to_position(sensor_pos1)
        #     ret = True
        # sensor_error = abs(tsX-srX) + abs(tsY-srY)
        # #error_array2.append(sensor_error)

        print("Move:", m + 1)
        # print("Estimated Pos:", "X:", eX1, "Y:", eY1)
        # print("True Pos:", "X:", tsX, "Y:", tsY)
        print("Correct Estimations 1:", nbr_correct1)
        print("Manhattan Distance 1:", error1)
        print("Average Error 1 :", avg_error1)

        print("Correct Estimations 2:", nbr_correct2)
        print("Manhattan Distance 2:", error2)
        print("Average Error 2 :", avg_error2)

    plt.plot(error_array2, "b--")
    plt.plot(error_array1, "r--")

    plt.show()


main()
