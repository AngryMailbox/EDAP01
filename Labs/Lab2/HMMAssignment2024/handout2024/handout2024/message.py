
import random
import queue
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        self.__f = probs
        
        
    def filter(self, sensorR) :
        obs_Mat = self.__om.get_o_reading(sensorR)
        trans_Transp_mat= self.__tm.get_T_transp()
        self.__f = obs_Mat @ trans_Transp_mat @ self.__f
        self.__f /= np.sum(self.__f)
        
        return self.__f
    
    def smoothing(self,last_five, sensorR):
        self.filter(sensorR)
        nbr_states = self.__sm.get_num_of_states()
        b = np.ones(nbr_states)
        
        for i in range(4,-1,-1):
            o_mat = self.__om.get_o_reading(last_five[i][1])
            t_mat = self.__tm.get_T()
            b = o_mat @ t_mat @ b
        
        s = self.__f * b
        s /= np.sum(s)
            
        return s
        