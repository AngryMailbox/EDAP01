import numpy as np


#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, TM, OM, SM):
        self.TM = TM
        self.OM = OM
        self.SM = SM
        self.probs = probs
    
    def fwd_filter(self, reading):
        #this is just that formula that is in the lecture slide
        #i have divided it up so that it is easier to read
        TeTranspose = self.TM.get_T_transp()
        Oread = self.OM.get_o_reading(reading).diagonal()
        self.probs = np.matmul(TeTranspose, self.probs)
        self.probs = np.multiply(self.probs, Oread)
        alpha = 1 / sum(self.probs)
        self.probs = np.multiply(alpha, self.probs)
        return self.probs

    def fwd_bwd_smoothing(self, lastFive, reading):
        if len(lastFive) < 5:
            return self.fwd_filter(reading)

        numStates = self.SM.get_num_of_states()
        T = self.TM.get_T()
        probs = self.probs

        #fwd
        fwd = np.zeros((5, numStates))
        fwd[0] = probs * self.OM.get_o_reading(lastFive[0][1])


        for t in range(1, 5):
            observe = self.OM.get_o_reading(lastFive[t][1])
            fwd[t] = np.dot(fwd[t-1], T) * observe

        #bwd
        bwd = np.ones((5, numStates))
        for t in range(3, -1, -1):
            observe = self.OM.get_o_reading(lastFive[t+1][1])
            bwd[t] = np.dot(T, bwd[t+1] * observe)

        smoothed = fwd * bwd
        smoothed /= np.sum(smoothed, axis=1, keepdims=True)

        return smoothed[-1]
        