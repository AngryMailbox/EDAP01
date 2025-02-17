from models import *
from view_control.Localizer import Localizer
from models.StateModel import StateModel

# Testing the models, e.g., for an 4x8 grid

states = StateModel(4, 8)
loc = Localizer(states, 1)
tMat = loc.get_transition_model()
sVecs = loc.get_observation_model()
tMat.plot_T()
sVecs.plot_o_diags()
print(sVecs.get_o_reading(0))
print(sVecs.get_o_reading(None))

print(loc.update())
