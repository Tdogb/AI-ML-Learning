from Experimenting.TestPlane.Controller import ControlFunction
from pyfme.aircrafts import Cessna172
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.wind import NoWind
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment import Environment
from pyfme.utils.trimmer import steady_state_trim
from pyfme.models.state.position import EarthPosition
from pyfme.models import EulerFlatEarth
from pyfme.simulator import Simulation
from Controller import *

aircraft = Cessna172()
atmosphere = ISA1976()
gravity = VerticalConstant()
wind = NoWind()

environment = Environment(atmosphere, gravity, wind)

pos = EarthPosition(x=0, y=0, height=1000)
psi = 0.5  # rad
TAS = 45  # m/s
controls0 = {'delta_elevator': 0, 'delta_aileron': 0, 'delta_rudder': 0, 'delta_t': 0.5}

trimmed_state, trimmed_controls = steady_state_trim(
    aircraft,
    environment,
    pos,
    psi,
    TAS,
    controls0
)

# Environment conditions for the current state:
environment.update(trimmed_state)

# Forces and moments calculation:
forces, moments = aircraft.calculate_forces_and_moments(trimmed_state, environment, controls0)
system = EulerFlatEarth(t0=0, full_state=trimmed_state)
controls = {
    'delta_elevator': 0,
    'delta_aileron': 1,
    'delta_rudder': 2,
    'delta_t': 3
}
sim = Simulation(aircraft, system, environment, controls, ControlFunction())
results = sim.propagate(10)
print(results)