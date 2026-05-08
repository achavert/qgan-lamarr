from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, LeakyReLU
import QGAN, SingleGaussian, RangeBinning

# Qubit scale
num_qubits = 3
nbins = 2**num_qubits

# Real distribution
def sample_dist(_size, _nbins):
    _range = 0.25
    _sample = SingleGaussian(mean = 0.0, sd = 0.1,shots = _size)
    return RangeBinning(_sample, _nbins = _nbins, _range = (-_range, _range))

sample = sample_dist(2**10, nbins)
print(f'Real sample example: {sample}')


# Quantum generator
reps = 2
qc = QuantumCircuit(num_qubits)
qc.h(range(num_qubits))
theta = ParameterVector("θ", length = (reps + 1) * num_qubits * 2)

p = 0
for q in range(num_qubits):
        qc.ry(theta[q], q)
        p += 1
for r in range(reps):

    for q in range(num_qubits):
        qc.cz(q%num_qubits, (q + 1)%num_qubits)
    for q in range(num_qubits):
        qc.ry(theta[p], q)
        p += 1


# Classical discriminator
discriminator = Sequential([
    InputLayer(shape=(nbins,)),
    Dense(50),
    LeakyReLU(),
    Dense(1, activation='linear')])


# Build model
model = QGAN(num_qubits = num_qubits,
             generator = qc,
             discriminator = discriminator,
             real_dist = sample_dist,
             wass = True)


# Fit model
model.discriminator_lr = 1e-4 # discriminator learning-rate
model.fit(epochs = 2000, # number of epochs
          shots = 2**10, # number of sampling shots
          step_balance = 5.0, # dis:gen training balance
          opt = 'ADAM_PSR', # optimizer
          lr = 1e-3, # optimizer parameters
          manager = True)
