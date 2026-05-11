from typing import Callable, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from keras import Model
import tensorflow as tf
from .optimize import QGAN_optimizer
from .manager import FileManager
from tqdm import tqdm 
from .tools import dict2vector
from . import metrics
from scipy.spatial.distance import jensenshannon


class QGAN():
    '''
    ## Hybrid Quantum Generative Adversarial Networks

    Hybrid model composed of two competing agents:
        - Quantum generator: quantum variational circuit trained to generates samples indistinguishable from a real ditribution.
        - Classical discriminator: classical DNN trained to discriminate whether a samples comes from the real distribution or was created by the generator.
    The training this model consist of an alternate update of both netwworks, creating a competition game between them that would likely end in a Nash equilibrium, when the discriminator is uncapable of detecting the generated samples from the real, guessing wrong half of the time.

    Args
    ----
        - num_qubits    : number of qubits for the generator (and discriminator) circuits. Corresponds to log2(Nb), where Nb is the number of bins for the objective distribution.
            
        - generator : parametrized circuit used for the generator's QNN.
        
        - discriminator : classical NN model for the discriminator.

        - real_dist : function that returns a sample of the real distribution.

        - wass  : selection to use the Wasserstein's losses for the QGAN.
        
        - callback  : reference to a user's callback function that has two parameters (objective_weights, objective_value) and returns 'None'. The callback is 
            invoked in each iteration of the optimization and has access to intermediate data during training.
    '''
    def __init__(self,
                 num_qubits: int,
                 generator: QuantumCircuit,
                 discriminator: tf.keras.Model,
                 real_dist: Callable[[int, int], dict],
                 wass: bool = False,
                 callback: Callable[[np.ndarray, float], None] | None = None,
                 seed: int = 42) -> None:
        
        if num_qubits is None or generator is None or discriminator is None or real_dist is None:
            raise ValueError("num_qubits, generator, discriminator and real_dist must be provided.")

        # Network parameters
        self._num_qubits = num_qubits
        self._dim = 2 ** num_qubits
        self._bins = [format(b, f'0{int(num_qubits)}b') for b in range(2**num_qubits)]
        self._generator = generator
        self._discriminator = discriminator
        self._real_dist = real_dist
        self.wass = wass
        self._callback = callback
        

        # Simulation parameters
        self._sampler = StatevectorSampler()
        self._nshots = 2**10

        # Training parameters
        self.discriminator_lr = 1e-3
        self._discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr) # 1e-4
        self.generator_losses: List[float] = []
        self.discriminator_losses: List[float] = []
        self.metrics = {#'kullback_leibler' : [], 
                        #'generator_entropy' : [], 
                        #'kolmogorov_smirnov' : [], 
                        # 'wasserstein_distance' : [],
                        # 'chi2_test' : [],
                        'jensen_shannon' : [],
                        'jensen_shannon_avg' : [],
                        'fidelity': [],
                        'fidelity_avg' : []}
        self._trained_generator_weights = None 
        
    '''
    Sampling (generator & real distribution)
    '''
    def generator_eval(self, weights_gen: np.array) -> dict:
        '''
        Samples the generator circuit for the input weights, returns counts from measurement
        '''
        qc_gen = self._generator.copy()
        qc_gen.measure_all()
        pub = (qc_gen, weights_gen)
        job = self._sampler.run([pub], shots = self._nshots)
        counts = job.result()[0].data.meas.get_counts()
        return counts
           
    def real_dist_eval(self) -> dict:
        '''
        Samples the real distribution 
        '''
        return self._real_dist(self._nshots, self._dim)   
        
    '''
    Discriminator
    ''' 
    def discriminator_forward(self, sample: dict, training: bool = True):
        '''
        Discriminator forward pass
        '''
        x = dict2vector(sample, self._bins)
        x_tensor = tf.convert_to_tensor(x[None, :])
        return self._discriminator(x_tensor, training = training)

    '''
    Loss functions
    '''
    def generator_loss(self, weights_gen: np.ndarray) -> float:

        fake_sample = self.generator_eval(np.array(weights_gen))
        d_fake = self.discriminator_forward(fake_sample, training = False)
        
        if self.wass:
            '''
            Wasserstein loss: -D(G(z)) 
            '''
            loss = -tf.reduce_mean(d_fake)
        else:
            '''
            Non-saturating loss: -log(D(G(z))
            '''
            loss = -tf.math.log(d_fake + 1e-12)
            
        return np.array(loss.numpy().squeeze(), dtype = np.float64)
        
    def discriminator_loss(self, weights_gen: np.ndarray) -> tf.Tensor:

        if self.wass:
            '''
            Wasserstein loss with Gradient Penalty: D(G(z)) - D(x) + lambda * GP
            '''
            fake_sample = self.generator_eval(weights_gen)
            real_sample = self.real_dist_eval()
            
            x_fake = tf.convert_to_tensor(dict2vector(fake_sample, self._bins)[None, :])
            x_real = tf.convert_to_tensor(dict2vector(real_sample, self._bins)[None, :])
            
            d_fake = self._discriminator(x_fake, training = True)
            d_real = self._discriminator(x_real, training = True)
            w_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            
            alpha = tf.random.uniform([1, 1], minval=0., maxval=1.)
            interpolated = alpha * x_real + (1 - alpha) * x_fake
            
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                d_interpolated = self._discriminator(interpolated, training = True)
            grads = gp_tape.gradient(d_interpolated, interpolated)
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            lambda_gp = 10.0 
            
            return w_loss + lambda_gp * gp

        else :
            '''
            GAN loss: - [ log(D(x)) + log(1 - D(G(z))) ]
            '''
            fake_sample = self.generator_eval(weights_gen)
            real_sample = self.real_dist_eval()
            
            d_fake =  self.discriminator_forward(fake_sample, training = True)
            d_real = self.discriminator_forward(real_sample, training = True)
            loss = - 0.5 * (tf.math.log(1.0 - d_fake + 1e-12) + tf.math.log(d_real + 1e-12))
            return tf.reduce_mean(loss)

    '''
    Training
    '''
    def train_discriminator(self, weights_gen: np.ndarray) -> float:
        '''
        One gradient update step for the discriminator
        '''
        with tf.GradientTape() as tape:
            d_loss = self.discriminator_loss(weights_gen)
            
        grads = tape.gradient(d_loss, self._discriminator.trainable_variables)
        self._discriminator_optimizer.apply_gradients(zip(grads, self._discriminator.trainable_variables))
        return float(d_loss.numpy())

    def fit(self, 
            epochs: int = 100, 
            step_balance: float = 1.0,
            shots: int | None = None,
            initial_weights: np.ndarray | None = None,
            manager: bool | None = None,
            opt: str | None = 'ADAM_PSR',
            **opt_args):

        '''
            Training setup
        '''
        if shots is not None:
            self._nshots = shots

        if initial_weights is not None:
            weights_gen = initial_weights
        else :
            if self._trained_generator_weights is not None:
                weights_gen = self._trained_generator_weights
            else :
                weights_gen = np.zeros(self._generator.num_parameters) #np.random.uniform(0, 2*np.pi, self._generator.num_parameters)

        self.baseline_js = self.compute_baseline_js(n_samples = 50)

        if manager:
            metadata = {'epochs': epochs, 
                        'shots': self._nshots,
                        'baseline_js': self.baseline_js,
                        'bins': self._dim,
                        'tranning_balance': step_balance,
                        'discriminator_lr': self.discriminator_lr,
                        'initial_weights': weights_gen,
                        'optimizer': opt,
                        'wasserstein': self.wass,
                        **opt_args}
            self.FileManager = FileManager(self._generator, self._discriminator, metadata)

        optimizer = QGAN_optimizer(name = opt, **opt_args)
        
        '''
        Training schedule
        '''
        print("Training started")
        for stp in tqdm(range(epochs)):
                
            # Train discriminator
            for _ in range(int(np.ceil(step_balance))):
                d_loss_val = self.train_discriminator(weights_gen)
            self.discriminator_losses.append(d_loss_val)
                    
            # Train generator
            for _ in range(int(np.ceil(1/step_balance))):
                weights_gen, g_loss_val = optimizer.step(self.generator_loss, weights_gen)
            self.generator_losses.append(g_loss_val)
                
            # Callback
            if self._callback:
                self._callback(weights_gen, g_loss_val)
        
            # Metrics
            current_sample = self.get_sample(self._nshots, weights_gen = weights_gen)
            real_dist_sample = self.real_dist_eval()
            self.compute_metrics(current_sample, real_dist_sample, stp)

            if manager:
                _metrics = {key: self.metrics[key][stp] for key in self.metrics.keys()}
                self.manage(stp, weights_gen, g_loss_val, d_loss_val, _metrics)
        
        self._trained_generator_weights = weights_gen  

        
        print("Training completed")   

    '''
    Model evaluation and sampler   
    '''
    def get_sample(self, nsamples, weights_gen = None):
        if weights_gen is None:
            weights_gen = self._trained_generator_weights
            
        original_shots = self._nshots
        self._nshots = nsamples
        counts = self.generator_eval(weights_gen)
        self._nshots = original_shots
        return counts
        
    '''
    Metrics
    '''
    def compute_metrics(self, sample1, sample2, stp):
        js = metrics.jensen_shannon(sample1, sample2, self._bins)
        fid = metrics.fidelity(sample1, sample2, self._bins)
        
        self.metrics['jensen_shannon'].append(js)
        self.metrics['jensen_shannon_avg'].append(metrics.metric_avg(stp, self.metrics['jensen_shannon']))
        self.metrics['fidelity'].append(fid)
        self.metrics['fidelity_avg'].append(metrics.metric_avg(stp, self.metrics['fidelity']))
    
    def compute_baseline_js(self, n_samples: int = 10) -> float:
        '''
        Average JS divergence between two independent samples of the real distribution (sampling noise floor).
        '''
        baseline_values = []
        for _ in range(n_samples):
            sample_a = self.real_dist_eval()
            sample_b = self.real_dist_eval()
            
            vec_a = dict2vector(sample_a, self._bins)
            vec_b = dict2vector(sample_b, self._bins)
            baseline_values.append(jensenshannon(vec_a, vec_b))
            
        return float(np.mean(baseline_values)), float(np.std(baseline_values)) 
    
    def manage(self, _step, _params, _gen_loss, _dis_loss, _metrics):
        self.FileManager.update_param(_params)
        self.FileManager.update_losses(_step, _gen_loss, _dis_loss)
        self.FileManager.update_metrics(_step, _metrics)
        
    
class XCQGAN(QGAN):
    '''
    ## Conditional QGAN - XGate based encoding

    Before the trainable ansatz, X-gates are applied to the state initialization, depending on some input real value. Being X a random continous variable, the model should return the conditioned distribution P(Y|X). The enconding consist of binning the distribution of X, associating each bin with a state from the computational basis of the circuit. The ansatz should create an entanglement map that brings each input state to the corresponding output distribution.
    '''

    def __init__(self,
                 num_qubits: int,
                 generator: QuantumCircuit,
                 discriminator: tf.keras.Model,
                 real_dist: Callable[[int, int], dict],
                 num_classes: int,
                 class_weights: list | None = None,
                 wass: bool = False,
                 callback: Callable | None = None,
                 seed: int = 42) -> None:
        
        if num_classes > 2 ** num_qubits:
            raise ValueError(
                f"num_classes ({num_classes}) cannot exceed 2**num_qubits=({2**num_qubits}).")
        
        super().__init__(num_qubits = num_qubits,
                         generator = generator,
                         discriminator = discriminator,
                         real_dist = real_dist,
                         wass = wass,
                         callback = callback,
                         seed = seed,)
 
        self._num_classes = num_classes
        if class_weights is None:
            self._class_weights = np.ones(num_classes) / num_classes
        else:
            w = np.array(class_weights, dtype=float)
            self._class_weights = w / w.sum()

