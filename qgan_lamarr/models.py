from typing import Callable, List
import numpy as np
from qiskit import QuantumCircuit, transpile
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
                 seed: int = 42, 
                 qmio: bool = False) -> None:
        
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
        self.qmio = qmio
        

        # Simulation parameters
        if self.qmio:
            from qmio import QmioRuntimeService
            self._sampler = QmioRuntimeService()
        else :
            self._sampler = StatevectorSampler()
        self._nshots = 2**10

        # Training parameters
        self.discriminator_lr = 1e-3
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
        
        if self.qmio:
            pub = qc_gen.assign_parameters(weights_gen, inplace = True)
            pub = transpile(pub, self._sampler)
            with self._sampler.backend(name='qpu') as bk:
                job = bk.run(pub, shots = self._nshots)
                counts = job.result().get_counts()

        else :
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
        self._discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr)
        
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



'''
------------------------------------------------------------------------
    Hybrid quantum Condicional Generative Adversarial Network (QCGAN)
------------------------------------------------------------------------

'''


class CondGenerator1D:
    def __init__(self,
                 num_qubits: int
                 ) -> None :
        self.num_qubits = num_qubits
        self.ansatz_cnt = 0
        self.input_cnt = 0
        self.schedule = {}

    def add_ansatz_layer(self, qc : QuantumCircuit) -> None:
        if qc.num_qubits != self.num_qubits:
            raise ValueError('Ansatz qubit number not matching the circuit')
        self.schedule.update({f'G_{self.ansatz_cnt}' : qc}); self.ansatz_cnt += 1

    def add_input_layer(self, xmap : list | None = None) -> None:
        if xmap is None:
            _xmap = self.prepare_xmap()
        else :
            _xmap = xmap
        self.schedule.update({f'X_{self.input_cnt}' : _xmap}); self.input_cnt += 1
    
    def prepare_xmap(self):
        xmap = []
        bins = [format(b, f'0{int(self.num_qubits)}b') for b in range(2**self.num_qubits)]
        for xclass in bins:
            xqc = QuantumCircuit(self.num_qubits)
            for xbit in range(self.num_qubits):
                if xclass[self.num_qubits-1-xbit] == '1':
                    xqc.x(xbit)
            xmap.append(xqc)   
        return xmap 
    
    def summary(self) -> None:
        print('____________________________')
        print('')
        print('Generator model summary')
        print('')
        for key in self.schedule.keys():
            print('____________________________')
            print('')
            if 'G_' in key:
                print('Generator layer: '+key+f' | num_param : {self.schedule[key].num_parameters}')
            if 'X_' in key:
                print('Input layer: '+key+f' ')
            print('')
        print('____________________________')

class QCGAN(QGAN):

    def __init__(self,
                 num_qubits: int,
                 generator: CondGenerator1D,
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
        
        self.num_parameters = 0
        for _key in self._generator.schedule:
            if 'G_' in _key:
                self.num_parameters = self.num_parameters + self._generator.schedule[_key].num_parameters

        self._num_classes = num_classes
        if class_weights is None:
            self._class_weights = np.ones(num_classes) / num_classes
        else:
            w = np.array(class_weights, dtype=float)
            self._class_weights = w / w.sum()

        for c in range(self._num_classes):
            self.metrics.update({f'jensen_shannon_c{c}' : [],
                                 f'jensen_shannon_avg_c{c}' : [],
                                 f'fidelity_c{c}': [],
                                 f'fidelity_avg_c{c}' : []})
    '''
    Build generator functions
    '''        
    def build_generator(self, _class: int):
        qc_g = QuantumCircuit(self._num_qubits)
        for _key in self._generator.schedule:
            if 'X_' in _key:
                q_input = self._generator.schedule[_key][_class].copy()
                qc_g = qc_g.compose(q_input)
            elif 'G_' in _key:
                qc_ansatz = self._generator.schedule[_key]
                qc_g = qc_g.compose(qc_ansatz)
        qc_g.measure_all()
        return qc_g 
        
    '''
    Sampling (generator & real distribution)
    '''
    def cond_generator_eval(self, _class: int, weights_gen: np.array) -> dict:
        '''
        Samples the generator circuit for the input weights, returns counts from measurement
        '''
        qc_gen = self.build_generator(_class)
        pub = (qc_gen, weights_gen)
        job = self._sampler.run([pub], shots = self._nshots)
        counts = job.result()[0].data.meas.get_counts()
        return counts
           
    def cond_real_dist_eval(self, _class) -> dict:
        '''
        Samples the real distribution 
        '''
        return self._real_dist(_class, self._nshots, self._dim)   
        
    '''
    Discriminator
    ''' 
    def cond_discriminator_forward(self, _class, sample: dict, training: bool = True):
        '''
        Discriminator forward pass
        '''
        x = dict2vector(sample, self._bins)   
        x_norm = _class / max(self._num_classes - 1, 1)
        x_cond = np.append(x, x_norm).astype(np.float32)      
        x_tensor = tf.convert_to_tensor(x_cond[None, :], dtype = tf.float32)
        return self._discriminator(x_tensor, training=training)


    '''
    Conditional loss functions
    '''
    def cond_generator_loss(self, _class, weights_gen: np.ndarray) -> float:

        fake_sample = self.cond_generator_eval(_class, np.array(weights_gen))
        d_fake = self.cond_discriminator_forward(_class, fake_sample, training = False)
        
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
        
    def cond_discriminator_loss(self, _class, weights_gen: np.ndarray) -> tf.Tensor:

        if self.wass:
            '''
            Wasserstein loss with Gradient Penalty: D(G(z)) - D(x) + lambda * GP
            '''
            fake_sample = self.cond_generator_eval(_class, weights_gen)
            real_sample = self.cond_real_dist_eval(_class)
            
            x_norm = _class / max(self._num_classes - 1, 1)
            x_fake = tf.convert_to_tensor(np.append(dict2vector(fake_sample, self._bins), x_norm)[None, :].astype(np.float32))
            x_real = tf.convert_to_tensor(np.append(dict2vector(real_sample, self._bins), x_norm)[None, :].astype(np.float32))
            
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
            fake_sample = self.cond_generator_eval(_class, weights_gen)
            real_sample = self.cond_real_dist_eval(_class)
            
            d_fake =  self.cond_discriminator_forward(_class, fake_sample, training = True)
            d_real = self.cond_discriminator_forward(_class, real_sample, training = True)
            loss = - 0.5 * (tf.math.log(1.0 - d_fake + 1e-12) + tf.math.log(d_real + 1e-12))
            return tf.reduce_mean(loss)
        
    '''
    Total loss functions
    '''
    def total_generator_loss(self, weights_gen: np.ndarray) -> float:
        tot_gen_loss = 0.0
        for _c in range(self._num_classes):
            tot_gen_loss = tot_gen_loss + self._class_weights[_c] * self.cond_generator_loss(_c, weights_gen)
        return tot_gen_loss

    def total_discriminator_loss(self, weights_gen: np.ndarray) -> tf.Tensor:
        tot_dis_loss = 0.0
        for _c in range(self._num_classes):
            tot_dis_loss = tot_dis_loss + self._class_weights[_c] * self.cond_discriminator_loss(_c, weights_gen)
        return tot_dis_loss
    
    '''
    Training
    '''
    def cond_train_discriminator(self, weights_gen: np.ndarray) -> float:
        '''
        One gradient update step for the discriminator
        '''
        with tf.GradientTape() as tape:
            d_loss = self.total_discriminator_loss(weights_gen)
            
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
                weights_gen = np.zeros(self.num_parameters)
        self.baseline_js = self.cond_compute_baseline_js(n_samples = 50)

        if manager:
            metadata = {'epochs': epochs, 
                        'shots': self._nshots,
                        'baseline_js': self.baseline_js,
                        'bins': self._dim,
                        'num_classes': self._num_classes,
                        'tranning_balance': step_balance,
                        'discriminator_lr': self.discriminator_lr,
                        'initial_weights': weights_gen,
                        'optimizer': opt,
                        'wasserstein': self.wass,
                        **opt_args}
            self.FileManager = FileManager(self._generator, self._discriminator, metadata)

        optimizer = QGAN_optimizer(name = opt, **opt_args)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr)
        
        '''
        Training schedule
        '''
        print("Training started")
        for stp in tqdm(range(epochs)):
                
            # Train discriminator
            for _ in range(int(np.ceil(step_balance))):
                d_loss_val = self.cond_train_discriminator(weights_gen)
            self.discriminator_losses.append(d_loss_val)
                    
            # Train generator
            for _ in range(int(np.ceil(1/step_balance))):
                weights_gen, g_loss_val = optimizer.step(self.total_generator_loss, weights_gen)
            self.generator_losses.append(g_loss_val)
                
            # Callback
            if self._callback:
                self._callback(weights_gen, g_loss_val)
    
            # Metrics
            for c in range(self._num_classes):
                current_sample = self.cond_get_sample(c, self._nshots, weights_gen = weights_gen)
                real_dist_sample = self.cond_real_dist_eval(c)
                self.cond_compute_metrics(c, current_sample, real_dist_sample, stp)
            self.total_compute_metrics(stp)

            if manager:
                _metrics = {key: self.metrics[key][stp] for key in self.metrics.keys()}
                self.manage(stp, weights_gen, g_loss_val, d_loss_val, _metrics)
        
        self._trained_generator_weights = weights_gen  

        
        print("Training completed")   

    '''
    Model evaluation and sampler   
    '''
    def cond_get_sample(self, _class, nsamples, weights_gen = None):
        if weights_gen is None:
            weights_gen = self._trained_generator_weights
            
        original_shots = self._nshots
        self._nshots = nsamples
        counts = self.cond_generator_eval(_class, weights_gen)
        self._nshots = original_shots
        return counts
        
    '''
    Metrics
    '''
    def cond_compute_metrics(self, _class, sample1, sample2, stp):
        js = metrics.jensen_shannon(sample1, sample2, self._bins)
        fid = metrics.fidelity(sample1, sample2, self._bins)
        
        self.metrics[f'jensen_shannon_c{_class}'].append(js)
        self.metrics[f'jensen_shannon_avg_c{_class}'].append(metrics.metric_avg(stp, self.metrics[f'jensen_shannon_c{_class}']))
        self.metrics[f'fidelity_c{_class}'].append(fid)
        self.metrics[f'fidelity_avg_c{_class}'].append(metrics.metric_avg(stp, self.metrics[f'fidelity_c{_class}']))
    
    def total_compute_metrics(self, stp):
        js = np.sum([self._class_weights[c] * self.metrics[f'jensen_shannon_c{c}'][stp] for c in range(self._num_classes)])
        fid = np.sum([self._class_weights[c] * self.metrics[f'fidelity_c{c}'][stp] for c in range(self._num_classes)])

        self.metrics[f'jensen_shannon'].append(js)
        self.metrics[f'jensen_shannon_avg'].append(metrics.metric_avg(stp, self.metrics[f'jensen_shannon']))
        self.metrics[f'fidelity'].append(fid)
        self.metrics[f'fidelity_avg'].append(metrics.metric_avg(stp, self.metrics[f'fidelity']))

    def cond_compute_baseline_js(self, n_samples: int = 10) -> float:
        '''
        Average JS divergence between two independent samples of the real distribution (sampling noise floor).
        '''
        baseline_values = []
        for _ in range(n_samples):
            total_value = 0.0
            for c in range(self._num_classes):
                sample_a = self.cond_real_dist_eval(c)
                sample_b = self.cond_real_dist_eval(c)
            
                vec_a = dict2vector(sample_a, self._bins)
                vec_b = dict2vector(sample_b, self._bins)
                total_value = total_value + self._class_weights[c] * jensenshannon(vec_a, vec_b)
            baseline_values.append(total_value)
            
        return float(np.mean(baseline_values)), float(np.std(baseline_values)) 

    def manage(self, _step, _params, _gen_loss, _dis_loss, _metrics):
        self.FileManager.update_param(_params)
        self.FileManager.update_losses(_step, _gen_loss, _dis_loss)
        self.FileManager.update_metrics(_step, _metrics)        





'''
-----------------------
    Old X-Map QCGAN
-----------------------
'''    
class XMapQCGAN(QGAN):
    '''
    ## Quantum Conditional GAN - XGate map encoding

    Before the trainable ansatz, X-gates are applied to the state initialization, depending on some input real value. Being X a random continous variable, the model should return the conditioned distribution P(Y|X). The enconding consist of binning the distribution of X, associating each bin with a state from the computational basis of the circuit. The ansatz should create an entanglement map that brings each input state to the corresponding output distribution.
    '''

    def __init__(self,
                 num_qubits: int,
                 generator: QuantumCircuit,
                 discriminator: tf.keras.Model,
                 real_dist: Callable[[int, int], dict],
                 num_classes: int,
                 class_weights: list | None = None,
                 xmap: list | None = None,
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

        for c in range(self._num_classes):
            self.metrics.update({f'jensen_shannon_c{c}' : [],
                                 f'jensen_shannon_avg_c{c}' : [],
                                 f'fidelity_c{c}': [],
                                 f'fidelity_avg_c{c}' : []})
        if xmap is None:
            self.prepare_xmap()
        else :
            self.xmap = xmap
        
        

    '''
    Sampling (generator & real distribution)
    '''
    def cond_generator_eval(self, _class: int, weights_gen: np.array) -> dict:
        '''
        Samples the generator circuit for the input weights, returns counts from measurement
        '''
        qc_input = self.xmap[_class].copy()
        qc_ansatz = self._generator.copy()
        qc_gen = qc_input.compose(qc_ansatz, range(self._num_qubits))
        qc_gen.measure_all()
        pub = (qc_gen, weights_gen)
        job = self._sampler.run([pub], shots = self._nshots)
        counts = job.result()[0].data.meas.get_counts()
        return counts
           
    def cond_real_dist_eval(self, _class) -> dict:
        '''
        Samples the real distribution 
        '''
        return self._real_dist(_class, self._nshots, self._dim)   
        
    '''
    Discriminator
    ''' 
    def cond_discriminator_forward(self, _class, sample: dict, training: bool = True):
        '''
        Discriminator forward pass
        '''
        x = dict2vector(sample, self._bins)   
        x_norm = _class / max(self._num_classes - 1, 1)
        x_cond = np.append(x, x_norm).astype(np.float32)      
        x_tensor = tf.convert_to_tensor(x_cond[None, :], dtype = tf.float32)
        return self._discriminator(x_tensor, training=training)


    '''
    Conditional loss functions
    '''
    def cond_generator_loss(self, _class, weights_gen: np.ndarray) -> float:

        fake_sample = self.cond_generator_eval(_class, np.array(weights_gen))
        d_fake = self.cond_discriminator_forward(_class, fake_sample, training = False)
        
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
        
    def cond_discriminator_loss(self, _class, weights_gen: np.ndarray) -> tf.Tensor:

        if self.wass:
            '''
            Wasserstein loss with Gradient Penalty: D(G(z)) - D(x) + lambda * GP
            '''
            fake_sample = self.cond_generator_eval(_class, weights_gen)
            real_sample = self.cond_real_dist_eval(_class)
            
            x_norm = _class / max(self._num_classes - 1, 1)
            x_fake = tf.convert_to_tensor(np.append(dict2vector(fake_sample, self._bins), x_norm)[None, :].astype(np.float32))
            x_real = tf.convert_to_tensor(np.append(dict2vector(real_sample, self._bins), x_norm)[None, :].astype(np.float32))
            
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
            fake_sample = self.cond_generator_eval(_class, weights_gen)
            real_sample = self.cond_real_dist_eval(_class)
            
            d_fake =  self.cond_discriminator_forward(_class, fake_sample, training = True)
            d_real = self.cond_discriminator_forward(_class, real_sample, training = True)
            loss = - 0.5 * (tf.math.log(1.0 - d_fake + 1e-12) + tf.math.log(d_real + 1e-12))
            return tf.reduce_mean(loss)
        
    '''
    Total loss functions
    '''
    def total_generator_loss(self, weights_gen: np.ndarray) -> float:
        tot_gen_loss = 0.0
        for _c in range(self._num_classes):
            tot_gen_loss = tot_gen_loss + self._class_weights[_c] * self.cond_generator_loss(_c, weights_gen)
        return tot_gen_loss

    def total_discriminator_loss(self, weights_gen: np.ndarray) -> tf.Tensor:
        tot_dis_loss = 0.0
        for _c in range(self._num_classes):
            tot_dis_loss = tot_dis_loss + self._class_weights[_c] * self.cond_discriminator_loss(_c, weights_gen)
        return tot_dis_loss
    
    '''
    Training
    '''
    def cond_train_discriminator(self, weights_gen: np.ndarray) -> float:
        '''
        One gradient update step for the discriminator
        '''
        with tf.GradientTape() as tape:
            d_loss = self.total_discriminator_loss(weights_gen)
            
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
                weights_gen = np.zeros(self._generator.num_parameters)
        self.baseline_js = self.cond_compute_baseline_js(n_samples = 50)

        if manager:
            metadata = {'epochs': epochs, 
                        'shots': self._nshots,
                        'baseline_js': self.baseline_js,
                        'bins': self._dim,
                        'num_classes': self._num_classes,
                        'tranning_balance': step_balance,
                        'discriminator_lr': self.discriminator_lr,
                        'initial_weights': weights_gen,
                        'optimizer': opt,
                        'wasserstein': self.wass,
                        **opt_args}
            self.FileManager = FileManager(self._generator, self._discriminator, metadata)
            self.FileManager.save_xmap(self.xmap)

        optimizer = QGAN_optimizer(name = opt, **opt_args)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr)
        
        '''
        Training schedule
        '''
        print("Training started")
        for stp in tqdm(range(epochs)):
                
            # Train discriminator
            for _ in range(int(np.ceil(step_balance))):
                d_loss_val = self.cond_train_discriminator(weights_gen)
            self.discriminator_losses.append(d_loss_val)
                    
            # Train generator
            for _ in range(int(np.ceil(1/step_balance))):
                weights_gen, g_loss_val = optimizer.step(self.total_generator_loss, weights_gen)
            self.generator_losses.append(g_loss_val)
                
            # Callback
            if self._callback:
                self._callback(weights_gen, g_loss_val)
    
            # Metrics
            for c in range(self._num_classes):
                current_sample = self.cond_get_sample(c, self._nshots, weights_gen = weights_gen)
                real_dist_sample = self.cond_real_dist_eval(c)
                self.cond_compute_metrics(c, current_sample, real_dist_sample, stp)
            self.total_compute_metrics(stp)

            if manager:
                _metrics = {key: self.metrics[key][stp] for key in self.metrics.keys()}
                self.manage(stp, weights_gen, g_loss_val, d_loss_val, _metrics)
        
        self._trained_generator_weights = weights_gen  

        
        print("Training completed")   

    '''
    Model evaluation and sampler   
    '''
    def cond_get_sample(self, _class, nsamples, weights_gen = None):
        if weights_gen is None:
            weights_gen = self._trained_generator_weights
            
        original_shots = self._nshots
        self._nshots = nsamples
        counts = self.cond_generator_eval(_class, weights_gen)
        self._nshots = original_shots
        return counts
        
    '''
    Metrics
    '''
    def cond_compute_metrics(self, _class, sample1, sample2, stp):
        js = metrics.jensen_shannon(sample1, sample2, self._bins)
        fid = metrics.fidelity(sample1, sample2, self._bins)
        
        self.metrics[f'jensen_shannon_c{_class}'].append(js)
        self.metrics[f'jensen_shannon_avg_c{_class}'].append(metrics.metric_avg(stp, self.metrics[f'jensen_shannon_c{_class}']))
        self.metrics[f'fidelity_c{_class}'].append(fid)
        self.metrics[f'fidelity_avg_c{_class}'].append(metrics.metric_avg(stp, self.metrics[f'fidelity_c{_class}']))
    
    def total_compute_metrics(self, stp):
        js = np.sum([self._class_weights[c] * self.metrics[f'jensen_shannon_c{c}'][stp] for c in range(self._num_classes)])
        fid = np.sum([self._class_weights[c] * self.metrics[f'fidelity_c{c}'][stp] for c in range(self._num_classes)])

        self.metrics[f'jensen_shannon'].append(js)
        self.metrics[f'jensen_shannon_avg'].append(metrics.metric_avg(stp, self.metrics[f'jensen_shannon']))
        self.metrics[f'fidelity'].append(fid)
        self.metrics[f'fidelity_avg'].append(metrics.metric_avg(stp, self.metrics[f'fidelity']))

    def cond_compute_baseline_js(self, n_samples: int = 10) -> float:
        '''
        Average JS divergence between two independent samples of the real distribution (sampling noise floor).
        '''
        baseline_values = []
        for _ in range(n_samples):
            total_value = 0.0
            for c in range(self._num_classes):
                sample_a = self.cond_real_dist_eval(c)
                sample_b = self.cond_real_dist_eval(c)
            
                vec_a = dict2vector(sample_a, self._bins)
                vec_b = dict2vector(sample_b, self._bins)
                total_value = total_value + self._class_weights[c] * jensenshannon(vec_a, vec_b)
            baseline_values.append(total_value)
            
        return float(np.mean(baseline_values)), float(np.std(baseline_values)) 
    

    # ---------------------------------------------------------------------------------------------------------
    # Holevo bound (TESTING)
    # ---------------------------------------------------------------------------------------------------------
    def compute_cl_holevo_bound(self):
        chi_indv = 0
        for c in range(self._num_classes):
            chi_indv = chi_indv + self._class_weights[c] * self.compute_indv_shannon_entropy_real(c)
        chi_glob = self.compute_glob_shannon_entropy_real()
        return chi_glob - chi_indv
            
    def compute_indv_shannon_entropy_real(self, _c):
        samp = self.cond_real_dist_eval(_c)
        vec = dict2vector(samp, self._bins)
        return np.sum([-vec[_x]*np.log(vec[_x]+1e-12) for _x in range(self._dim)])
    
    def compute_glob_shannon_entropy_real(self):
        glob_vec = np.zeros(self._dim)
        for c in range(self._num_classes):
            samp = self.cond_real_dist_eval(c)
            vec = dict2vector(samp, self._bins)
            glob_vec = glob_vec + self._class_weights[c] * vec
        return np.sum([-glob_vec[_x]*np.log(glob_vec[_x]+1e-12) for _x in range(self._dim)])
    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------



    def prepare_xmap(self):
        self.xmap = []
        for xclass in self._bins:
            xqc = QuantumCircuit(self._num_qubits)
            for xbit in range(self._num_qubits):
                if xclass[self._num_qubits-1-xbit] == '1':
                    xqc.x(xbit)
            self.xmap.append(xqc)

    def manage(self, _step, _params, _gen_loss, _dis_loss, _metrics):
        self.FileManager.update_param(_params)
        self.FileManager.update_losses(_step, _gen_loss, _dis_loss)
        self.FileManager.update_metrics(_step, _metrics)        





class SandwichQCGAN(XMapQCGAN):
    '''
    ## Quantum Conditional GAN — Sandwich encoding
 
    Circuit structure per class:
 
        ansatz_A(θ_A)  ──  xmap[class]  ──  ansatz_B(θ_B)  ──  measure
 
    Both parameter vectors θ_A and θ_B are concatenated into a single flat
    array and trained jointly, exactly as in XMapQCGAN.  Everything else
    (discriminator, losses, metrics, training loop, file manager) is inherited
    unchanged.
 
    Parameters
    ----------
    generator      : parametrized QuantumCircuit used as the *first* ansatz (A).
    generator_post : parametrized QuantumCircuit used as the *second* ansatz (B).
                     Must have the same number of qubits as ``generator``.
                     Its parameters must be named differently from those of
                     ``generator`` (use a distinct ParameterVector name, e.g. "φ").
    All other parameters are identical to XMapQCGAN.
 
    Notes
    -----
    ``self._generator`` stores ansatz A; ``self._generator_post`` stores ansatz B.
    ``self._generator.num_parameters + self._generator_post.num_parameters``
    determines the total parameter count seen by the optimizer.
    The weight vector passed to every loss/eval function follows the layout
    ``[θ_A (len n_A) | θ_B (len n_B)]``.
    '''
 
    def __init__(self,
                 num_qubits: int,
                 generator: QuantumCircuit,
                 generator_post: QuantumCircuit,
                 discriminator: tf.keras.Model,
                 real_dist: Callable[[int, int], dict],
                 num_classes: int,
                 class_weights: list | None = None,
                 xmap: list | None = None,
                 wass: bool = False,
                 callback: Callable | None = None,
                 seed: int = 42) -> None:
 
        if generator_post.num_qubits != num_qubits:
            raise ValueError(
                f"generator_post has {generator_post.num_qubits} qubits "
                f"but num_qubits={num_qubits}.")
 
        # Check parameter names don't collide
        names_A = {p.name for p in generator.parameters}
        names_B = {p.name for p in generator_post.parameters}
        overlap = names_A & names_B
        if overlap:
            raise ValueError(
                f"generator and generator_post share parameter names: {overlap}. "
                "Use distinct ParameterVector names (e.g. 'θ' and 'φ').")
 
        super().__init__(num_qubits=num_qubits,
                         generator=generator,
                         discriminator=discriminator,
                         real_dist=real_dist,
                         num_classes=num_classes,
                         class_weights=class_weights,
                         xmap=xmap,
                         wass=wass,
                         callback=callback,
                         seed=seed)
 
        self._generator_post = generator_post
        self._n_params_A = generator.num_parameters
        self._n_params_B = generator_post.num_parameters
 
    # ── Circuit assembly ──────────────────────────────────────────────────────
 
    def _split_weights(self, weights_gen: np.ndarray):
        '''Split the flat parameter vector into (θ_A, θ_B).'''
        return weights_gen[:self._n_params_A], weights_gen[self._n_params_A:]
 
    def cond_generator_eval(self, _class: int, weights_gen: np.ndarray) -> dict:
        '''
        Build and sample the sandwich circuit for the given class:
            ansatz_A(θ_A) | xmap[class] | ansatz_B(θ_B) | measure
        '''
        theta_A, theta_B = self._split_weights(weights_gen)
 
        qc_A    = self._generator.copy()
        qc_xmap = self.xmap[_class].copy()
        qc_B    = self._generator_post.copy()
 
        qc_gen = qc_A.compose(qc_xmap, range(self._num_qubits))
        qc_gen = qc_gen.compose(qc_B,   range(self._num_qubits))
        qc_gen.measure_all()
 
        # Bind both parameter sets; Qiskit accepts a flat list ordered by
        # circuit.parameters (which is a ParameterView sorted by name).
        # We build an explicit dict to be unambiguous.
        param_dict = {**dict(zip(self._generator.parameters,      theta_A)),
                      **dict(zip(self._generator_post.parameters, theta_B))}
 
        job    = self._sampler.run([(qc_gen, param_dict)], shots=self._nshots)
        counts = job.result()[0].data.meas.get_counts()
        return counts
 
    # ── fit: only num_parameters changes vs XMapQCGAN ────────────────────────
 
    def fit(self,
            epochs: int = 100,
            step_balance: float = 1.0,
            shots: int | None = None,
            initial_weights: np.ndarray | None = None,
            manager: bool | None = None,
            opt: str | None = 'ADAM_PSR',
            **opt_args):
        '''
        Train the SandwichQCGAN.
 
        The optimizer sees a single flat parameter vector of length
        n_params_A + n_params_B.  See XMapQCGAN.fit() for all other details.
        '''
        if shots is not None:
            self._nshots = shots
 
        self._discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr)
 
        total_params = self._n_params_A + self._n_params_B
 
        if initial_weights is not None:
            weights_gen = initial_weights
        else:
            if self._trained_generator_weights is not None:
                weights_gen = self._trained_generator_weights
            else:
                weights_gen = np.zeros(total_params)
 
        if len(weights_gen) != total_params:
            raise ValueError(
                f"initial_weights has length {len(weights_gen)} but "
                f"n_params_A + n_params_B = {total_params}.")
 
        self.baseline_js = self.cond_compute_baseline_js(n_samples=50)
 
        if manager:
            metadata = {
                'epochs': epochs,
                'shots': self._nshots,
                'baseline_js': self.baseline_js,
                'bins': self._dim,
                'num_classes': self._num_classes,
                'tranning_balance': step_balance,
                'discriminator_lr': self.discriminator_lr,
                'initial_weights': weights_gen,
                'optimizer': opt,
                'wasserstein': self.wass,
                'n_params_A': self._n_params_A,
                'n_params_B': self._n_params_B,
                **opt_args,
            }
            self.FileManager = FileManager(self._generator, self._discriminator, metadata)
            self.FileManager.save_xmap(self.xmap)
 
        optimizer = QGAN_optimizer(name=opt, **opt_args)
 
        print("Training started")
        for stp in tqdm(range(epochs)):
 
            for _ in range(max(1, int(step_balance))):
                d_loss_val = self.cond_train_discriminator(weights_gen)
            self.discriminator_losses.append(d_loss_val)
 
            weights_gen, g_loss_val = optimizer.step(self.total_generator_loss, weights_gen)
            self.generator_losses.append(g_loss_val)
 
            if self._callback:
                self._callback(weights_gen, g_loss_val)
 
            for c in range(self._num_classes):
                current_sample   = self.cond_get_sample(c, self._nshots, weights_gen=weights_gen)
                real_dist_sample = self.cond_real_dist_eval(c)
                self.cond_compute_metrics(c, current_sample, real_dist_sample, stp)
            self.total_compute_metrics(stp)
 
            if manager:
                _metrics = {key: self.metrics[key][stp] for key in self.metrics.keys()}
                self.manage(stp, weights_gen, g_loss_val, d_loss_val, _metrics)
 
        self._trained_generator_weights = weights_gen
        print("Training completed")
