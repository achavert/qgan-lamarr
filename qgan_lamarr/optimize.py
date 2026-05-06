from scipy.optimize import minimize
from pennylane import SPSAOptimizer
import numpy as np

class QGAN_optimizer():
    def __init__(self, name, **kargs):
        self.name = name
        self.kargs = kargs
        self.optimizer = None
        
        if self.name == 'COBYLA':
            self.optimizer = 'COBYLA_placeholder'

        elif self.name == 'SPSA':
            maxiter = self.kargs.get('maxiter', 1)
            a = self.kargs.get('a', 0.05)
            c = self.kargs.get('c', 0.1)
            alpha = self.kargs.get('alpha', 0.602)
            gamma = self.kargs.get('gamma', 0.101)
            self.optimizer = SPSAOptimizer(maxiter = maxiter, 
                                           a = a,
                                           c = c,
                                           alpha = alpha,
                                           gamma = gamma)
        elif self.name == 'ADAM_PSR':
            maxiter = self.kargs.get('maxiter', 1)
            lr = self.kargs.get('lr', 0.1)
            beta1 = self.kargs.get('beta1', 0.9)
            beta2 = self.kargs.get('beta2', 0.99)
            eps = self.kargs.get('eps', 1e-08)
            self.optimizer = AdamOptimizerPSR(lr = lr,
                                              beta1 = beta1,
                                              beta2 = beta2,
                                              eps = eps)
        
            
    def step(self, _generator_loss, _weights_gen):
        if self.optimizer == None:
            raise Exception('No optimizer selected')
            
        if self.name == 'COBYLA':
            def objective_gen(w):
                g_loss = _generator_loss(w)
                return float(g_loss)
            res = minimize(objective_gen, x0 = _weights_gen, method = 'COBYLA', options = self.kargs)
            return res.x, res.fun

        elif self.name == 'SPSA':
            weights_gen, g_loss_val = self.optimizer.step_and_cost(_generator_loss, _weights_gen)
            return weights_gen, g_loss_val

        elif self.name == 'ADAM_PSR':
            weights_gen, g_loss_val = self.optimizer.step(_generator_loss, _weights_gen)
            return weights_gen, g_loss_val

class AdamOptimizerPSR():
    def __init__(self, 
                 lr = 0.1,
                 beta1 = 0.9,
                 beta2 = 0.99,
                 eps = 1e-08):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.a = None
        self.b = None

    def step(self, _loss_function, _params):
        if self.a is None:
            self.a = np.zeros_like(_params)
            self.b = np.zeros_like(_params)
        self.t += 1    
        grads = parameter_shift_rule(_loss_function, _params)

        self.a = self.beta1 * self.a + (1 - self.beta1) * grads
        self.b = self.beta2 * self.b + (1 - self.beta2) * (grads**2)

        hat_a = self.a / (1 - self.beta1**self.t)
        hat_b = self.b / (1 - self.beta2**self.t)

        new_params = _params - self.lr * hat_a / (np.sqrt(hat_b) + self.eps)

        new_loss = _loss_function(new_params)

        return new_params, new_loss
        
            
        
def parameter_shift_rule(_loss_function, _params, shift = np.pi/2):
    grads = np.zeros_like(_params)

    for i in range(len(_params)):
        shift_vec = np.zeros_like(_params)
        
        shift_vec[i] = shift
        forward = _loss_function(_params + shift_vec)
        backward = _loss_function(_params - shift_vec)

        grads[i] = 0.5 * (forward - backward)
    return grads
        

