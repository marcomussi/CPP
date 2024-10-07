import numpy as np

from regressors import GaussianProcessRegressor

class PricingAgentIndep: 

    
    def __init__(self, kernel_L, sigma_process, input_dim, horizon, actions):

        self.regressor = GaussianProcessRegressor(kernel_L, sigma_process, input_dim)
        self.input_dim = input_dim
        self.horizon = horizon
        self.actions = actions
        self.num_actions = self.actions.shape[0]
        self.beta = 2 * np.log(self.horizon)
        self.reset()
        

    def reset(self):
        
        self.no_data = True
        self.regressor.reset()

    
    def pull(self):

        if not self.no_data:
            mu, sigma = self.regressor.compute(self.actions.reshape(self.num_actions, 1))
            self.last_action = np.argmax(self.actions * (mu + self.beta * sigma))
        else:
            self.last_action = np.random.choice(self.num_actions)
        
        return self.last_action
            
    
    def update(self, reward):
        
        self.regressor.update_data(self.actions[self.last_action], reward)
        self.no_data = False