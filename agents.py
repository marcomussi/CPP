import numpy as np

from efficientregressors import GaussianProcessRegressor


class PricingAgentIndependent: 

    
    def __init__(self, kernel_L, sigma_process, horizon, actions):

        self.regressor = GaussianProcessRegressor(kernel_L, sigma_process, 1)
        self.horizon = horizon
        self.actions = actions
        self.num_actions = self.actions.shape[0]
        self.beta = 2 * np.log(self.horizon)
        self.no_data = True

    
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




class PricingAgentComplementaryBROKEN: 

    
    def __init__(self, num_products, actions, costs, kernel_L, sigma_process, horizon, graph_dict=None):

        self.num_products = num_products
        self.num_actions = len(actions)
        
        self.actions = actions
        self.costs = costs
        
        self.kernel_L = kernel_L
        self.sigma_process = sigma_process
        
        self.horizon = horizon
        self.beta = 4 * np.log(self.horizon)
        
        self.interactions_count = 0

        self.total_sales = np.zeros(self.num_products)

        actions0, actions1 = np.meshgrid(self.actions, self.actions)
        self.byvariate_actions = np.hstack((actions0.ravel().reshape(-1, 1), actions1.ravel().reshape(-1, 1)))
        actions0, actions1 = np.meshgrid(np.linspace(0, self.num_actions-1, self.num_actions, dtype=int), 
                                         np.linspace(0, self.num_actions-1, self.num_actions, dtype=int))
        self.byvariate_actions_map = np.hstack((actions0.ravel().reshape(-1, 1), actions1.ravel().reshape(-1, 1)))

        self.graph_dict = graph_dict
        if graph_dict is None:
            self.known_relations = False
            self.interactions_dataset = -1 * np.ones((num_products, horizon, 2))
        else:
            self.known_relations = True
            self.used_graph_dict = graph_dict
            self._create_regressors()

    
    def pull(self):

        if self.interactions_count == 0:
            
            self.last_actions_vect = np.random.choice(self.num_actions, self.num_products)
        
        else:
            
            if self.known_relations:
                
                for key in list(self.used_graph_dict.keys()):
                    
                    if len(self.used_graph_dict[key]) == 0: 
                        
                        mu, sigma = self.regressors[key].compute(self.actions.reshape(self.num_actions, 1))
                        self.last_actions_vect[key] = np.argmax(self.actions * (mu + self.beta * sigma))
                        
                    else: 
                        
                        trainee_lst = self.used_graph_dict[key]
                        aux1 = np.array([self.total_sales[trainee_lst[i]] for i in range(len(trainee_lst))])
                        aux2 = np.array([self.costs[trainee_lst[i]] for i in range(len(trainee_lst))])
                        trainee_w = np.sum(aux1 * aux2)
                        trainer_w = self.total_sales[key] * self.costs[key]
                        
                        mu_y1, mu_y2, sigma = self.regressors[key].compute(self.byvariate_actions)
                        idx = np.argmax(
                            self.byvariate_actions[:, 0] * trainer_w * (mu_y1 + self.beta * sigma) + 
                            self.byvariate_actions[:, 1] * trainee_w * (mu_y2 + self.beta * sigma)
                        )
                        self.last_actions_vect[key] = self.byvariate_actions_map[idx, 0]
                        for i in range(len(trainee_lst)):
                            self.last_actions_vect[trainee_lst[i]] = self.byvariate_actions_map[idx, 1]
            
            else:
                
                raise NotImplementedError("unknown graph scenario not implemented")     
        
        return self.last_actions_vect
            
    
    def update(self, sales, impressions, interactions=None):

        assert sales.shape == (self.num_products, ) and impressions.shape == (self.num_products, ), "update(): error in input"

        self.total_sales = self.total_sales + sales # check correttezza 
        
        rewards = sales / impressions
        
        if self.known_relations:
            
            for key in list(self.used_graph_dict.keys()):
                if len(self.used_graph_dict[key]) == 0: 
                    self.regressors[key].update_data(self.actions[self.last_actions_vect[key]], rewards[key])
                else: 
                    trainee_lst = self.used_graph_dict[key]
                    for i in range(len(trainee_lst)):
                        if i < len(trainee_lst) - 1:
                            assert self.last_actions_vect[trainee_lst[i]] == self.last_actions_vect[trainee_lst[i+1]
                                ], "update(): error, actions must be the same for all trainee"
                    trainee_lst = self.used_graph_dict[key]
                    rw = np.array([rewards[trainee_lst[i]] for i in range(len(trainee_lst))])
                    aux1 = np.array([sales[trainee_lst[i]] for i in range(len(trainee_lst))])
                    aux2 = np.array([self.costs[trainee_lst[i]] for i in range(len(trainee_lst))])
                    aux = aux1 * aux2
                    avg_reward = (rw.reshape(1, len(trainee_lst)) @ aux.reshape(len(trainee_lst), 1)) / np.sum(aux)

                    self.regressors[key].update_data(
                        [self.actions[self.last_actions_vect[key]], self.actions[self.last_actions_vect[self.used_graph_dict[key][0]]]],
                        rewards[key], avg_reward
                    )
        
        else:

            raise NotImplementedError("unknown graph scenario not implemented")
            self._update_graph_data()

        self.interactions_count += 1

    
    def _create_regressors(self):

        self.regressors = {}
        
        for key in list(self.used_graph_dict.keys()):
            if len(self.used_graph_dict[key]) == 0: 
                self.regressors[key] = GaussianProcessRegressor(self.kernel_L, self.sigma_process, 1)
            else: 
                self.regressors[key] = DoubleGaussianProcessRegressor(self.kernel_L, self.sigma_process, 2)

    
    def _update_graph_data(self):

        raise NotImplementedError("unknown graph scenario not implemented")
