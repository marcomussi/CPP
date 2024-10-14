class PricingAgentComplementaryOLD: 

    
    def __init__(self, num_products, actions, costs, kernel_L, sigma_process, horizon, graph_dict=None, verbose=True):

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

        self.verbose = verbose
        
        if graph_dict is None:
            self.known_relations = False
            self.interactions_dataset = -1 * np.ones((num_products, horizon, 2))
        else:
            self.known_relations = True
            self.used_graph_dict = graph_dict
            self._create_regressors()

        if self.verbose:
            print("num products and actions", self.num_products, self.num_actions)
            print("actions", self.actions)
            print("costs", self.costs)
            print("horizon and beta", self.horizon, self.beta)
            print("total sales", self.total_sales)
            print("byvariate_actions", self.byvariate_actions)
            print("byvariate_actions_map", self.byvariate_actions_map)
            print("used_graph_dict", self.used_graph_dict)

    
    def pull(self):

        if self.verbose:
            print("Executing pull(). Interaction count before this: " + str(self.interactions_count))
        
        if self.interactions_count == 0:
            
            self.last_actions_vect = np.zeros(self.num_products, dtype=int)
        
        else:

            self.last_actions_vect = -1 * np.ones(self.num_products, dtype=int)
            
            if self.known_relations:

                if self.verbose:
                    print("Executing pull(). self.used_graph_dict.keys(): " + str(self.used_graph_dict.keys()))
                
                for key in list(self.used_graph_dict.keys()):

                    if self.verbose:
                        print("Executing pull(). running for cicle for key: " + str(key))
                    
                    if len(self.used_graph_dict[key]) == 0: 

                        if self.verbose:
                            print("Executing pull(). We have " + str(len(self.used_graph_dict[key])) + "trained products: we use single regressor")
                        
                        mu, sigma = self.regressors[key].compute(self.actions.reshape(self.num_actions, 1))
                        self.last_actions_vect[key] = np.argmax(self.actions * (mu + self.beta * sigma))
                        
                    else: 
                        
                        trainee_lst = self.used_graph_dict[key]
                        aux1 = np.array([self.total_sales[trainee_lst[i]] for i in range(len(trainee_lst))])
                        aux2 = np.array([self.costs[trainee_lst[i]] for i in range(len(trainee_lst))])
                        trainee_w = np.sum(aux1 * aux2)
                        trainer_w = self.total_sales[key] * self.costs[key]

                        if self.verbose:
                            print("Executing pull(). We have " + str(len(self.used_graph_dict[key])) + "trained products: we use double regressor")
                            print("Executing pull(). trainee_lst: " + str(trainee_lst))
                            print("Executing pull(). aux1 (sales): " + str(aux1))
                            print("Executing pull(). aux2 (costs): " + str(aux2))
                            print("Executing pull(). trainee_w: " + str(trainee_w))
                            print("Executing pull(). trainer_w: " + str(trainer_w))
                        
                        mu_y1, mu_y2, sigma = self.regressors[key].compute(self.byvariate_actions)
                        idx = np.argmax(
                            self.byvariate_actions[:, 0] * trainer_w * (mu_y1 + self.beta * sigma) + 
                            self.byvariate_actions[:, 1] * trainee_w * (mu_y2 + self.beta * sigma)
                        )

                        if self.verbose:
                            print("Executing pull(). mu_y1: " + str(mu_y1))
                            print("Executing pull(). mu_y2: " + str(mu_y2))
                            print("Executing pull(). sigma: " + str(sigma))
                            print("Executing pull(). idx: " + str(idx))
                            
                        self.last_actions_vect[key] = self.byvariate_actions_map[idx, 0]
                        for i in range(len(trainee_lst)):
                            self.last_actions_vect[trainee_lst[i]] = self.byvariate_actions_map[idx, 1]
            
            else:
                
                raise NotImplementedError("unknown graph scenario not implemented")     

        if self.verbose:
            print("Executing pull(). Playing: " + str(self.last_actions_vect))
        
        return self.last_actions_vect
            
    
    def update(self, sales, impressions, interactions=None):
        
        assert sales.shape == (self.num_products, ) and impressions.shape == (self.num_products, ), "update(): error in input"
        
        self.total_sales = self.total_sales + sales
        
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
                    aux1 = np.nan_to_num(aux1, nan=1.0)
                    aux1 = np.maximum(aux1, 1)
                    aux = aux1 * aux2
                    avg_reward = (rw.reshape(1, len(trainee_lst)) @ aux.reshape(len(trainee_lst), 1)) / np.sum(aux)

                    if self.verbose:
                        print("Executing update(). aux1 (sales): " + str(aux1))
                        print("Executing update(). aux2 (costs): " + str(aux2))
                        print("Executing update(). avg_reward: " + str(avg_reward))
                    
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
        
        if self.verbose:
            print("Created regressors: " + str(self.regressors) + " from dict: " + str(self.used_graph_dict))

    
    def _update_graph_data(self):

        raise NotImplementedError("unknown graph scenario not implemented")