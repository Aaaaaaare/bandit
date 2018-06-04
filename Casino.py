import numpy as np

types = ['bernoulli', 'normal', 'uniform']

class arm:
    def __init__(self, type_b='bernoulli', is_cost=False, random=True, parameters=None):
        # Extra information goes here
        self.parameters = parameters
        self.type = type_b
        self.random = random
        self._cost = is_cost

        if parameters != None:
            # The cost can only follow a uniform distribution
            if self._cost == True:
                if 'cost' in parameters:
                    self.cost_upper = parameters['cost']
                else:
                    self.cost_upper = np.random.rand()
            if 'mean' in parameters:
                self.mean = parameters['mean']    
    
            if 'variance' in parameters:
                self.variance = parameters['variance']

            if 'prob' in parameters:
                self.prob = parameters['prob']
                
        else:   
            if type_b == 'normal':
                self.cost = 0
                self.mean = np.random.rand()/2 + 0.5
                self.variance = 0.1
            if type_b == 'bernoulli':
                self.prob = np.random.rand()
            if type_b == 'uniform':
                self.upper = np.random.rand()
            if self._cost:
                self.cost_upper = np.random.uniform(0.3, 1)
                self.cost_lower = np.random.uniform(0.1, self.cost_upper)
        
        if self.type not in types:
            raise NameError('ERROR 1: Payoff distribution not recognized')

    def get_params(self):
        if self.type == 'normal':
            return self.type, self.cost, self.mean, self.variance
        elif self.type == 'bernoulli':
            return self.type, self.prob
        elif self.type == 'uniform':
            return self.type, self.upper

    def get_type(self):
        return self.type

    def get_Expected_reward(self):
        if self.type == 'bernoulli':
            return self.prob
         
        if self.type == 'normal':
            return self.mean
                
        if self.type == 'uniform':   
            return 0.5*self.upper

    def get_best_reward(self):
        if self.type == 'bernoulli':
            return 1
         
        if self.type == 'normal':
            return self.mean + 2*self.variance
                
        if self.type == 'uniform':   
            return self.upper
        
    def play(self):
        return self.generate_reward(), self.generate_cost()

    def generate_reward(self):
        if self.type == 'bernoulli':
            if np.random.rand() < self.prob:
                return 1
            else:
                return 0

        elif self.type == 'normal':
            rw = np.random.normal(self.mean, self.variance)
            rw = max(0, rw)
            rw = min(1, rw)
            return rw

        elif self.type == 'uniform':
            return np.random.uniform(0, self.upper)
        else: # never
            return 1
            
    def generate_cost(self):
        if self._cost:
            return np.random.uniform(self.cost_lower, self.cost_upper)
        else:
            #return max(np.random.normal(self.cost_mean, self.cost_vart), 0)
            return 0
                
class bandit:
    def __init__(self, num_arms=2, type_b='bernoulli', is_cost=False, infinity=False, params=None):
        self.infinity = infinity
        self.num_arms = num_arms
        self.type_b = type_b
        self.is_cost = is_cost
        self.arms = [ arm(type_b=self.type_b, is_cost=self.is_cost, random=True) for i in range(self.num_arms)]
        self.t = 0

        self.best_mean_reward = self.best_arm_reward()

    def add_arm(self, given_arm):
        if type(given_arm) == arm:
            self.arms.append(given_arm)
            self.num_arms = self.num_arms + 1
        else:
            print ('ERROR: Object is not an arm')

    def get_type_b(self):
        return self.type_b

    def play_arm(self, index):
        self.t = self.t + 1
        return self.arms[index].play()

    def best_arm_reward(self):
        best_arm_ = sorted(self.arms, key=lambda arm: arm.get_Expected_reward())[-1]
        return best_arm_.get_Expected_reward()
        #best_arm_ = self.best_arm()
        #return self.arms[best_arm_].get_Expected_reward()

    def best_arm(self):
        #b_arm_ = np.argmax(self.arms.get_Expected_reward())
        # list(data[i] for i in range(len(data)-1, -1, -1)) ?????
        # RETURNS THE INDEX OF THE BEST AMR 
        max_rew = -1
        best_arm_ = -1
        for i in range(self.num_arms):
            new_rew = self.arms[i].get_Expected_reward()
            if self.arms[i].get_Expected_reward() > max_rew:
                best_arm_ = i
                max_rew = new_rew
        return best_arm_

    def get_arm_i_expected_reward(self, i):
        return self.arms[i].get_Expected_reward()

    def get_best_expected_reward(self):
        return self.best_mean_reward

    def print_all_arms(self):
        for i, arm in enumerate(self.arms):
            print ('arm {}: expected reward: {}'.format(i, arm.get_Expected_reward()))