import numpy as np 


# Generic player


class player:
	def __init__(self, num_arms):
		self.num_arms = num_arms
		self.rewards = np.zeros(num_arms)
		self.costs = np.zeros(num_arms)
		self.pulls = np.zeros(num_arms)
		self.reward = 0.0
		self.cost = 0.0
		self.t = 0

	def play(self, casino):
		self.reward = self.reward + 0.0
		self.cost = self.cost + 0.0
		self.t = self.t + 1

	def get_prize(self):
		return self.reward, self.cost

	def regret(self, casino):
		best_arm_ = casino.best_arm_reward()
		return (best_arm_ - (self.get_prize()[0]*.1)/(1.0*self.t))

	def update(self, r, c, arm_):
		self.t = self.t + 1
		self.pulls[arm_] = self.pulls[arm_] + 1
		self.rewards[arm_] = self.rewards[arm_] + r
		self.costs[arm_] = self.costs[arm_] + c

		self.reward = self.reward + r
		self.cost = self.cost + c

	def best_arm(self):
		return np.argmax(self.rewards/(self.pulls + 0.01))

	def get_id(self):
		return 'Generic'

	def missing_arm(self):
		for a_ in range(self.num_arms):
			if self.pulls[a_] == 0:
				if a_ == self.num_arms - 1:
					self.cold_start = False
				return a_
		return -1



# =============================================================
#	Agent that plays random
#
# =============================================================

class random_player(player):
	def __init__(self, num_arms):
		self.num_arms = num_arms
		self.rewards = np.zeros(num_arms)
		self.costs = np.zeros(num_arms)
		self.pulls = np.zeros(num_arms)
		self.reward = 0.0
		self.cost = 0.0
		self.t = 0

	def play(self, casino):
		arm_ = np.random.randint(self.num_arms)
		r, c = casino.play_arm(arm_)

		self.update(r, c, arm_)

	def regret(self, casino):
		best_arm_ = casino.best_arm_reward()
		return (best_arm_ - (self.get_prize()[0]*.1)/(1.0*self.t))

	def get_id(self):
		return 'Radom'


# =============================================================
#	Agent that plays with epsilon greedy policy
#
# =============================================================

class eps_greedy_player(player):
	def __init__(self, num_arms, params=None):
		super().__init__(num_arms)
		if params != None:
			if 'epsilon' in params:
				self.epsilon = params['epsilon']
			else:
				self.epsilon = 0.1
		else:
			self.epsilon = 0.1
		self.epsilon = self.epsilon/(self.num_arms-1)

		self.num_arms = num_arms
		self.rewards = np.zeros(num_arms)
		self.costs = np.zeros(num_arms)
		self.pulls = np.zeros(num_arms)

	def play(self, casino):
		# the value to test agaisnt epsilon
		e = np.random.rand()
		arm_ = 0

		if e < self.epsilon:
			arm_ = np.random.choice(list(set(range(self.num_arms)) -
                                    {self.best_arm()}))

		else:
			arm_ = self.best_arm()
		
		r, c = casino.play_arm(arm_)

		self.update(r, c, arm_)

	def best_arm(self):
		return np.argmax(self.rewards / (self.pulls + 0.01))

	def estimated_payout(self):
		return self.rewards/(self.pulls+0.01)

	def rel_regret(self):
		max_ = np.max(np.nan_to_num(self.rewards/self.pulls))
		return (max_ - sum(self.rewards) / sum(self.pulls))

	def get_id(self):
		return 'Epsilon greedy'


# =============================================================
#	Agent that plays with softmax policy
#
# =============================================================

class softmax_player(player):
	def __init__(self, num_arms, params=None):
		super().__init__(num_arms)
		if params != None:
			if 'tau' in params:
				self.tau = params['tau']
			else:
				self.tau = 0.1
		else:
			self.tau = 0.1


		self.num_arms = num_arms
		self.rewards = np.zeros(num_arms)
		self.costs = np.zeros(num_arms)
		self.pulls = np.zeros(num_arms)
		self.q = np.zeros(self.num_arms)
		self.cold_start = True
		self.start = True

	def play(self, casino):

		if self.cold_start:
			arm_ = self.missing_arm()
		else:
			self.q = self.rewards/(self.pulls + 0.01)
			self.norm = sum(np.exp(self.q/self.tau))

			soft_probs = np.exp(self.q/self.tau)/self.norm

			cumulative_prob = [sum(soft_probs[:i+1]) for i in range(len(soft_probs))]

			index = np.random.rand()

			found = False
			arm_ = None
			i = 0

			while not found:
				if index < cumulative_prob[i]:
					arm_ = i
					found = True
				else:
					i += 1

		r, c = casino.play_arm(arm_)

		self.update(r, c, arm_)


	def best_arm(self):
		return np.argmax(self.rewards / (self.pulls + 0.01))

	def estimated_payout(self):
		return self.rewards/(self.pulls+0.01)

	def rel_regret(self):
		max_ = np.max(np.nan_to_num(self.rewards/self.pulls))
		return ( max_ - sum(self.rewards) / sum(self.pulls) )

	def get_id(self):
		return 'soft-max'



# =============================================================
#	Agent that plays with UCB policy
#
# =============================================================

class ucb1(player):
	def __init__(self, num_arms, params=None):
		super().__init__(num_arms)
		if params != None:
			if 'epsilon' in params:
				self.epsilon = params['epsilon']
			else:
				self.epsilon = 0.1
			if 'cold_start' in params:
				self.cold_start = params['cold_start']
			else:
				self.cold_start = True
		else:
			self.epsilon = 0.1
			self.cold_start = True

		self.num_arms = num_arms
		self.rewards = np.zeros(num_arms)
		self.costs = np.zeros(num_arms)
		self.pulls = np.zeros(num_arms)

		if self.cold_start:
			self.varepsilon = 0.0
		else:
			self.varepsilon = 0.01

	def play(self, casino):

		# Handle cold start. Not all bandits tested yet.
		if self.cold_start:
			arm_ = self.missing_arm()
		else:
			total_pulls = sum(self.pulls)
			q = self.rewards / (self.pulls + self.varepsilon)

			ucb_ = q + np.sqrt(2*np.log(total_pulls)/(self.pulls + self.varepsilon))
			arm_ = np.argmax(ucb_) 

		r, c = casino.play_arm(arm_)

		self.update(r, c, arm_)
			
	def best_arm(self):
		return np.argmax(self.rewards / (self.pulls + self.varepsilon))

	def estimated_payout(self):
		return self.rewards/(self.pulls + self.varepsilon)

	def rel_regret(self):
		max_ = np.max(np.nan_to_num(self.rewards/self.pulls))
		return ( max_ - sum(self.rewards) / sum(self.pulls + self.varepsilon) )

	def get_id(self):
		return 'ucb1'


# =============================================================
#	Agent that plays with UCB-V policy
#	http://certis.enpc.fr/~audibert/ucb_alt.pdf
# =============================================================
class ucb_v(player):
	def __init__(self, num_arms, params=None):
		super().__init__(num_arms)
		if params != None:
			if 'zeta' in params:
				self.zeta = params['zeta']
			else:
				self.zeta = 1.2
			if 'c' in params:
				self.c = params['c']
			else:
				self.c = 1
			if 'b' in params:
				self.b = params['b']
			else:
				self.b = 1
		else:
			self.zeta = 1.2
			self.c = 1
			self.b = 1

		self.num_arms = num_arms
		self.rewards = np.zeros(num_arms)
		self.rewards2 = np.zeros(num_arms)
		self.costs = np.zeros(num_arms)
		self.pulls = np.zeros(num_arms)
		self.cold_start = True
		self.t = 0

	def play(self, casino):

		# Handle cold start. Not all bandits tested yet.
		if self.cold_start:
			arm_ = self.missing_arm()
		else:
			x_barr = self.rewards / (self.pulls)
			variance = self.rewards2/(self.pulls) - ( self.rewards*self.rewards/(self.pulls*self.pulls) )
			
			ucbv_ = x_barr + np.sqrt(2*variance*np.log(self.t)/self.pulls) + 3*self.c*self.b*np.log(self.t)/self.pulls
			arm_ = np.argmax(ucbv_)

		r, c = casino.play_arm(arm_)

		self.update_all(r, c, arm_)

	# returns the first arm that founds that hasnt been played
	def missing_arm(self):
		for a_ in range(self.num_arms):
			if self.pulls[a_] == 0:
				if a_ == (self.num_arms - 1):
					self.cold_start = False
				return a_
		return -1

	def update_all(self, r, c, arm_):
		self.pulls[arm_] = self.pulls[arm_] + 1
		self.rewards[arm_] = self.rewards[arm_] + r
		self.rewards2[arm_] = self.rewards2[arm_] + r*r
		self.costs[arm_] = self.costs[arm_] + c

		self.reward = self.reward + r
		self.cost = self.cost + c
		self.t = self.t + 1

	def best_arm(self):
		return np.argmax(self.rewards / (self.pulls + 0.01))

	def estimated_payout(self):
		return self.rewards/(self.pulls + 0.01)

	def rel_regret(self):
		max_ = np.max(np.nan_to_num(self.rewards/self.pulls))
		return ( max_ - sum(self.rewards) / sum(self.pulls) )

	def get_id(self):
		return 'ucb-v'


# =============================================================
#	Agent that plays with kl-ucb policy
#	http://proceedings.mlr.press/v19/garivier11a/garivier11a.pdf
# =============================================================
class kl_ucb(player):
	def __init__(self, num_arms, params=None):
		super().__init__(num_arms)
		if params != None:
			if 'zeta' in params:
				self.zeta = params['zeta']
			else:
				self.zeta = 1.2
			if 'c' in params:
				self.c = params['c']
			else:
				self.c = 1
			if 'b' in params:
				self.b = params['b']
			else:
				self.b = 1
		else:
			self.zeta = 1.2
			self.c = 1
			self.b = 1

		self.num_arms = num_arms
		self.rewards = np.zeros(num_arms)
		self.kl = np.zeros(num_arms)
		self.costs = np.zeros(num_arms)
		self.pulls = np.zeros(num_arms)
		self.t = 0

		self.cold_start = False

	def play(self, casino):

		# Handle cold start. Not all bandits tested yet.
		if self.cold_start:
			f, arm_ = self.missing_arm()
		else:
			for a_ in range(self.num_arms):
				self.kl[a_] = get_kl[a_]
			arm_ = np.argmax(kl)

		r, c = casino.play_arm(arm_)
		self.t = self.t + 1

		self.pulls[arm_] = self.pulls[arm_] + 1
		self.rewards[arm_] = self.rewards[arm_] + r
		self.costs[arm_] = self.costs[arm_] + c

		self.reward = self.reward + r
		self.cost = self.cost + c

	def get_kl(self, k):
		delta = 1e-8
		eps = 1e-12
		logndn = np.log(self.t)/self.pulls[k]
		p = max(self.rewards[k]/self.pulls[k], delta)
		if p >= 1:
			return 1
		converged = False
		q = 1.0*p + delta
		for i in range (20):
			if not converged:
				f = logndn - kl_div(p, q)
				df = - dkl_div(p, q)
				if f*f < eps:
					converged = True
				q = min(1-delta, max(q-f/df, p+delta))
		if not converged:
			print ('WARNING: kl didnt converged for arm {}'.format(k))
		return q

	def kl_div(self, p, q):
		t1 = p*np.log(p/q)
		t2 = (1.0-p)*np.log((1.0-p)/(1.0-q))
		return (t1+t2)

	def dkl_div(self, p, q):
		return (q-p)/(q*(1.0-q))

	# returns the first arm that founds that hasnt been played
	def missing_arm(self):
		for a_ in range(self.num_arms):
			if self.pulls[a_] == 0:
				if a_ == self.num_arms - 1:
					self.cold_start = False
				return a_

	def best_arm(self):
		return np.argmax(self.rewards / (self.pulls + 0.01))

	def estimated_payout(self):
		return self.rewards/(self.pulls+0.01)

	def rel_regret(self):
		max_ = np.max(np.nan_to_num(self.rewards/self.pulls))
		return ( max_ - sum(self.rewards) / sum(self.pulls) )

	def get_id(self):
		return 'kl-ucb'



