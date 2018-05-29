import numpy as np 


# Generic player
class player:
	def __init__(self, num_arms=10, budget=1, is_infinity=False):
		# When infinity arms are present, the idea is to play
		# with random arms. So we need random indexes.
		self.all_arms = np.arange(num_arms)
		
		if is_infinity:
			np.random.shuffle(self.all_arms)

		# If it is infinity, then we pick random k arms, following
		# the k definition in the paper of those guys
		self.c = 1.0
		self.beta = 1
		if is_infinity:
			if self.beta < 1:
				self.num_arms = self.c * np.power(self.budget, self.beta/2.0)
			elif self.beta >= 1:
				self.num_arms = self.c * np.power(self.budget, (self.beta)/(self.beta + 1.0))
			else:
				print ('ERROR in beta value. Working with all arms')
				self.num_arms = num_arms
		else:
			self.num_arms = num_arms

		#self.all_arms = np.zeros(num_arms)
		self.rewards = np.zeros(self.num_arms)
		self.costs = np.zeros(self.num_arms)
		self.pulls = np.zeros(self.num_arms)
		self.reward = 0.0
		self.cost = 0.0
		self.budget = budget
		self.valid_budget = budget
		self.t = 0
		self.is_infinity = is_infinity


	def play(self, casino):
		#Do nothing
		self.reward = self.reward + 0.0
		self.cost = self.cost + 0.0
		self.t = self.t + 1

	def play_masked_arm(self, casino, arm_):
		arm_masked = self.all_arms[arm_]
		r, c = casino.play(arm_masked)
		return r, c

	def get_num_arms(self):
		return self.num_arms

	def get_prize(self):
		return self.reward, self.cost

	def regret(self, casino):
		best_arm_reward = casino.get_best_expected_reward()
		return (best_arm_reward - (self.get_prize()[0]*1.0)/(1.0*self.t))

	def local_regret(self, casino):
		best_local_arm_reward = -10
		for j in self.num_arms:
			i = self.all_arms[j]
			rr = casino.get_arm_i_expected_reward(i)
			if rr > best_local_arm_reward:
				best_local_arm_reward = rr

		return (best_local_arm_reward - (self.get_prize()[0]*1.0)/(1.0*self.t))

	def update(self, r, c, arm_):
		
		self.budget = self.budget - c

		if self.budget >= 0:
			self.valid_budget = self.budget
			self.reward = self.reward + r
			self.cost = self.cost + c

			self.t = self.t + 1
			self.pulls[arm_] = self.pulls[arm_] + 1
			self.rewards[arm_] = self.rewards[arm_] + r
			self.costs[arm_] = self.costs[arm_] + c
		else:
			print ('Player {}: Budget exhausted. Action {} denied\n\tFinal remaining budget: {}'.format(self.get_id(), self.t+1, self.budget+c))

	def remaining_budget(self):
		return self.budget

	def remaining_valid_budget(self):
		return self.valid_budget

	def best_arm(self):
		return np.argmax(self.rewards/(self.pulls + 0.01))

	def best_arm_casino(self):
		j = (int)(self.best_arm())
		return self.all_arms[j]

	def get_total_plays(self):
		return self.t

	def get_id(self):
		return 'Generic'

	# returns the first arm that founds that hasnt been played
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
	def __init__(self, num_arms, budget, is_infinity):
		super().__init__(num_arms, budget, is_infinity)

	def play(self, casino):
		arm_ = np.random.randint(self.num_arms)

		if self.is_infinity:
			r, c = self.play_masked_arm(casino, arm_)
		else:
			r, c = casino.play_arm(arm_)

		self.update(r, c, arm_)

	def get_id(self):
		return 'Random'


# =============================================================
#	Agent that plays with epsilon greedy policy
#
# =============================================================

class eps_greedy_player(player):
	def __init__(self, num_arms, budget, is_infinity, params=None):
		super().__init__(num_arms, budget, is_infinity)
		if params != None:
			if 'epsilon' in params:
				self.epsilon = params['epsilon']
			else:
				self.epsilon = 0.1
		else:
			self.epsilon = 0.1
		self.epsilon = self.epsilon/(self.num_arms-1)

	def play(self, casino):
		# the value to test agaisnt epsilon
		e = np.random.rand()
		arm_ = 0

		if e < self.epsilon:
			arm_ = np.random.choice(list(set(range(self.num_arms)) -
                                    {self.best_arm()}))
		else:
			arm_ = self.best_arm()
		
		if self.is_infinity:
			r, c = self.play_masked_arm(casino, arm_)
		else:
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
	def __init__(self, num_arms, budget, is_infinity, params=None):
		super().__init__(num_arms, budget, is_infinity)
		if params != None:
			if 'tau' in params:
				self.tau = params['tau']
			else:
				self.tau = 0.1
		else:
			self.tau = 0.1

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

		if self.is_infinity:
			r, c = self.play_masked_arm(casino, arm_)
		else:
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
	def __init__(self, num_arms, budget, is_infinity, params=None):
		super().__init__(num_arms, budget, is_infinity)
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

		if self.is_infinity:
			r, c = self.play_masked_arm(casino, arm_)
		else:
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
	def __init__(self, num_arms, budget, is_infinity, params=None):
		super().__init__(num_arms, budget, is_infinity)
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

		self.rewards2 = np.zeros(self.num_arms)
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

		if self.is_infinity:
			r, c = self.play_masked_arm(casino, arm_)
		else:
			r, c = casino.play_arm(arm_)

		self.update_all(r, c, arm_)

	# returns the first arm that founds that hasnt been played
	# def missing_arm(self):
	# 	for a_ in range(self.num_arms):
	# 		if self.pulls[a_] == 0:
	# 			if a_ == (self.num_arms - 1):
	# 				self.cold_start = False
	# 			return a_
	# 	return -1

	def update_all(self, r, c, arm_):
		self.budget = self.budget - c 
		if self.budget >= 0:
			self.valid_budget = self.budget
			self.pulls[arm_] = self.pulls[arm_] + 1
			self.rewards[arm_] = self.rewards[arm_] + r
			self.rewards2[arm_] = self.rewards2[arm_] + r*r
			self.costs[arm_] = self.costs[arm_] + c

			self.reward = self.reward + r
			self.cost = self.cost + c
			self.t = self.t + 1
		else:
			print ('Player {}: Budget exhausted. Action {} denied\n\tFinal remaining budget: {}'.format(self.get_id(), self.t+1, self.budget+c))

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
	def __init__(self, num_arms, budget, is_infinity, params=None):
		super().__init__(num_arms, budget, is_infinity)
		if params != None:
			if 'c' in params:
				self.c = params['c']
			else:
				self.c = 0.0
		else:
			self.c = 0.0

		self.kl = np.zeros(self.num_arms)
		self.t = 0

		self.cold_start = True

	def play(self, casino):

		# Handle cold start. Not all bandits tested yet.
		if self.cold_start:
			arm_ = self.missing_arm()
		else:
			for a_ in range(self.num_arms):
				self.kl[a_] = self.get_max_kl(a_)
			arm_ = np.argmax(self.kl)

		if self.is_infinity:
			r, c = self.play_masked_arm(casino, arm_)
		else:
			r, c = casino.play_arm(arm_)
		
		self.update(r, c, arm_)

	# as noted by Garivier,  for any p ∈ [0, 1] the function
	# q |--> d(p, q) is strictly convex and increasing on the interval [p, 1].
	def get_max_kl(self, k):
		delta = 1e-8
		eps = 1e-12
		# by recomendation of Garivier, c = 0
		logndn = (np.log(self.t)  + self.c*np.log(np.log(self.t)))/self.pulls[k]
		# logndn = np.log(self.t)/self.pulls[k]
		p = max(self.rewards[k]/self.pulls[k], delta)

		# if p >= upper bound of the reward
		if p >= 1:
			return 1

		# Newton iterations
		converged = False
		q = 1.0*p + delta
		for i in range (20):
			if not converged:
				f = logndn - self.kl_div(p, q)
				df = - self.dkl_div(p, q)
				if f*f < eps:
					converged = True
				q = min(1-delta, max(q-np.nan_to_num(f/df), p+delta))
		if not converged:
			print ('WARNING: kl didn\'t converged for arm {}. t = {}'.format(k, self.t))
		return q

	#computes the kl divergence
	def kl_div(self, p, q):
		t1 = p*np.log(p/q)
		t2 = (1.0-p)*np.log((1.0-p)/(1.0-q))
		return (t1+t2)

	# differenciate with respect to q, the variable
	def dkl_div(self, p, q):
		return (q-p)/(q*(1.0-q))

	# returns the first arm that founds that hasnt been played
	# def missing_arm(self):
	# 	for a_ in range(self.num_arms):
	# 		if self.pulls[a_] == 0:
	# 			if a_ == (self.num_arms - 1):
	# 				self.cold_start = False
	# 			return a_
	# 	return -1

	def best_arm(self):
		return np.argmax(self.rewards / (self.pulls + 0.01))

	def estimated_payout(self):
		return self.rewards/(self.pulls+0.01)

	def rel_regret(self):
		max_ = np.max(np.nan_to_num(self.rewards/self.pulls))
		return ( max_ - sum(self.rewards) / sum(self.pulls) )

	def get_id(self):
		return 'kl-ucb'