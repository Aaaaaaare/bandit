import pro_players
from my_players import kl_ucb_alpha, ucb_alpha, RCB_I_alpha

import numpy as np

# ======================================================
# Generic class
#
# ======================================================
class alpha_0:
	def __init__(self):
		self.none = None
		self.gambler = None
		self.num_arms = 1

	def get_num_arms(self):
		return self.num_arms

	def get_prize(self):
		return self.gambler.get_prize()
		#return self.gambler.real_reward, self.gambler.real_cost

	def get_budget_used(self):
		return self.gambler.real_cost

	def best_arm(self):
		return np.argmax(self.gambler.rewards/(self.gambler.costs))

	def get_total_plays(self):
		return self.gambler.t

	def add_arm(self):
		self.gambler.add_arm()
		self.num_arms = self.num_arms + 1

	def remaining_budget(self):
		return self.gambler.remaining_budget()

	# Manage budget
	def remaining_valid_budget(self):
		return self.gambler.valid_budget

	def best_arm_reward(self):
		return max(self.gambler.real_rewards/(self.gambler.real_costs))

	def best_arm_info(self):
		i = np.argmax(self.gambler.real_rewards/(self.gambler.real_costs + 1e-8))
		r = self.gambler.real_rewards[i]/self.gambler.pulls[i]
		c = self.gambler.real_costs[i]/self.gambler.pulls[i]
		p = self.gambler.pulls[i]

		return {'index': i, 'reward': r, 'cost': c, 'pulls': p}

# =====================================================
# Budget know.
# Infinite arms - Unknown.
# =====================================================
class alpha_I(alpha_0):
	def __init__(self, budget=10000, params=None):
		super().__init__()
		# The number of arms is indefined. We will play by the budget.
		# Following the authors of [], choose arms acoordint to:
		
		# algorithm parameter
		self.beta = 1.0
		self.c = 0.37

		self.budget = budget
		gb = 'rcb'

		# If speficied, override
		if params != None:
			if 'beta' in params:
				self.beta = params['beta']
		if params != None:
			if 'gambler' in params:
				gb = params['gambler']

		if self.beta < 1:
			self.num_arms = int(self.c * np.power(self.budget, self.beta/2.0))
		elif self.beta >= 1:
			self.num_arms = int(self.c * np.power(self.budget, (self.beta)/(self.beta + 1.0)))
		else:
			print ('ERROR in beta value. Working with all arms')
			self.num_arms = int(num_arms)
		
		# upper limit of the number of arms. Should be np.e = 2.71
		self.max_num_arms = np.e*self.num_arms

		# At the beginining, we dont add arms.
		# We play with the ones we have
		self.new_arm = False
		self.initial_run = True

		
		# create my gambler. It is Blind to the infinity of the arms
		if gb == 'kl':
			self.gambler = kl_ucb_alpha(num_arms = self.num_arms, budget = self.budget)
		elif gb == 'ucb1':
			self.gambler = ucb_alpha(num_arms = self.num_arms, budget = self.budget)
		elif gb == 'rcb':
			self.gambler = RCB_I_alpha(num_arms = self.num_arms, budget = self.budget)
		else:
			print ('Error in gambler. {} is not a valid MBA. Using UCB1 instead'.format(gb))
			self.gambler = ucb_alpha(num_arms = self.num_arms, budget = self.budget)

	def play(self, casino):

		# in the first run, we play all the arms that we started with
		if self.initial_run:
			self.gambler.play(casino)
			if not self.gambler.is_an_arm_missing():
				self.initial_run = False
				self.new_arm = True

		# We need to add an arm if all arms have been played... (**)
		else:
			# If a arm needs to be added, we add it.
			# then, the gambler's play function will play it
			# because it will have 0 pulls
			if self.new_arm:
				#self.gambler.add_arm()
				#self.num_arms = self.num_arms + 1
				self.add_arm()

			self.gambler.play(casino)
			# (**)...and if we haven'f find a better arm of the rest yet.
			# Also, there is a top number of arms, 3*num_arms_initial
			if self.new_arm:
				if self.gambler.is_new_arm_better():
					self.new_arm = False
				if self.num_arms >= self.max_num_arms:
					self.new_arm = False

	def get_id(self):
		return 'alpha-I' + ', ' + self.gambler.get_id()



# =====================================================
# Budget Unknown.
# Infinite arms - Unknown.
# =====================================================
class alpha_II(alpha_0):
	def __init__(self, num_arms, params=None):
		super().__init__()
		# The number of arms is indefined. The budget is undefined...

		# algorithm parameter
		self.beta = 1.0

		gb = 'rcb'

		# If speficied, override
		if params != None:
			if 'beta' in params:
				self.beta = params['beta']
		if params != None:
			if 'gambler' in params:
				gb = params['gambler']


		# We start with one arms
		self.num_arms = 1

		# we dont know the budget. 10 is just a "there is still some"
		self.budget = 10

		# Hand-picked maximum
		# self.max_num_arms = num_arms
		self.max_num_arms = 1e8

		# Positivity factor
		self.gamma = 1.0

		# Hope factor
		self.theta = 1.0

		# At the beginining, we dont add arms.
		# We play with the ones we have
		self.new_arm = False
		self.initial_run = True
		self.counter = self.num_arms

		self.tt = 1.0
	
		# create my gambler. It is Blind to the infinity of the arms
		if gb == 'kl':
			self.gambler = kl_ucb_alpha(num_arms = self.num_arms, budget = self.budget)
		elif gb == 'ucb1':
			self.gambler = ucb_alpha(num_arms = self.num_arms, budget = self.budget)
		elif gb == 'rcb':
			self.gambler = RCB_I_alpha(num_arms = self.num_arms, budget = self.budget)
		else:
			print ('Error in gambler. {} is not a valid MBA. Using UCB1 instead'.format(gb))
			self.gambler = ucb_alpha(num_arms = self.num_arms, budget = self.budget)

	def play(self, casino):

		# We need to add an arm if all arms have been played...
		# This part can be removed, I thing. This is just to make it
		# easy to read
		if self.initial_run:
			self.gambler.play(casino)
			self.counter -= 1
			if not self.gambler.is_an_arm_missing():
				self.initial_run = False

		else:
			# is the counter running out of time?
			if self.counter <= 0:
				# pick an arm with prob epsilon
				e_ = np.random.rand()
				epsilon = self.get_epsilon()
				if e_ < epsilon:
					#self.gambler.add_arm()
					#self.num_arms = self.num_arms + 1
					self.add_arm()
				# restart the counter
				self.counter = self.num_arms

			# Now, play. If an arm was added, then the gambler will 
			# play it, because it's number of pulls will be zero
			self.gambler.play(casino)
			# As we dont know the real budget... it will always be refiled
			self.gambler.set_budget(10)

			self.counter -= 1

	def get_epsilon(self):
		#t1 = 1.0/np.log(self.gambler.t)

		t2_1 = max(self.gambler.rewards/(self.gambler.rewards + self.gambler.costs))
		t2_2 = np.mean(self.gambler.rewards/(self.gambler.rewards + self.gambler.costs))
		t3_1 = self.gambler.rewards[-1]/(self.gambler.rewards[-1] + self.gambler.costs[-1])
		#t3_2 = t2_2

		t2 = 1.0 - (t2_1 - t2_2)
		
		#t3 = np.log(1.0 + (t3_1 - t3_2))
		if t3_1 > t2_2:
			self.tt += 1.0
		else:
			self.tt = max(1.0, self.tt - 1.0)

		t1 = 1.0/np.log(self.gambler.t/self.tt)

		return (t1 * np.power(t2, self.gamma) )

	def get_id(self):
		return 'alpha-II' + ', ' + self.gambler.get_id()