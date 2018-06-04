from players import player
import numpy as np 

# =============================================================
#	Agent that plays following RCB-I
#
# =============================================================
class RCB_I(player):
	def __init__(self, num_arms=10, budget=1, is_infi=False, params=None):
		#super().__init__(num_arms, budget)

		# algorith parameter
		self.beta = 1.0

		# If specified, override
		if params != None:
			if 'beta' in params:
				self.beta = params['beta']

		self.is_infi = is_infi

		self.budget = budget

		if is_infi:
			self.c = 1.0
			if self.beta < 1:
				self.k = int(self.c * np.power(self.budget, self.beta/2.0))
			elif self.beta >= 1:
				self.k = int(self.c * np.power(self.budget, (self.beta)/(self.beta + 1.0)))
			else:
				print ('ERROR in beta value. Working with all arms')
				self.k = int(num_arms)
		else:
			self.k = num_arms

		super().__init__(self.k, budget, is_infi)

		self.cold_start = True

	def play(self, casino):
		# Handle cold start. Not all bandits tested yet.
		if self.cold_start:
			arm_ = self.missing_arm()
		else:
			confidence_i = np.sqrt( self.exploration(self.t) / (2.0*self.pulls) )
			r_bar_aux = self.rewards/self.pulls
			r_bar = r_bar_aux + confidence_i
			c_bar_aux = self.costs/self.pulls
			c_bar = c_bar_aux - confidence_i

			numerator_ = np.array([ min(r_bar_, 1) for r_bar_ in r_bar ])
			denominator_ = np.array([ max(c_bar_, 1e-12) for c_bar_ in c_bar ])
			d = (numerator_/denominator_)

			arm_ = np.argmax(d)

		if self.is_infinity:
			r, c = self.play_masked_arm(casino, arm_)
		else:
			r, c = casino.play_arm(arm_)
		
		self.update(r, c, arm_)

	# Define the exploration sequence function.
	# according to the authorsÑ
	# 2*log(4*log_2(t+1)) <= exploration(t) <= log(t)
	def exploration(self, t):
		v = min(2*np.log(4*np.log2(t)+4), np.log(t))
		return np.log(t)

	def get_number_arms_played(self):
		return self.k

	def get_num_arms():
		return self.k

	def get_id(self):
		return 'RBC-I'


# =============================================================
#	Agent that plays following RCB-AIR
#	-- This algorithm works for any-budget, meaning
# 	it will also work if the budget is unknow
# =============================================================
class RCB_AIR(player):
	def __init__(self, num_arms=10, is_infi=False, params=None):
		
		# We need to feed the arms with the idea of a budget
		# 10 means: there is still budget, eventhogh we dont know it.
		budget = 10
		super().__init__(num_arms, budget)

		# algorith parameter
		self.beta = 1.0

		# If specified, override
		if params != None:
			if 'beta' in params:
				self.beta = params['beta']

		# It will always asume is not infinity
		self.is_infi = is_infi

		# This is a total number of arms, required for simplicity of simulation
		self.num_arms_total = num_arms

		# it strts with 0 arms.
		# k is the number of arms it is able to index, so it is the
		# aparent number of arms.
		self.k = 0


	def play(self, casino):
		# Add a new arm??
		condition = 0.0
		# Added for coherency
		self.t = self.t + 1
		if self.beta < 1.0:
			condition = np.power(self.t, self.beta/2.0)
		else:
			condition = np.power(self.t, self.beta/(self.beta + 1.0))

		# if the current number of arms is less than the condition,
		# then we add a new arm. Notice that at the begining, condition = 0, k =0
		# so it will always add an arm at the begining. 
		if self.k <= condition:
			# To not overflow:
			if self.k < self.num_arms_total:
				# Index one more arm
				self.k = self.k + 1

			# Play the newly added arm
			# -1 because the indexes.. 
			arm_ = 	(self.k - 1)

		# If no arm needs to be added: Run the normal algorithm in the remaining arms
		elif True:
			confidence_i = np.sqrt( self.exploration(self.t) / (2*self.pulls[:self.k]+0.01) )
			r_bar = self.rewards[:self.k]/(self.pulls[:self.k] + 0.01)
			r_bar = r_bar + confidence_i
			c_bar = self.costs[:self.k]/(self.pulls[:self.k] + 0.01)
			c_bar = c_bar - confidence_i

			numerator_ = np.array([ min(r_bar_, 1) for r_bar_ in r_bar])
			denominator_ = np.array([ max(c_bar_, 0) for c_bar_ in c_bar ])
			d = np.nan_to_num(numerator_/denominator_)

			arm_ = np.argmax(d)
		# this will never happen.
		# **Remainder of previous coding
		else:
			print ('WARNING: Something is extremely wrong!!!!')
			arm_ = 0

		r, c = casino.play_arm(arm_)
		
		# Removed because the Update function will update it again
		self.t = self.t - 1

		self.update(r, c, arm_)
		# we reset the budget again to 10
		self.budget = 10

	def get_number_arms_played(self):
		return self.k

	def get_num_arms():
		return self.k

	def exploration(self, t):
		v = min(2*np.log(4*np.log2(t)+4), np.log(t))
		return np.log(t)

	def get_id(self):
		return 'RBC-AIR'