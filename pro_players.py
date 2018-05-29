from players import player
import numpy as np 

# =============================================================
#	Agent that plays following RCB-I
#
# =============================================================
class RCB_I(player):
	def __init__(self, num_arms=10, budget=1, is_infi= False, params=None):
		#super().__init__(num_arms, budget)
		if params != None:
			if 'beta' in params:
				self.beta = params['beta']
			else:
				self.beta = 1.0
		else:
			self.beta = 1.0
		self.is_infi = is_infi

		if is_infi:
			self.c = 1.0
			if self.beta < 1:
				self.k = self.c * np.power(self.budget, self.beta/2.0)
			elif self.beta >= 1:
				self.k = self.c * np.power(self.budget, (self.beta)/(self.beta + 1.0))
			else:
				print ('ERROR in beta value. Working with all arms')
				self.k = num_arms
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

	def exploration(self, t):
		return np.log(t)

	def get_id(self):
		return 'RBC-I'


# =============================================================
#	Agent that plays following RCB-AIR
#	-- This algorithm works for any-budget, meaning
# 	it will also work if the budget is unknow
# =============================================================
class RCB_AIR(player):
	def __init__(self, num_arms=10, budget=1, is_infi=False, params=None):
		super().__init__(num_arms, budget)
		if params != None:
			if 'beta' in params:
				self.beta = params['beta']
			else:
				self.beta = 1.0
		else:
			self.beta = 1.0
		self.is_infi = is_infi
		self.num_arms_total = num_arms

		# it strts with 0 arms
		self.k = 0


	def play(self, casino):
		# Add a new arm??
		condition = 0.0
		if self.beta < 1.0:
			condition = np.power(self.t, self.beta/2.0)
		else:
			condition = np.power(self.t, self.beta/(self.beta + 1.0))

		if self.k <= condition:
			if self.k < self.num_arms_total:
				self.k = self.k + 1

		# Run the normal algorithm in the remaining arms
		if True:
			confidence_i = np.sqrt( np.log(self.t) / (2*self.pulls[:self.k]+0.01) )
			r_bar = self.rewards[:self.k]/(self.pulls[:self.k] + 0.01)
			r_bar = r_bar + confidence_i
			c_bar = self.costs[:self.k]/(self.pulls[:self.k] + 0.01)
			c_bar = c_bar - confidence_i

			numerator_ = np.array([ min(r_bar_, 1) for r_bar_ in r_bar])
			denominator_ = np.array([ max(c_bar_, 0) for c_bar_ in c_bar ])
			d = np.nan_to_num(numerator_/denominator_)

			arm_ = np.argmax(d)
		else:
			print ('WARNING: Something is extremely wrong!!!!')
			arm_ = 0

		r, c = casino.play_arm(arm_)
		
		self.update(r, c, arm_)

	def get_number_arms_played(self):
		return self.k

	def get_id(self):
		return 'RBC-AIR'
