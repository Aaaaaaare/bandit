from players import player, kl_ucb
import numpy as np 


# =====================================================
#
# The kl-ucb algorith with a modification for the cost
#
# =====================================================
class kl_ucb_alpha(kl_ucb):
	def __init__(self, num_arms, budget, params=None):
		# Start with a defined number of arms.
		# It will think it is always not infinity?, so It will make as many 
		# arms as requested.
		is_infinity = False
		super().__init__(num_arms, budget, is_infinity)

		# For the new reward management.
		self.a = 1.0
		self.b = 1.0

		self.real_reward = 0
		self.real_cost = 0

		# To handle cold start of newly added arms
		# queue has the index of the arms with 0 pulls
		self.queue = np.where(self.pulls == 0)[0]

	# It needs to behave like the normal kl-ucb, but with
	# a modification of the reward.
	def play(self, casino):
		# Handle cold start. Not all arms tested yet.
		if len(self.queue > 0):  # self.cold_start ???
			arm_ = self.missing_arm()
		# If all arms have been played at least once, proceed:
		else:
			for a_ in range(self.num_arms):
				self.kl[a_] = self.get_max_kl(a_)
			arm_ = np.argmax(self.kl)

		# This will always be False
		if self.is_infinity:
			print ('MISTAAAAAAAAAAAAAAAAAAAAAAAAKE')
			r, c = self.play_masked_arm(casino, arm_)
		else:
			r, c = casino.play_arm(arm_)
		
		# It needs to behave like the normal kl-ucb, but with
		# a modification of the reward.
		self.real_reward += r
		self.real_cost += c
		r_plus = ((self.a *r - self.b * c) + self.b )/(self.a + self.b)

		self.update(r_plus, c, arm_)


	# Adds an arm at the end of the list
	def add_arm(self):
		self.num_arms = self.num_arms + 1
		self.reward = np.append(self.reward, 0.0)
		self.costs = np.append(self.costs, 0.0)
		self.pulls = np.append(self.pulls, 0)

		self.kl = np.append(self.kl, 0.0)

		# We will need to play it,
		# so we add it to the queue
		# Is the arm number self.num_arms, so it index is that -1
		self.cold_start = True
		self.queue = np.append(self.queue, self.num_arms - 1)

	def is_new_arm_better():
		# If the last one beats all
		np.argmax(self.reward/self.costs) == self.num_arms - 1

	def get_id(self):
		return 'kl-ucb-alpha'

	def missing_arm(self):
		# take the next element in the queue
		a_ = self.queue[0]
		# remove the element from the queue
		self.queue = np.delete(self.queue, 0)
		# return the index of the arm to play
		return a_

	def is_an_arm_missing(self):
		if len(self.queue) > 0:
			return True
		return False

	def set_budget(self, val):
		self.budget = val

	def get_prize():
		return self.real_reward, self.real_cost