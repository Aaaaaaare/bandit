class Ghostwriter:
	def __init__(self, name):
		self.code = 1
		if name != None:
			self.name = name
		else:
			self.name = 'Unnamed.txt'


	def write(self, string):
		try:
			with open(self.name, "a") as outfile:
				outfile.write(string + '\n')
		except IOError:
			print ('Could not open file')

	def write_(self, _obj):
		try:
			with open(self.name, "a") as outfile:
				if type(_obj) == dict:
					# for k, v in _obj.items():
					# 	if hasattr(v, '__iter__'):
					# 		outfile.write(k + '\n')
					# 		dumpclean(v)
					# 	else:
					# 		outfile.write('%s : %s\n' % (k, v))
					for x in _obj:
						outfile.write('{}: \t{}\n'.format(x, _obj[x]))
				elif type(_obj) == list:
					# for v in _obj:
					# 	if hasattr(v, '__iter__'):
					# 		dumpclean(v)
					# 	else:
					# 		outfile.write(v + '\n')
					for x in _obj:
						outfile.write(x + '\n')
				else:
					outfile.write(_obj + '\n')
		except IOError:
			print ('Could not open file')