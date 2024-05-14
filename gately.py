
import numpy as np

def Gately(x,z,f):
	vis = {}
	Mis = {}
	atts = {}

	for i in range(0,x.shape[0]):
		vi_sample = z.copy()
		vi_sample[i] = x[i]
		vis[i] = (f(vi_sample.reshape(1,-1)) - f(z.reshape(1,-1)))
		
		mvi_sample  = x.copy()		
		mvi_sample[i] = z[i]
		Mis[i] = (f(x.reshape(1,-1)) - f(mvi_sample.reshape(1,-1)))

	for i in range(0,x.shape[0]):
		if vis[i] > Mis[i]:
			temp = vis[i].copy()
			vis[i] = Mis[i]
			Mis[i] = temp
	interaction_penalty = (f(x.reshape(1,-1)) - f(z.reshape(1,-1)) - sum(vis.values()))
	sum_individuals = sum(Mis.values()) - sum(vis.values())

	up = sum(list(Mis.values())) - (f(x.reshape(1,-1)) - f(z.reshape(1,-1))) 
	down = (f(x.reshape(1,-1)) - f(z.reshape(1,-1))) - sum(list(vis.values()))


	if down != 0:

		if (up/down) == -1:
			print("cant compute")
			return atts
		else:
			for i in range(0,x.shape[0]):
				att = vis[i] + ((np.abs(Mis[i]-vis[i])/(sum_individuals))*interaction_penalty)
				atts[i] = att
	else:
		for i in range(0,x.shape[0]):
			att = vis[i]
			atts[i] = att


	return atts

def Gately_classification(x,z,f):
	vis = {}
	Mis = {}
	atts = {}

	for i in range(0,x.shape[0]):
		vi_sample = z.copy()
		vi_sample[i] = x[i]
		vis[i] = (f(vi_sample.reshape(1,-1))[0][0] - f(z.reshape(1,-1))[0][0])
		
		mvi_sample  = x.copy()		
		mvi_sample[i] = z[i]
		Mis[i] = (f(x.reshape(1,-1))[0][0] - f(mvi_sample.reshape(1,-1))[0][0])

	for i in range(0,x.shape[0]):
		if vis[i] > Mis[i]:
			temp = vis[i].copy()
			vis[i] = Mis[i]
			Mis[i] = temp
	interaction_penalty = (f(x.reshape(1,-1))[0][0] - f(z.reshape(1,-1))[0][0] - sum(vis.values()))
	sum_individuals = sum(Mis.values()) - sum(vis.values())

	up = sum(list(Mis.values())) - (f(x.reshape(1,-1))[0][0] - f(z.reshape(1,-1))[0][0]) 
	down = (f(x.reshape(1,-1))[0][0] - f(z.reshape(1,-1))[0][0]) - sum(list(vis.values()))


	if down != 0:

		if (up/down) == -1:
			print("cant compute")
			return -1
		else:
			for i in range(0,x.shape[0]):
				att = vis[i] + ((np.abs(Mis[i]-vis[i])/(sum_individuals))*interaction_penalty)
				atts[i] = att
	else:
		for i in range(0,x.shape[0]):
			att = vis[i]
			atts[i] = att


	return atts


def Gately_distribution(x,z,f):
	vis = {}
	Mis = {}
	atts = {}
	distribution = []
	for i in range(0,x.shape[0]):
		vi_sample = z.copy()
		vi_sample[i] = x[i]
		vis[i] = (f(vi_sample.reshape(1,-1)) - f(z.reshape(1,-1)))
		
		mvi_sample  = x.copy()		
		mvi_sample[i] = z[i]
		Mis[i] = (f(x.reshape(1,-1)) - f(mvi_sample.reshape(1,-1)))

		distribution.append(vi_sample)
		
		distribution.append(mvi_sample)

	for i in range(0,x.shape[0]):
		if vis[i] > Mis[i]:
			temp = vis[i].copy()
			vis[i] = Mis[i]
			Mis[i] = temp


	interaction_penalty = (f(x.reshape(1,-1)) - f(z.reshape(1,-1)) - sum(vis.values()))
	sum_individuals = sum(Mis.values()) - sum(vis.values())

	up = sum(list(Mis.values())) - (f(x.reshape(1,-1)) - f(z.reshape(1,-1))) 
	down = (f(x.reshape(1,-1)) - f(z.reshape(1,-1))) - sum(list(vis.values()))


	if down != 0:

		if (up/down) == -1:
			print("cant compute")
			return atts
		else:
			for i in range(0,x.shape[0]):
				att = vis[i] + ((np.abs(Mis[i]-vis[i])/(sum_individuals))*interaction_penalty)
				atts[i] = att
	else:
		for i in range(0,x.shape[0]):
			att = vis[i]
			atts[i] = att


	return atts, np.asarray(distribution)



