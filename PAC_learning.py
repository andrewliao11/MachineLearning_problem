import argparse, seaborn
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from math import ceil, pow
from tqdm import tqdm

def get_multivariate_normal(args, size):
    MU = [args.mu_x,args.mu_y]
    SIGMA = [[pow(args.sigma_x,2), args.rho_xy*args.sigma_x*args.sigma_x], \
                [args.rho_xy*args.sigma_x*args.sigma_x, pow(args.sigma_y,2)]]
    return np.random.multivariate_normal(MU, SIGMA, int(size))

def get_unknown_concept(args):
    if args.concept_type == 'given':
	return [-0.5,2.5], [3,8]
    elif args.concept_type == 'rand':
	v = [args.mu_x+(np.random.rand(1)*2-1)-2, \
			args.mu_y+(np.random.rand(1)*2-1)-2]
	u = [args.mu_x+(np.random.rand(1)*2-1)+2, \
			args.mu_y+(np.random.rand(1)*2-1)+2]
	return v, u

def is_in_rectangle(s, v, u):
    if (s[0]>v[0]) and (s[0]<u[0]) and (s[1]>v[1]) and (s[1]<u[1]):
	return True
    else:
	return False

def greater_than_2eplison(args, v, u):
    err = 0
    N = pow((1.8595/args.eplison),2)
    S = get_multivariate_normal(args, N)
    for s in S:
	err += is_in_rectangle(s, v, u)+0
    if (1-err/N) < 3*args.eplison:
	print "P(c) < 3*eplison"
	return False
    else:
	plt.scatter(S[:,0], S[:,1], marker='o')
        plt.gca().add_patch(patches.Rectangle(v, u[0]-v[0], u[1]-v[1], fill=None, edgecolor='r'))
        plt.savefig('target_concept.png')
	plt.close()
	print "P(c) > 3*eplison"
	return True

def sample_data(args, v, u, verbose=True):

    m = ceil(4/args.eplison*np.log(4/args.delta))    
    if verbose:
    	print "choose %d samples" % int(m)
    S = get_multivariate_normal(args, m)
    label = np.full([int(m)], False, np.bool)
    for idx, s in enumerate(S):
	if is_in_rectangle(s, v, u):
	    label[idx] = True
    return S, label

def find_hs(data, label, verbose=True):

    init = True
    if verbose:
    	print "estimating hypothesis"
    for d, l in zip(data, label):
	if l and init:
	    ll_x = d[0]
	    ll_y = d[1]
	    ur_x = d[0]
            ur_y = d[1]
	    init = False
	if l:
	    if not is_in_rectangle(d, [ll_x,ll_y], [ur_x,ur_y]):
		if d[0]<ll_x: 
		    ll_x = d[0]
       	 	if d[1]<ll_y: 
		    ll_y = d[1]
        	if d[0]>ur_x: 
		    ur_x = d[0]
        	if d[1]>ur_y: 
		    ur_y = d[1]
		
    return [[ll_x, ll_y], [ur_x, ur_y]]

def test_generalization_error(args, v, u, hs, verbose=True):

    M_eplison = ceil(pow((19.453/args.eplison),2))
    if verbose:
    	print "choose %d sample for testing(empirical generalization error)" \
				% int(M_eplison)
    S = get_multivariate_normal(args, M_eplison)
    Rhs_err = 0
    for s in S:
	label = is_in_rectangle(s, v, u)
	pred = is_in_rectangle(s, hs[0], hs[1])
	if label != pred:
	    Rhs_err = Rhs_err+1
    Rhs_err = Rhs_err/M_eplison
    return Rhs_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAC learning')
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--eplison', type=float, default=0.1)
    parser.add_argument('--mu_x', type=float, default=2)
    parser.add_argument('--mu_y', type=float, default=3)
    parser.add_argument('--sigma_x', type=float, default=1)
    parser.add_argument('--sigma_y', type=float, default=3)
    parser.add_argument('--rho_xy', type=float, default=0.5)
    parser.add_argument('--concept_type', choices=['given', 'rand'], default='given')
    parser.add_argument('--verification', type=bool, default=False)
    args = parser.parse_args()
    print "Specification:"
    print(args)

    valid_concept = False
    print "finding suitable target concept..."
    while not valid_concept:
    	v, u = get_unknown_concept(args)
    	valid_concept = greater_than_2eplison(args, v, u)
    
    if args.verification:
	miss = 0
	for i in tqdm(range(int(10/args.delta))):
	    data, label = sample_data(args, v, u, verbose=False)
	    hs = find_hs(data, label, verbose=False)
	    err = test_generalization_error(args, v, u, hs, verbose=False)
	    print err
	    if err > args.eplison:
		miss += 1
	print "%d times of estimation with generation error larger than eplison" % miss

    else:
    	data, label = sample_data(args, v, u)
    	hs = find_hs(data, label)

    	# plot the estimated concept and target concept
    	plt.scatter(data[:,0], data[:,1], marker='o')
    	plt.gca().add_patch(patches.Rectangle(v, u[0]-v[0], u[1]-v[1], fill=None, edgecolor='r'))
    	plt.gca().add_patch(patches.Rectangle(hs[0], hs[1][0]-hs[0][0], hs[1][1]-hs[0][1], fill=None, edgecolor='g'))
    	plt.savefig('hs.png')
    	plt.close()
    	test_generalization_error(args, v, u, hs)
	print "R(hs) = %f" % Rhs_err
