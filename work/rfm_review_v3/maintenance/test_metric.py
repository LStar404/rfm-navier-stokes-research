"""Original normalized H2 test Gram assembly without exploratory driver imports."""
from . import rfm_solver as s
np = s.np

def normalized_test_gram(problem,order,q):
    z,w=s.leggauss(q);z=(z+1)/2;w=w*problem.length/2
    b=s.polynomial_jet(s.Poly([0,0,1,-2,1]),z,problem.length)
    modes=[]
    for degree in range(order):
        p=s.polynomial_jet(s.Legendre.basis(degree,domain=[0,1]),z,problem.length)
        modes.append((b[0]*p[0],b[1]*p[0]+b[0]*p[1],
            b[2]*p[0]+2*b[1]*p[1]+b[0]*p[2]))
    v,f,d=np.asarray(modes).transpose(1,0,2)
    M=(v*w)@v.T;D=(d*w)@d.T;C=(d*w)@v.T
    # <Delta(v_i v_j), Delta(v_k v_l)>; retain transpose explicitly.
    G=np.kron(D,M)+np.kron(M,D)+np.kron(C,C.T)+np.kron(C.T,C)
    diagonal=np.sqrt(np.diag(G));G=G/diagonal[:,None]/diagonal[None,:]
    return (G+G.T)/2
