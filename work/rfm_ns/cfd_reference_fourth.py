"""Fourth-order independent FD cavity reference, including wall closure.

For inward normal s and known psi_s(0)=d, the boundary identity is
psi_ss(0)=[8psi(h)-3psi(2h)+(8/9)psi(3h)-(1/8)psi(4h)]/h^2
             -(25/6)d/h + O(h^4).
It is exact for polynomials through degree five. No RFM data is used.
"""
from cfd_reference import *
from math import factorial

def derivative_matrix(n,derivative):
    h=1/n;rr=[];cc=[];vv=[]
    for i in range(1,n):
        if i in (1,n-1) and derivative==2:
            ids=np.arange(6) if i==1 else np.arange(n-5,n+1)
        else:
            start=max(0,min(i-2,n-4));ids=np.arange(start,start+5)
        delta=ids-i;A=np.array([delta.astype(float)**k for k in range(len(ids))])
        b=np.zeros(len(ids));b[derivative]=factorial(derivative)
        weights=np.linalg.solve(A,b)/h**derivative
        rr.extend([i-1]*len(ids));cc.extend(ids.tolist());vv.extend(weights.tolist())
    return sparse.csr_matrix((vv,(rr,cc)),shape=(n-1,n+1))

class CavityFourth(CavityFD):
    order=4
    def __init__(self,n,Re):
        self.n=n;self.m=n-1;self.d=(n-1)**2;self.h=1/n;self.Re=float(Re);self.nu=1/self.Re
        m=n-1;S=sparse.csr_matrix((np.ones(m),(np.arange(m),np.arange(1,n))),shape=(m,n+1))
        dx=derivative_matrix(n,1);dd=derivative_matrix(n,2)
        self.P=sparse.kron(S.T,S.T,format='csr')
        self.Xall=sparse.kron(dx,S,format='csr');self.Yall=sparse.kron(S,dx,format='csr')
        Lall=sparse.kron(dd,S,format='csr')+sparse.kron(S,dd,format='csr')
        self.L=Lall@self.P;self.Dx=self.Xall@self.P;self.Dy=self.Yall@self.P
        self.I=sparse.eye(self.d,format='csr')
        rr=[];cc=[];vv=[];self.bfull=np.zeros((n+1)**2);coefs=np.array([8.,-3.,8/9,-1/8])
        ind=lambda i,j:(i-1)*m+j-1
        full=lambda i,j:i*(n+1)+j
        for j in range(1,n):
            for offset,coef in enumerate(coefs,1):
                rr.extend([full(0,j),full(n,j)]);cc.extend([ind(offset,j),ind(n-offset,j)])
                vv.extend([-coef/self.h**2,-coef/self.h**2])
        for i in range(1,n):
            for offset,coef in enumerate(coefs,1):
                rr.extend([full(i,0),full(i,n)]);cc.extend([ind(i,offset),ind(i,n-offset)])
                vv.extend([-coef/self.h**2,-coef/self.h**2])
            self.bfull[full(i,n)]=-(25/6)*lid(i*self.h)/self.h
        self.Bfull=sparse.csr_matrix((vv,(rr,cc)),shape=((n+1)**2,self.d))
        self.BL=Lall@self.Bfull;self.BX=self.Xall@self.Bfull;self.BY=self.Yall@self.Bfull
        self.bL=Lall@self.bfull;self.bX=self.Xall@self.bfull;self.bY=self.Yall@self.bfull
    def reconstruct(self,z):
        n=self.n;m=self.m;pi=z[:self.d];wi=z[self.d:]
        psi=(self.P@pi).reshape(n+1,n+1)
        w=(self.P@wi+self.Bfull@pi+self.bfull).reshape(n+1,n+1)
        u=np.zeros_like(psi);v=np.zeros_like(psi)
        u[1:-1,1:-1]=(self.Dy@pi).reshape(m,m);v[1:-1,1:-1]=-(self.Dx@pi).reshape(m,m)
        x=np.linspace(0,1,n+1);u[:,-1]=lid(x)
        return dict(x=x,y=x.copy(),psi=psi,omega=w,u=u,v=v)
    def divergence(self,fields):return self.Xall@fields['u'].ravel()+self.Yall@fields['v'].ravel()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--grids',default='32,64,128,256');parser.add_argument('--res',default='100,400');args=parser.parse_args()
    grids=[int(x) for x in args.grids.split(',')];res=[int(x) for x in args.res.split(',')]
    summary_path=ROOT/'cfd_reference4_summary.json'
    records=[] if not summary_path.exists() else [r for r in json.loads(summary_path.read_text())['rows'] if r['Re'] not in res]
    final={}
    for Re in res:
        previous=None;cumulative=0.
        for n in grids:
            path=ROOT/f'cfd_reference4_cavity_Re{Re}_n{n}.npz'
            if path.exists():
                dat=np.load(path);fields={key:dat[key] for key in ('x','y','psi','omega','u','v')};rec=json.loads(str(dat['record']))
            else:
                if previous is None:
                    current=None;continuation=[]
                    for rr in ([10,25,50,100] if Re==100 else [10,25,50,100,200,400]):
                        current,cr=solve_cavity(n,rr,initial=current,_model_class=CavityFourth);continuation.append(cr)
                    fields,rec=current,continuation[-1];rec['continuation_seconds']=sum(cr['total_seconds'] for cr in continuation)
                    rec['continuation_Re']=[cr['Re'] for cr in continuation]
                    rec['initialization']='zero state followed by Reynolds continuation'
                else:
                    fields,rec=solve_cavity(n,Re,initial=previous,_model_class=CavityFourth)
                    rec['initialization']='prolongation from previous coarser grid'
                np.savez_compressed(path,**fields,record=json.dumps(rec))
            cumulative+=rec.get('continuation_seconds',rec['total_seconds']);rec['cumulative_reference_solve_seconds']=cumulative
            if previous is not None:rec['difference_from_previous_grid']=comparison(previous,fields,nq=160)
            records.append(rec);previous=fields
            print('SOLVED4',Re,n,rec['total_seconds'],rec['residual_rms'],rec['top_wall_max_abs_vorticity_201'],rec.get('difference_from_previous_grid'),flush=True)
            (ROOT/'cfd_reference4_summary.json').write_text(json.dumps(dict(sources=SOURCES,rows=records),indent=2))
        final[Re]=fields
    print('DONE',flush=True)
if __name__=='__main__':main()
