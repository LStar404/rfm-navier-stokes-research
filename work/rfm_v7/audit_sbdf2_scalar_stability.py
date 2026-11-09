"""Independent 80-decimal-digit roots for the derived SBDF2 scalar boundary."""
from decimal import Decimal as D, localcontext
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def add(a,b): return a[0]+b[0],a[1]+b[1]
def mul(a,b): return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def scale(a,c): return a[0]*c,a[1]*c
def norm(a): return (a[0]*a[0]+a[1]*a[1]).sqrt()


def roots(d,v):
    a=D('1.5')+d
    b=(-D(2),-D(2)*v)
    c=(D('.5'),v)
    discriminant=add(mul(b,b),scale(c,-4*a))
    magnitude=norm(discriminant)
    x=((magnitude+discriminant[0])/2).sqrt()
    y=((magnitude-discriminant[0])/2).sqrt()
    if discriminant[1]<0: y=-y
    result=[scale(add(scale(b,-1),scale((x,y),sign)),1/(2*a)) for sign in (1,-1)]
    errors=[]
    for z in result:
        value=add(add(scale(mul(z,z),a),mul(b,z)),c)
        denom=a*norm(z)**2+norm(b)*norm(z)+norm(c)
        errors.append(norm(value)/denom)
    return result,errors


def main():
    target=HERE/'audit_sbdf2_scalar_stability.json'
    if target.exists(): raise FileExistsError(target)
    records=[]
    with localcontext() as ctx:
        ctx.prec=80
        tolerance=D('1e-60')
        for ds in ('1e-10','1e-8','1e-6','1e-4','.01','.1','1','10','100','10000'):
            d=D(ds)
            q=(1+d)*(2/D(3).sqrt()*(d*(1+d)).sqrt()-d)
            assert 0<q<(1+d)*(2+d)
            for fs in ('0','.25','.5','.99','1','1.01','2','10'):
                factor=D(fs)
                v=(q*factor).sqrt()
                values,residuals=roots(d,v)
                radius=max(map(norm,values))
                if factor<1: passed=radius<1-tolerance
                elif factor>1: passed=radius>1+tolerance
                else: passed=abs(radius-1)<tolerance and min(map(norm,values))<1-tolerance
                assert passed and max(residuals)<D('1e-65')
                records.append(dict(d=ds,v=str(v),squared_frequency_fraction_of_boundary=fs,
                    root_moduli=list(map(lambda z:str(norm(z)),values)),
                    maximum_relative_polynomial_residual=str(max(residuals)),passed=bool(passed)))
        for vs in ('0','1e-6','.01','.1','1','10'):
            v=D(vs)
            values,residuals=roots(D(0),v)
            radius=max(map(norm,values))
            passed=abs(radius-1)<tolerance if v==0 else radius>1+tolerance
            assert passed and max(residuals)<D('1e-65')
            records.append(dict(d='0',v=vs,root_moduli=list(map(lambda z:str(norm(z)),values)),
                maximum_relative_polynomial_residual=str(max(residuals)),passed=bool(passed)))
    result=dict(status='passed',precision_decimal_digits=80,checks=len(records),records=records,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        theory_sha256=hashlib.sha256((HERE/'theory_sbdf2_scalar_stability.txt').read_bytes()).hexdigest(),
        scope='Numerical check of the independently derived scalar characteristic roots, not a nonlinear NS proof.')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(result['status'],result['checks'])


if __name__=='__main__': main()
