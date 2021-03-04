//Log likelihood component and its gradient and hessian matrix
//for multivariate normal model with vector response y of dimension
//d and parameter vector beta with d(d+3)/2 elements.
//The first d elements of beta provide a location shift v and the next elements
//are unpacked with unpack.cpp to yield a d by d lower triangular matrix m with positive
//diagonal elements. The density f(y) is det(m)g(v+my), where g is the density of the standard
//multivariate normal distribution of dimension d.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct vecmat
{
    vec v;
    mat m;
};
struct resp
{
  ivec iresp;
  vec dresp;
};
vec pack(const vecmat & u);
vecmat unpack(const int &, const vec &);
f2v normalv(const int & order, const resp & y, const vec & beta)
{
    double d;
    int i,i1,j,j1,k,k1,n;
    vec a,z;
    vecmat b,dec;
    f2v results;
    if(order>0) results.grad.set_size(beta.n_elem);
    if(order>1) results.hess.set_size(beta.n_elem,beta.n_elem);
    n=y.dresp.n_elem;
    a.set_size(n);
    z.set_size(n);
    dec.v.set_size(n);
    dec=unpack(n,beta);
    z=dec.v+dec.m*y.dresp;
    a=diagvec(dec.m);
    if(min(a)<=0.0)
    {
        results.value=datum::nan;
        if(order>0) results.grad.fill(datum::nan);
        if(order>1) results.hess.fill(datum::nan);
        return results;
    }
    results.value=0.0;
    for(i=0;i<n;i++)
    {
        results.value=results.value+log(a(i)*normpdf(z(i)));
    }
    if(order>0)
    {
        b.v.set_size(n);
        b.m.set_size(n,n);
        b.v=-z;
        for(i=0;i<n;i++)
        {
            b.m(i,i)=1.0/a(i)-z(i)*y.dresp(i);
            if(i>0)
            {
                for(j=0;j<i;j++)
                {
                    b.m(i,j)=-z(i)*y.dresp(j);
                }
            }
        }
        results.grad=pack(b);
    }
    if(order>1)
    {
        k=n;
        for(i=0;i<n;i++)
        {
            results.hess(i,i)=-1.0;
            for(j=0;j<=i;j++)
            {
                results.hess(i,k)=-y.dresp(j);
                results.hess(k,i)=results.hess(i,k);
                k=k+1;
            }
        }
        for(i=0;i<n;i++)
        {
            for(j=0;j<=i;j++)
            {
                k=n+j+(i+1)*i/2;
                for(j1=0;j1<=j;j1++)
                {
                    k1=n+j1+i*(i+1)/2;
                    results.hess(k,k1)=-y.dresp(j)*y.dresp(j1);
                    if(j1<j)
                    {
                        results.hess(k1,k)=results.hess(k,k1);
                    }
                    if(i==j&&j1==i)
                    {
                        results.hess(k,k)=results.hess(k,k)-1.0/(a(i)*a(i));
                    }
                    k1=k1+1;
                }
                k=k+1;
            }
        }
    }
    return results;
}
