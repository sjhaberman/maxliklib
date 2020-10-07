//Log likelihood component and its gradient
//for multivariate normal model with vector response y of dimension
//d and parameter vector beta with d(d+3)/2 elements.
//The first d elements of beta provide a location shift v and the next elements
//are unpacked with unpack.cpp to yield a d by d lower triangular matrix m with positive
//diagonal elements. The density f(y) is det(m)g(v+my), where g is the density of the standard
//multivariate normal distribution of dimension d.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
struct vecmat
{
    vec v;
    mat m;
};
vec pack(vecmat & u);
vecmat unpack(int &,vec &);
f1v normalv1(vec & y,vec & beta)
{
    double d;
    int i,j,n;
    vec a,z;
    vecmat b,dec;
    f1v results;
    results.grad.set_size(beta.n_elem);
    n=y.n_elem;
    dec=unpack(n,beta);
    z=dec.v+dec.m*y;
    a=diagvec(dec.m);
    if(min(a)<=0.0)
    {
        results.value=datum::nan;
        results.grad.fill(datum::nan);
        return results;
    }
    results.value=-0.5*dot(z,z)+sum(log(a));
    b.v=-z;
    b.m.set_size(n,n);
    for(i=0;i<n;i++)
    {
        b.m(i,i)=1.0/a(i)-z(i)*y(i);
        if(i>0)
        {
            for(j=0;j<i;j++)
            {
                b.m(i,j)=-z(i)*y(j);
            }
        }
    }
    results.grad=pack(b);
    return results;
}
