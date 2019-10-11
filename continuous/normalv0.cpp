//Log likelihood component
//for multivariate normal model with vector response y of dimension
//d and parameter vector beta with d(d+3)/2 elements.
//The first d elements of beta provide a location shift v and the next elements
//are unpacked with unpack.cpp to yield a d by d lower triangular matrix m with positive
//diagonal elements. The density f(y) is det(m)g(v+my), where g is the density of the standard
//multivariate normal distribution of dimension d.
//


#include<armadillo>
using namespace arma;
using namespace std;
struct fd0bv
{
    double value;
    
    
    bool fin;
};

struct vecmat
{
    vec v;
    mat m;
};

vecmat unpack(int,vec);
fd0bv normalv0(vec y,vec beta)
{
    double d;
    int i,j;
    vec a,z;
    vecmat b,dec;
    fd0bv results;
    dec=unpack(y.n_elem,beta);
    d=double(y.n_elem);
    z=dec.v+dec.m*y;
    a=diagvec(dec.m);
    if(min(a)<=0.0)
    {
        results.fin=false;
        return results;
    }
    results.fin=true;
    results.value=-0.5*d*log(2.0*datum::pi)-0.5*dot(z,z)+sum(log(a));
    
    
    
    
    
    
    
     
    
    return results;
}
