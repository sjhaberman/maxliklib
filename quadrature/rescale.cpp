//Rescale for adaptive quadrature transformation for integration of function on vector space.
//Transformation of points is loc+lt*point for the vector loc
//and the matrix lt.
#include<armadillo>
using namespace std;
using namespace arma;
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct vecmat
{
    vec v;
    mat m;
};
maxf2v quadmax(const vecmat & );
vecmat rescale(const vecmat & pv)
{ 
    vecmat result;
    int d,i,q;
    q=pv.v.n_elem;
    d=pv.m.n_rows;
    maxf2v quadfit;
    quadfit.locmax.set_size(d);
    quadfit.hess.set_size(d,d);
    quadfit.grad.set_size(d);
    result.v.set_size(d);
    result.m.set_size(d,d); 
    result.v.zeros();
    result.m=eye(d,d);      
    quadfit=quadmax(pv); 
    if(isnan(quadfit.max)) return result;
    result.v =quadfit.locmax;
    result.m=chol(inv_sympd(-quadfit.hess),"lower");
    return result;
}
