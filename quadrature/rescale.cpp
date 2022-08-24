//Rescale for adaptive quadrature transformation for integration of function on vector space.
//Transformation of points is loc+lt*point for the vector loc
//and the lower triangular
//matrix lt.
#include<armadillo>
using namespace arma;
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct resp
{
  ivec iresp;
  vec dresp;
};
struct xsel
{
  bool all;
  ivec list;
};
struct pwr
{
    double weight;
    resp theta;
};
struct vecmat
{
    vec v;
    mat m;
};
struct adq
{
    bool adapt;
    xsel xselect;
    vecmat tran;
};
maxf2v quadmax(const std::vector<maxf2v> & );
adq rescale(const adq & scale, const std::vector<maxf2v> & functdat)
{ 
    adq result;
    int d,i,q;
    maxf2v quadfit;
    result.adapt=scale.adapt;
    result.xselect.all=scale.xselect.all;
    d=functdat[0].locmax.size();
    result.xselect.list.copy_size(scale.xselect.list);
    result.xselect.list=scale.xselect.list;
    result.tran.v.copy_size(scale.tran.v);
    result.tran.v=scale.tran.v;
    result.tran.m.copy_size(scale.tran.m);
    result.tran.m=scale.tran.m;
    if(!result.adapt) return result; 
    quadfit.locmax.resize(d);
    quadfit.hess.resize(d,d);
    quadfit.grad.resize(d);     
    quadfit=quadmax(functdat);    
    result.tran.v=quadfit.locmax;
    result.tran.m=chol(inv_sympd(-quadfit.hess),"lower");
    return result;
}
