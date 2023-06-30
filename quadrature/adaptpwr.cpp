// Adaptive quadrature transformation for integration of function on vector space.
// oldtheta is unchanged if adq.adapt is false or adq.xselect.all is false and
// adq.xselect.list has no elements.  Otherwise, oldtheta.iresp is unchanged, the vector
// point of elements of oldtheta.dresp specified by adq.xselect are changed to
// scale.tran.v+scale.tran.m*point, and
// oldtheta.weight is multiplied by the determinant scale.mult of adq.tran.m.
#include<armadillo>
using namespace arma;
struct resp
{
  ivec iresp;
  vec dresp;
};
struct xsel
{
  bool all;
  uvec list;
};
struct pwr
{
    double weight;
    resp theta;
};
// Combination of vector and lower triangular matrix for use in transformations.
struct vecmat
{
    vec v;
    mat m;
};
// Adaptive quadrature specifications.
// The choice to use is indicated by adapt, xselect shows the elements involved, tran shows the
// transformation.
struct adq
{
    bool adapt;
    double mult;
    xsel xselect;
    vecmat tran;
};
pwr adaptpwr(const pwr & oldtheta, const adq & scale)
{ 
    int i,p,n;
    pwr results;
    results.theta.iresp.copy_size(oldtheta.theta.iresp);
    results.theta.dresp.copy_size(oldtheta.theta.dresp);
    results.theta.iresp=oldtheta.theta.iresp;
    results.theta.dresp=oldtheta.theta.dresp;
    results.weight=oldtheta.weight;
    if(!scale.adapt)return results;
    if(scale.xselect.all)
    {
        results.theta.dresp=scale.tran.v+scale.tran.m*results.theta.dresp;
    }
    else
    {
        p=scale.xselect.list.n_elem;
        if(p==0)return results;
        results.theta.dresp.elem(scale.xselect.list)
             =scale.tran.v+scale.tran.m*results.theta.dresp.elem(scale.xselect.list);       
    }
    results.weight=results.weight*scale.mult;
    return results;
}
