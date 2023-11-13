// Adaptive quadrature transformation for integration of function on vector space.
// thetas are unchanged if scale.adapt is false or scale.linselect.all and scale.quadselect.all
// are false and scale.linselect.list and scale.quadselect.list have no elements.
// Otherwise, oldtheta(i).theta.iresp is always unchanged, and adaptpwr(i).dresp is
// A*thetas(i).theta.dresp+b. of thetas
// point of elements of oldtheta(i).dresp specified by scale.xselect are changed to
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
struct xselv
{
  bool all;
  umat list;
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
    xsel linselect;
    xselv quadselect;
};
struct rescale
{
    double mult;
    vecmat tran;
};
field<pwr> adaptpwr(const field<pwr> & thetas, const adq & scale, const rescale & newscale)
{ 
    int i,m, n, p, q;
    m=thetas.n_elem;
    field<pwr> results(m);
    for(i=0;i<m; i++)
    {
        results(i).theta.iresp.copy_size(thetas(1).theta.iresp);
        results(i).theta.dresp.copy_size(thetas(1).theta.dresp);
        results(i).theta.iresp=thetas(i).theta.iresp;
        if(!scale.adapt)
        {
             results(i).theta.dresp=thetas(i).theta.dresp;
             results(i).weight=thetas(i).weight;
        }
        else
        {
            results(i).theta.dresp=newscale.tran.v+newscale.tran.m*thetas(i).theta.dresp;
            results(i).weight=thetas(i).weight*newscale.mult;
        }
    }
    return results;
}
