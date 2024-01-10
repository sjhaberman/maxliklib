//Find quadrature points and weights for posterior distributions of
//latent vector for generalized IRT model.  The difference from posterior.cpp
//is just that fields of fields are involved.
#include<armadillo>
using namespace std;
using namespace arma;
struct resp
{
  ivec iresp;
  vec dresp;
};
// Weights and points for prior.
struct pwr
{
    double weight;
    double kernel;
    resp theta;
};
//Quadrature weights and point, function value, gradient, and Hessian.
struct pwrf2v
{
    double weight;
    double kernel;
    resp theta;
    double value;
    vec grad;
    mat hess;   
};
field<pwr> posterior(const field<pwrf2v> & );
field<field<pwr>> posteriors(const field<field<pwrf2v>> & irtcomps)
{
    
    int i,j, n,p;
    n=irtcomps.n_elem;
    field<field<pwr>> results(n);
    for(i=0;i<n;i++)
    {
        p=irtcomps(i).n_elem;
        results(i).set_size(p);
        for(j=0;j<p;j++)
        {
            results(i)(j).theta.dresp.copy_size(irtcomps(i)(j).theta.dresp);
            results(i)(j).theta.iresp.copy_size(irtcomps(i)(j).theta.iresp);
        }
        results(i)=posterior(irtcomps(i));
    }
    return results;
}

