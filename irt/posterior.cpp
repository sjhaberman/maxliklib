//Find quadrature points and weights for posterior distribution of
//latent vector for generalized IRT model.  The input in irtm.cpp
//applies here except that order and scale are not needed.
#include<armadillo>
using namespace std;
using namespace arma;
struct resp
{
  ivec iresp;
  vec dresp;
};
// Weights and points for prior and posterior.
struct pwr
{
    double weight;
    double kernel;
    resp theta;
};
struct pwrf2v
{
    double weight;
    double kernel;
    resp theta;
    double value;
    vec grad;
    mat hess;   
};
field<pwr> posterior(const field<pwrf2v> & irtcomps)
{
    int i, q;
    q=irtcomps.n_elem;
    field<pwr>results(q);
    double avelog,sumprob;
    vec prob(q),weights(q);
    for(i=0;i<q;i++)
    {
        weights(i)=irtcomps(i).weight/irtcomps(i).kernel;
        results(i).theta.iresp.copy_size(irtcomps(i).theta.iresp);
        results(i).theta.dresp.copy_size(irtcomps(i).theta.dresp);
        results(i).theta.iresp=irtcomps(i).theta.iresp;
        results(i).theta.dresp=irtcomps(i).theta.dresp; 
        results(i).kernel=irtcomps(i).kernel;
        prob(i)=irtcomps(i).value;
    }
    avelog=mean(prob);
    prob=prob-avelog*ones(q);
    
    prob=exp(prob)%weights;
    sumprob=sum(prob);    
    prob=prob/sumprob; 
    for(i=0;i<q;i++)results(i).weight=prob(i)*results(i).kernel;
    return results;
}

