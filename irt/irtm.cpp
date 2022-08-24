//Find log likelihood component
//and corresponding gradient and Hessian for
//generalized IRT model.  The component involves
//the response variables specified in data,
//the correspondence specified by thetamaps
//of the latent response and the predictors and of the latent response
//and the latent distribution, the prior distribution specified in prior, the
//adaptive quadrature information in adquad, and the model parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct model
{
  char type;
  char transform;
};
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
struct dat
{
    model choice;
    double weight;
    resp dep;
    vec offset;
    mat indep;
    xsel xselect;
};
//dep indicates if the dependent response is obtained
//from theta. drespcols gives the members of
//theta.dresp used for the response, and
//irespcols gives the members of
//theta.dresp used for the response.
//If dep is false,
//offsets are for effect of theta.dresp on model parameter
//without consideration of beta and
//indeps is the cube for interaction of beta and theta.dresp.
struct thetamap
{
    bool dep;
    xsel drespcols;
    xsel irespcols;
    mat offsets;
    cube indeps;
};
// Weights and points for prior.
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
    xsel xselect;
    vecmat tran;
};
pwr adaptpwr(const pwr &, const adq &);
adq rescale(const adq & , const std::vector<maxf2v> & );
f2v irtc (const int & , const std::vector<dat> & ,
    const std::vector<thetamap> & , const resp & ,
    const vec &  beta);
f2v irtm (const int & order, const std::vector<dat> & data,
    const std::vector<thetamap> & thetamaps, adq & scale,
    const std::vector<pwr> & thetas, const vec &  beta)
{
    f2v cresults, results;
    bool flag;
    double sumprob;
    int d,i,m,q;
    pwr newtheta;    
    newtheta.theta.iresp.copy_size(thetas[0].theta.iresp);
    newtheta.theta.dresp.copy_size(thetas[0].theta.dresp);
    flag=scale.adapt;
    m=beta.n_elem;
    q=thetas.size();
    vec prob(q);
    std::vector<maxf2v>cloglik;
    if(scale.adapt)
    {
        if(scale.xselect.all)
        {
             d=thetas[0].theta.dresp.n_elem;
        }
        else
        {
             d=scale.xselect.list.n_elem;
        }
        if(d>0)
        {
             cloglik.resize(q);
        }
        else
        {
             flag=false;
        }  
    }
    if(order>0)
    {
        results.grad.set_size(m);
        results.grad.zeros();
        cresults.grad.set_size(m);
    }
    if(order>1)
    {
        results.hess.set_size(m,m);
        results.hess.zeros();
        cresults.hess.set_size(m,m);
    }
    results.value=0.0;
//Enter prior points and add.
    for(i=0;i<q;i++)
    {
        newtheta=adaptpwr(thetas[i],scale); 
        if(flag)
        {
            cloglik[i].locmax.resize(d);
            if(scale.xselect.all)
            {
                 cloglik[i].locmax=newtheta.theta.dresp;
            }
            else
            {
                 cloglik[i].locmax=newtheta.theta.dresp.elem(scale.xselect.list);
            }
        }   
        cresults=irtc(order,data,thetamaps,newtheta.theta,beta);
        if(flag)cloglik[i].max=cresults.value;
        cresults.value=exp(cresults.value);
        prob(i)=cresults.value*newtheta.weight;
        if(order>1)cresults.hess=cresults.hess+cresults.grad*cresults.grad.t();
        if(order>0)results.grad=results.grad+prob(i)*cresults.grad;
        if(order>1)results.hess=results.hess+prob(i)*cresults.hess;
    }
    sumprob=sum(prob);
    if(order>0)results.grad=results.grad/sumprob;
    if(order>1)results.hess=results.hess/sumprob-results.grad*results.grad.t();
    results.value=log(sumprob);
    if(scale.adapt)scale = rescale(scale, cloglik);
    return results;
}

