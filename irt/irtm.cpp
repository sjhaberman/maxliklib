//Find log likelihood component
//and corresponding gradient and Hessian for
//generalized IRT model.  The component involves
//the response variables specified in data,
//the correspondence specified by thetamaps
//of the latent response and the predictors and of the latent response
//and the latent distribution, the prior distribution specified in prior, the
//adaptive quadrature information in adquad, and the model parameter beta.
//datasel selects responses to use.
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
maxf2v maxf2vvar(const int & order , const vec & y, const f2v & fy);
f2v irtc (const int & , const std::vector<dat> & ,
    const std::vector<thetamap> & , const xsel & , const resp & ,
    const vec &  beta);
f2v irtm (const int & order, const std::vector<dat> & data,
    const std::vector<thetamap> & thetamaps, const xsel & datasel, adq & scale,
    const std::vector<pwr> & thetas, const vec &  beta)
{
    f2v cresults, results;
    bool flag;
    double avelog,sumprob;
    int d,i,m,order1, q;
    pwr newtheta;    
    newtheta.theta.iresp.copy_size(thetas[0].theta.iresp);
    newtheta.theta.dresp.copy_size(thetas[0].theta.dresp);
    flag=scale.adapt;
    m=beta.n_elem;
    if(order==3)
    {
         order1=1;
    }
    else
    {
         order1=order;
    }
    q=thetas.size();
    vec prob(q),weights(q);
    std::vector<maxf2v>cloglik(q);
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
        if(d==0)flag=false;       
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
        if(order1>1)
        {
            results.hess.zeros();
            cresults.hess.set_size(m,m);
        }
    }
    results.value=0.0;
//Enter prior points and add.
    for(i=0;i<q;i++)
    {
        newtheta=adaptpwr(thetas[i],scale);
        weights(i)=newtheta.weight; 
        cloglik[i].locmax.resize(d);
        if(order>0)cloglik[i].grad.resize(d);
        if(order1>1)cloglik[i].hess.resize(d,d);
        if(scale.xselect.all)
        {
            cloglik[i].locmax=newtheta.theta.dresp;
        }
        else
        {
            cloglik[i].locmax=newtheta.theta.dresp.elem(scale.xselect.list);
        }  
        cresults=irtc(order1,data,thetamaps,datasel,newtheta.theta,beta);
        if(order1>1)cresults.hess=cresults.hess+cresults.grad*cresults.grad.t();
        cloglik[i]=maxf2vvar(order1,newtheta.theta.dresp,cresults);
        prob(i)=cresults.value;
    }
    avelog=mean(prob);
    prob=exp(prob-avelog)%weights;
    sumprob=sum(prob);    
    results.value=log(sumprob)+avelog;
    if(order>0)
    {
         prob=prob/sumprob;     
         for(i=0;i<q;i++)
         {
             results.grad=results.grad+prob(i)*cloglik[i].grad;
             if(order1>1)results.hess=results.hess+prob(i)*cloglik[i].hess;
         }
    }
    if(order1>1)results.hess=results.hess-results.grad*results.grad.t();
    if(order==3)results.hess=-results.grad*results.grad.t();
    if(flag)scale = rescale(scale, cloglik);
    return results;
}

