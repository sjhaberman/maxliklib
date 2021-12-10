//Find log likelihood component
//and corresponding gradient and Hessian for
//generalized IRT model.  The component involves
//the response variables specified in irtdata,
//the prior distribution specified in prior, and the
//sampling procedure specified in pws.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
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
  ivec list;
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
//respno is response number.  If dep is false,
//offsets are for effect of theta.dresp on model parameter
//without consideration of beta and
//indeps is the cube for interaction of beta and theta.dresp.
struct thetamap
{
    bool dep;
    int respno;
    xsel drespcols;
    xsel irespcols;
    mat offsets;
    cube indeps;
};
struct pwr
{
    double weight;
    resp theta;
};
f2v irtc (const int & , const std::vector<dat> & ,
    const std::vector<thetamap> & , const resp & ,
    const vec &  beta);
f2v irtm (const int & order, const std::vector<dat> & data,
    const std::vector<thetamap> & thetamaps,
    const std::vector<pwr> & thetas, const vec &  beta)
{
    f2v cresults, results;
    int i,m,q;
    m=beta.n_elem;
    q=thetas.size();
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
        cresults=irtc(order,data,thetamaps,thetas[i].theta,beta);
        cresults.value=exp(cresults.value);
        if(order>1) cresults.hess=
            cresults.value*(cresults.hess+cresults.grad
                *cresults.grad.t());
        if(order>0) cresults.grad=cresults.value*cresults.grad;
        results.value=results.value+thetas[i].weight*cresults.value;
        if(order>0)results.grad=results.grad+thetas[i].weight*cresults.grad;
        if(order>1)results.hess=results.hess+thetas[i].weight*cresults.hess;
    }
    if(order>0)results.grad=results.grad/results.value;
    if(order>1)results.hess=results.hess/results.value-results.grad*results.grad.t();
    results.value=log(results.value);
    return results;
}

