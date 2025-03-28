//Find log likelihood component
//and corresponding gradient and Hessian for
//generalized IRT model.  Input is as in genresplik.cpp except that a collection of
//theta responses and weights is used. They are contained in thetas.  Output is a field
//with members the struct pwrmaxf2v.  The output can be used for posteriors.
#include<armadillo>
using namespace std;
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
//Select elements of vector.  all indicates all elements.  list lists elements.
struct xsel
{
    bool all;
    uvec list;
};
//Specify a model.
//choice is model distribution.
//o is constant vector.
//x is tranformation from beta elements used to lamnbda
//that does not involve theta.
//c is transformation from beta elements used and theta double elements
//used to lambda.
struct pattern
{
    model choice;
    vec o;
    mat x;
    cube c;
};
// Weights and points for prior.
struct pwr
{
    double weight;
    double kernel;
    resp theta;
};
//Adaptive quadrature transformation.
struct dovecmat
{
    double s;
    vec v;
    mat m;
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
f2v genresplik(const int & , const field<pattern> & ,
    const xsel & , const field<resp> & , const resp & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const vec & , const xsel & , const vec & );
field<pwrf2v> irtmsave (const int & order, const field<pattern> & patterns, 
    const xsel & patternnumber, const field<resp> & data,
    const field <pwr> & thetas,
    dovecmat & obsscale,
    const field<xsel> & selectbeta, const xsel & selectbetano,
    const field<xsel> & selectbetac, const xsel & selectbetacno,
    const field<xsel> & selectthetai, const xsel & selectthetaino,
    const field<xsel> & selectthetad, const xsel & selectthetadno,
    const field<xsel> & selectthetac, const xsel & selectthetacno,
    const vec & w, const xsel & obssel, const vec &  beta)
{
    int dp, i, ip, m, q;
//Number of elements of parameter vector.
    m=beta.n_elem;
//Number of points.
    q=thetas.n_elem;
    field<pwrf2v> results(q);
    if(q<1)return results;
    pwr newtheta;
    dp=thetas(1).theta.dresp.n_elem;
    ip=thetas(1).theta.iresp.n_elem;
    newtheta.theta.iresp.set_size(ip);
    newtheta.theta.dresp.set_size(dp);
//Results for each prior point.  
    f2v cresults;
    if(order>0)cresults.grad.set_size(m);
    if(order>1)cresults.hess.set_size(m,m);
//Enter prior points.
    for(i=0;i<q;i++)
    {
        newtheta.theta.iresp=thetas(i).theta.iresp;
        newtheta.theta.dresp=obsscale.v+obsscale.m*thetas(i).theta.dresp;
        newtheta.weight=obsscale.s*thetas(i).weight;
        newtheta.kernel=thetas(i).kernel;
        results(i).weight=newtheta.weight;
        results(i).kernel=newtheta.kernel;
        results(i).theta.dresp.set_size(dp);
        results(i).theta.iresp.set_size(ip);
        results(i).theta.dresp=newtheta.theta.dresp;
        results(i).theta.iresp=newtheta.theta.iresp;
        cresults=genresplik(order, patterns, patternnumber, data, newtheta.theta,
            selectbeta, selectbetano,
            selectbetac, selectbetacno,
            selectthetai, selectthetaino,
            selectthetad, selectthetadno,
            selectthetac, selectthetacno,
            w, obssel, beta);
        results(i).value=cresults.value;
        if(order>0)
        {
            results(i).grad.set_size(m);
            results(i).grad=cresults.grad;
        }
        if(order>1)
        {
            results(i).hess.set_size(m,m);
            results(i).hess=cresults.hess;
        }
    }
    return results;
}

