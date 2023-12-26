//Find log likelihood component
//and corresponding gradient and Hessian for
//generalized IRT model.  Input is as in genresplik.cpp except that a collection of
//theta responses and weights is used. They are contained in thetas.
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
//Select elements of matrix.  all indicates all elements.  list lists elements in columns.
struct xselv
{
    bool all;
    umat list;
};
//Specify a model.
//choice is model distribution.
//o is constant vector.
//x is tranformation from beta elements used to lamnbda that does not involve theta.
//c is transformation from beta elements used and theta double elements  used to lambda.
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
// Adaptive quadrature specifications.
// The choice to use is indicated by adapt, linselect shows the elements involved.
// quadselect shows the quadratic elements involved.
struct adq
{
    bool adapt;
    xsel linselect;
    xselv quadselect;
};
//Adaptive quadrature transformation.
struct dovecmat
{
    double s;
    vec v;
    mat m;
};
//Find rescaling.
dovecmat fitquad(const field<f2v> & , const field<pwr> & , 
    const adq & , dovecmat & );
f2v genresplik(const int & , const field<pattern> & ,
    const xsel & , const field<resp> & , const resp & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const vec & , const xsel & , const vec & );
f2v irtm (const int & order, const field<pattern> & patterns, 
    const xsel & patternnumber, const field<resp> & data, const field <pwr> & thetas,  
    const adq & scale, dovecmat & obsscale, 
    const field<xsel> & selectbeta, const xsel & selectbetano,
    const field<xsel> & selectbetac, const xsel & selectbetacno,
    const field<xsel> & selectthetai, const xsel & selectthetaino,
    const field<xsel> & selectthetad, const xsel & selectthetadno,
    const field<xsel> & selectthetac, const xsel & selectthetacno,
    const vec & w, const xsel & obssel, const vec &  beta)
{
    f2v results;
    resp dummy;
    double avelog,sumprob;
    int i, m, order1, q;
//Number of elements of parameter vector.
    m=beta.n_elem;
    //Order adjustment for Louis method.
    if(order==3)
    {
         order1=1;
    }
    else
    {
         order1=order;
    }
    if(order>0)
    {
        results.grad.set_size(m);
        results.grad.zeros();      
    }
    if(order>1)
    {
        results.hess.set_size(m,m);
        if(order1>1)
        {
            results.hess.zeros();       
        }
    }
    q=thetas.n_elem;
    if(q<1)
    {
         results=genresplik(order1, patterns, patternnumber, data, dummy,
            selectbeta, selectbetano,
            selectbetac, selectbetacno,
            selectthetai, selectthetaino,
            selectthetad, selectthetadno,
            selectthetac, selectthetacno,
            w, obssel, beta);
         return results;
    }
    field<pwr> newthetas(q);
    for(i=0;i<q;i++)
    {
         newthetas(i).theta.iresp.copy_size(thetas(1).theta.iresp);
         newthetas(i).theta.dresp.copy_size(thetas(1).theta.dresp);
         newthetas(i).theta.iresp=thetas(i).theta.iresp;
         newthetas(i).theta.dresp=obsscale.v+obsscale.m*thetas(i).theta.dresp;
         newthetas(i).weight=obsscale.s*thetas(i).weight;
         newthetas(i).kernel=thetas(i).kernel;
    }

//Results for each prior point.  
    field<f2v> cresults(q);
    vec prob(q),weights(q);
    results.value=0.0;
//Enter prior points and add.
    for(i=0;i<q;i++)
    {
        if(order>0)cresults(i).grad.set_size(m);
        if(order1>1)cresults(i).hess.set_size(m,m);
        weights(i)=newthetas(i).weight/newthetas(i).kernel;
        cresults(i)=genresplik(order1, patterns, patternnumber, data, newthetas(i).theta,
            selectbeta, selectbetano,
            selectbetac, selectbetacno,
            selectthetai, selectthetaino,
            selectthetad, selectthetadno,
            selectthetac, selectthetacno,
            w, obssel, beta);
        if(isnan(cresults(i).value))
        {
            results.value=datum::nan;
            if(order>0)results.grad.fill(datum::nan);
            if(order>1)results.hess.fill(datum::nan);
            return results;
        }
        if(order1>1)cresults(i).hess=cresults(i).hess+cresults(i).grad*cresults(i).grad.t();
        prob(i)=cresults(i).value;
    }
    avelog=mean(prob);
    prob=prob-avelog*ones(q);
    prob=exp(prob)%weights;
    sumprob=sum(prob);
    if(order>0)prob=prob/sumprob;
    results.value=log(sumprob)+avelog;
    if(order>0)
    {        
         for(i=0;i<q;i++)
         {
             results.grad=results.grad+prob(i)*cresults(i).grad;
             if(order1>1)results.hess=results.hess+prob(i)*cresults(i).hess;
         }
    }
    if(order1>1)results.hess=results.hess-results.grad*results.grad.t();
    if(order==3)results.hess=-results.grad*results.grad.t();
    if(scale.adapt)obsscale=fitquad(cresults, newthetas, scale, obsscale);

    return results;
}

