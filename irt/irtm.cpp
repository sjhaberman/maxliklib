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
//Constant component of lambda.
struct lcomp
{
    int li;
    double value;
};
//Interaction of predictor and lambda.
struct lxcomp
{
    int li;
    int pi;
    double value;
};
//Interaction of predictor and lambda for predictor from another variable.
struct lxocomp
{
    int li;
    int pi;
    int ob;
    double value;
};
//Interaction of theta and lambda.
struct ltcomp
{
    int li;
    int th;
    double value;
};
//Interaction of beta and lambda.
struct lbcomp
{
    int li;
    int bi;
    double value;
};
//Interaction of beta and predictor with lambda.
struct lxbcomp
{
    int li;
    int pi;
    int bi;
    double value;
};
//Interaction of beta and predictor with lambda for predictor from another variable.
struct lxobcomp
{
    int li;
    int pi;
    int ob;
    int bi;
    double value;
};
//Interaction of beta and theta with lambda.
struct ltbcomp
{
    int li;
    int th;
    int bi;
    double value;
};
//Data structure.
struct dat
{
    resp y;
    vec x;
};    
//Specify a model.
//choice indicates transformation and type of model.
//dim is dimension of lambda;
//idim is dimension of integer response.
//ddim is dimension of double response.
//bdim is dimension of used beta elements.
//lcomps indicates constant components.
//lxcomps indicates components only dependent on the predictors.
//lxocomps indicates components only dependent on predictors from other variables.
//ltcomps indicates components only dependent on theta.
//lbcomps indicates components only dependent on parameters.
//lxbcomps indicates components dependent on predictors and parameters.
//lxobcomps indicates components dependent on predictors from other variables and parameters.
//ltbcompes indicates components dependent on theta and parameters.
//ithetas are integer elements from theta in response.
//dthetas are double elements from theta in response.
struct pattern
{
     model choice;
     int dim;
     int idim;
     int ddim;
     field<lcomp> lcomps;
     field<lxcomp>lxcomps;
     field<lxocomp>lxocomps;
     field<ltcomp>ltcomps;
     field<lbcomp>lbcomps;
     field<lxbcomp>lxbcomps;
     field<lxobcomp>lxobcomps;
     field<ltbcomp>ltbcomps;
     uvec ithetas;
     uvec dthetas;
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
f2v genresplik(const int & , const field<pattern> & , const uvec & , const field<uvec> & ,
    const resp & , const field<dat> &  , const field<xsel> & , const uvec & ,
    const vec & , const xsel & , const vec & );
f2v irtm (const int & order, const field<pattern> & patterns, 
    const uvec & patternnumber, const field<uvec> & selectobs, const field <pwr> & thetas,  
    const adq & scale, dovecmat & obsscale, const field<dat> & data,
    const field<xsel> & selectbeta, const uvec & betanumber, const vec & w, 
    const xsel & obssel, const vec &  beta)
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
         results=genresplik(order1, patterns, patternnumber, selectobs, dummy, 
            data, selectbeta, betanumber, w, obssel, beta);
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
        cresults(i)=genresplik(order1, patterns, patternnumber, selectobs, newthetas(i).theta, 
            data, selectbeta, betanumber, w, obssel, beta);
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

