//Find maximum likelihood estimates for latent response model.
//The components are specified as in
//irtm.cpp. Adaptive
//quadrature information is in obsscale
//Always order>0, and the
//gradient is found.  If order=2, the Hessian is computed.
//If order=3, then the approximate Hessian is found.
//datasel selects individual responses in an observation.
//obssel selects which responses to use.
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
//Parameters for function maximization.
struct params
{
    bool print;
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
maxf2v maxselect(const int &, const params & , const char & , const vec & ,  
    const function<f2v(const int & , const vec & )> f);
f2v irtms (const int & , const field<pattern> & ,
    const field<uvec> & , const uvec &  ,
    const field<field<uvec>> & , const uvec & ,
    const field<field <pwr>> & , const uvec & , 
    const field<adq> & , const uvec & , 
    field<dovecmat> & , 
    const field<field<dat>> & ,
    const field<field<xsel>> & , const uvec & , 
    const field<uvec> & , const uvec & ,
    const field<vec> & , const uvec & , const field<xsel> & , const uvec & ,
    const vec & , const xsel & , 
    const field<xsel> & , const uvec & , const vec &  );
maxf2v irtmle(const int & order, const params & mparams,
    const char & algorithm, 
    const field<pattern> & patterns,
    const field<uvec> & patternnumber, const uvec &  patno,
    const field<field<uvec>> & selectobs, const uvec & selectobsno,
    const field<field <pwr>> & thetas, const uvec & thetano, 
    const field<adq> & scale, const uvec & scaleno, 
    field<dovecmat> & obsscale, 
    const field<field<dat>> & data,
    const field<field<xsel>> & selectbeta, const uvec & selectbetano, 
    const field<uvec> & betanumber, const uvec & betanono,
    const field<vec> & w, const uvec & wno, const field<xsel> & obssel, const uvec & obsselno,
    const vec & obsweight, const xsel & datasel, 
    const field<xsel> & betasel, const uvec & betaselno, const vec &  start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    if(algorithm=='N'||algorithm=='L')results.hess.set_size(p,p);
    const function<f2v(const int & order,const vec & start)> f=
        [&patterns,
             &patternnumber, &patno,
             &selectobs, &selectobsno,
             &thetas, &thetano, 
             &scale, &scaleno, 
             &obsscale, 
             &data,
             &selectbeta, &selectbetano, 
             &betanumber, &betanono,
             &w, &wno, &obssel, &obsselno,
             &obsweight, &datasel, 
             &betasel, &betaselno](const int & order,const vec & start) mutable
             {return irtms(order, patterns,
             patternnumber, patno,
             selectobs, selectobsno,
             thetas, thetano, 
             scale, scaleno, 
             obsscale, 
             data,
             selectbeta, selectbetano, 
             betanumber, betanono,
             w, wno, obssel, obsselno,
             obsweight, datasel, 
             betasel, betaselno, start);};
    results=maxselect(order, mparams, algorithm, start, f);
    return results;
}

