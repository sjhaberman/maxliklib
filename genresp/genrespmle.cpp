//Find maximum likelihood estimates for response model.
//See genresplik.cpp and nrv.cpp for definitions of structs
//used.
//Newton-Raphson approach is used if algorithm is N.
//Louis approach is used if algorithm is L.
//Conjugate gradient method is used if algorithm is C.
//Gradient ascent is used if algorithm is G.
#include<armadillo>
using namespace std;
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
//ddim is dimension of do
//bdim is dimension of used beta elements.
//lcomps indicates constant components.
//lxcomps indicates components only dependent on the predictors.
//lxocomps indicates components only dependent on predictors from other variables.
//ltcomps indicates components only dependent on theta.
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

maxf2v maxselect(const int &, const params & , const char & , const vec & ,
           const function<f2v(const int & , const vec & )> f);
f2v genresplik(const int & , const field<pattern> & , const uvec & , const field<uvec> & , 
    const resp & , const field<dat> &  , 
    const field<xsel> & ,const uvec & ,
    const vec & , const xsel & , const vec & );
maxf2v genrespmle(const int & order, const params & mparams,
    const char & algorithm, const field<pattern> & patterns, const uvec & patternnumber,
    const field<uvec> & selectobs, const resp & theta, const field<dat> & data,
    const field<xsel> & selectbeta, const uvec & betanumber, const vec & w, const xsel & obssel,
    const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    const function<f2v(const int & order,const vec & start)>f=
        [&patterns,&patternnumber,&selectobs,&theta,&data,
        &selectbeta,&betanumber,&w,&obssel]
        (const int & order,const vec & start)
        {return genresplik(order,patterns,patternnumber,selectobs,theta,
        data,selectbeta,betanumber, w,obssel,start);};
    results=maxselect(order, mparams, algorithm, start, f);
    return results;
}
