//Find maximum likelihood estimates for latent response model.
//The components involve
//the response variables specified in irtdata,
//the prior distribution specified in thetamaps, and the
//sampling procedure specified in pws. This version does
//not use adaptive quadrature.
#include<armadillo>
using namespace arma;
using namespace std::placeholders;
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
struct thetamap
{
    bool dep;
    int respno;
    xsel drespcols;
    xsel irespcols;
    uvec indcols;
    uvec thetas;
};
struct pwr
{
    double weight;
    resp theta;
};
struct obs
{
    std::vector<dat> data;
};
maxf2v conjgrad(const int &, const params & , const vec & ,
           const std::function<f2v(const int & , const vec & )> f);
maxf2v gradascent(const int &, const params & , const vec & ,
           const std::function<f2v(const int & , const vec & )> f);
maxf2v nrv(const int &, const params & , const vec & ,
           const std::function<f2v(const int & , const vec & )> f);
f2v irtms (const int & , const vec & , const std::vector<obs> & , const std::vector<thetamap> & , const std::vector<pwr> & , const vec & );
maxf2v irtmle(const int & order, const params & mparams,
    const char & algorithm, const vec & obsweight,
    const std::vector<obs> & obsdata, const std::vector<thetamap> & thetamaps,
    const std::vector<pwr> & thetas, const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    if(algorithm=='N'||algorithm=='L')results.hess.set_size(p,p);
    auto f=std::bind(irtms,_1,obsweight,obsdata,thetamaps,thetas,_2);
    if(algorithm=='N'||algorithm=='L')results=nrv(order, mparams, start, f);
    if(algorithm=='C')results=conjgrad(order, mparams, start, f);
    if(algorithm=='G')results=gradascent(order, mparams, start, f);
    return results;
}

