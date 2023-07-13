//Find maximum likelihood estimates for latent response model.
//The components involve
//the response variables specified in obsdata,
//the prior distribution specified in obsthetamaps, and the
//sampling procedure specified in obsthetas.  Adaptive quadrature
//information is in obsscale.
//The initial parameter estimate is start.
//Algorithm parameters are in mparams.  The algorithm is specified in 
//algorithm, with G for steepest ascent, C for conjugate gradient, N for
//Newton-Raphson, and L for Louis.    If order>0, the
//gradient is found.  If order=2, the Hessian is computed.
//If order=3, then the approximate Hessian is found.
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
// Combination of vector and lower triangular matrix for use in transformations.
struct vecmat
{
    vec v;
    mat m;
};
// Adaptive quadrature specifications.
// The choice to use is indicated by adapt, xselect shows the elements involved, step
// shows the step size, tran shows the
// transformation.
struct adq
{
    bool adapt;
    
    xsel xselect;
    double step;   
};
struct rescale
{
    double mult;
    vecmat tran;
};
struct pwr
{
    double weight;
    resp theta;
};
maxf2v conjgrad(const int &, const params & , const vec & ,
    const function<f2v(const int & , const vec & )> f);
maxf2v gradascent(const int &, const params & , const vec & ,
    const function<f2v(const int & , const vec & )> f);
maxf2v nrv(const int &, const params & , const vec & ,  
    const function<f2v(const int & , const vec & )> f);
f2v irtms (const int & , const vec & , const vector<vector<dat>> & ,
    const vector<vector<thetamap>> & ,
    const vector<xsel> & , const xsel & , const adq & , const params & ,  vector<rescale> &,
    const vector<vector<pwr>> & ,
    const vector<xsel> & ,const vec & );
maxf2v irtmle(const int & order, const params & mparams,
    const char & algorithm, const vec & obsweight,
    const vector<vector<dat>> & obsdata,
    const vector<vector<thetamap>> & obsthetamaps, const vector<xsel> & datasel,
    const xsel & obssel,  const adq & scale, const params & mparamsn, vector<rescale> & obsscale,
    const vector<vector<pwr>> & obsthetas, const vector<xsel> & betasel, const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    if(algorithm=='N'||algorithm=='L')results.hess.set_size(p,p);
    auto f=
        [&obsweight,&obsdata,&obsthetamaps,&datasel,
        &obssel,&scale,&mparamsn,&obsscale,&obsthetas,&betasel]
        (const int &order,const vec &start) mutable
        {return irtms(order,obsweight,obsdata,obsthetamaps,datasel,
        obssel,scale,mparamsn,obsscale,obsthetas,betasel,start);};
    if(algorithm=='N'||algorithm=='L')results=nrv(order, mparams, start, f);
    if(algorithm=='C')results=conjgrad(order, mparams, start, f);
    if(algorithm=='G')results=gradascent(order, mparams, start, f);
    return results;
}

