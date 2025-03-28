//Find maximum likelihood estimates for latent response model.
//The arguments are specified as in
//irtm.cpp and irtms.cpp. Adaptive
//quadrature information is in obsscale
//Always order>0, and the
//gradient is found.  If order=2, the Hessian is computed.
//If order=3, then the approximate Hessian is found.
//The arguments for maximization are specified as in maxselect,
//although f is defined
//in irtmle.cpp.
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
//Select elements of vector.  all indicates all elements.
//list lists elements.
struct xsel
{
    bool all;
    uvec list;
};
//Select elements of matrix.  all indicates all elements.
//list lists elements in columns.
struct xselv
{
    bool all;
    umat list;
};
//Specify a model.
//choice is model distribution.
//o is constant vector.
//x is tranformation from beta elements used for lambda that does
//not involve theta.
//c is transformation from beta elements used and theta double
//elements  used for lambda.
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
// The choice to use is indicated by adapt,
// linselect shows the elements involved.
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
maxf2v maxselect(const int &, const params & , const char & ,
    const vec & ,
    const function<f2v(const int & , const vec & )> f);
f2v irtms (const int & , const field<pattern> & , 
    const field<xsel> & , const xsel &  ,
    const field<field<resp>> & , const field<field<pwr>> & , const xsel & ,
    const field<adq> & , const xsel & , field<dovecmat> & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<vec> & , const xsel & , const field<xsel> & , const xsel & ,
    const vec & , const xsel & , 
    const field<xsel> & , const xsel & , const vec &  );
maxf2v irtmle(const int & order, const params & mparams,
    const char & algorithm, const field<pattern> & patterns, 
    const field<xsel> & patternnumber, const xsel & patno,
    const field<field<resp>> & data, const field<field <pwr>> & thetas,
    const xsel & thetano,
    const field<adq> & scale, const xsel & scaleno,
    field<dovecmat> & obsscale,
    const field<xsel> & selectbeta, const field<xsel> & selectbetano,
    const xsel & selbetano,
    const field<xsel> & selectbetac, const field<xsel> & selectbetacno,
    const xsel & selbetacno,
    const field<xsel> & selectthetai, const field<xsel> & selectthetaino,
    const xsel & selthetaino,
    const field<xsel> & selectthetad, const field<xsel> & selectthetadno,
    const xsel & selthetadno,
    const field<xsel> & selectthetac, const field<xsel> & selectthetacno,
    const xsel & selthetacno,
    const field<vec> & w, const xsel & wno, const field<xsel> & obssel,
    const xsel & obsselno,
    const vec & obsweight, const xsel & datasel,
    const field<xsel> & betasel, const xsel & betaselno,
    const vec &  start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    if(algorithm=='N'||algorithm=='L')results.hess.set_size(p,p);
    const function<f2v(const int & order,const vec & start)> f=
        [ &patterns, 
        &patternnumber, &patno,
        &data, &thetas, &thetano,
        &scale, &scaleno, &obsscale,
        &selectbeta, &selectbetano, &selbetano,
        &selectbetac, &selectbetacno, &selbetacno,
        &selectthetai, &selectthetaino, &selthetaino,
        &selectthetad, &selectthetadno, &selthetadno,
        &selectthetac, &selectthetacno, &selthetacno,
        &w, &wno,  &obssel, &obsselno,
        &obsweight, &datasel,
        &betasel, &betaselno](const int & order,const vec & start) mutable
        {return irtms(order, patterns,
        patternnumber, patno,
        data, thetas, thetano,
        scale, scaleno, obsscale,
        selectbeta, selectbetano, selbetano,
        selectbetac, selectbetacno, selbetacno,
        selectthetai, selectthetaino, selthetaino,
        selectthetad, selectthetadno, selthetadno,
        selectthetac, selectthetacno, selthetacno,
        w, wno,  obssel, obsselno,
        obsweight, datasel,
        betasel, betaselno, start);};
    results=maxselect(order, mparams, algorithm, start, f);
    return results;
}

