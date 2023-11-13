//Find log likelihood
//and corresponding gradient and Hessian for
//generalized IRT model.  The components are specified as in
//irtm.cpp. Adaptive
//quadrature information is in obsscale and scale.
//The log likelihood is always given.  If order>0, the
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
//Add in components.
void addsel(const int & , const xsel & , const f2v & , f2v & , const double & );
//Components  for a single observation.
f2v irtm (const int & , const field<pattern> & , const uvec & , const field<uvec> & , 
    const field <pwr> & , const adq & , dovecmat &,
    const field<dat> & , const field<xsel> & , const uvec & , const vec & , 
    const xsel & , const vec & );
//order is 0 if only values are found, 1 if gradient vectors are found, 2 if Hessian
//matrices found, and 3 if Louis approximation is used.
//patterns are possible response definitions.
//patternnumber(j) is vector of pattern numbers for form j.
//patno(i) is number j of patternnumber vector for observation i.
//selectobs(k) is data selection pattern k.
//selectobsno(i) is data selection pattern number of observation i.
//data(i) is responses for observation i.
//thetas(k) is latent vector pattern k.
//thetano(i) is theta pattern number for observation i.
//scale(o) is scale pattern o.
//scaleno(i) is scale pattern number for observation i.
//obsscale(i) is scaling for observation i.
//selectbeta(r) is selection pattern r of beta parameters.
//selectbetano(i) is selection pattern for observation i.
//betanumber(s) is pattern s for mapping of beta parameters to items.
//betanono(i) is pattern number  for observation i for mapping of beta parameters to items.
//w(t) is vector of item weights for pattern t.
//wno(i) is number t for vector of item weights.
//obssel(u) specifies pattern u of items to use.
//obsselno(i) is number  for observation i of pattern u of items to use.
//obsweight(i) is weight of observation i.
//datasel is selection of observations.
//betasel(v) is selection pattern v of the parameters beta.
//betaselno(i) is the selection pattern of beta for observation i.
//beta is the parameter vector.
f2v irtms (const int & order, const field<pattern> & patterns,
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
    const field<xsel> & betasel, const uvec & betaselno, const vec &  beta)
{
    f2v cresults, results;
    int i, ii, j, k, m, n, nn, o, p, q, r, s, t, u, v;
    vec gamma;
    p=beta.n_elem;
    results.value=0.0;
    if(order>0)
    {
        results.grad.set_size(p);
        
        results.grad.zeros();
    }
    if(order>1)
    {
        results.hess.set_size(p,p);
        
        results.hess.zeros();
    }
// Observations to use.
    n=data.n_elem;
    if(datasel.all)
    {
        nn=n;
    }
    else
    {
        nn=datasel.list.n_elem;
    }
    if(nn==0) return results;
//Enter individual observations and add.
    for(ii=0;ii<nn;ii++)
    {
        if(datasel.all)
        {
             i=ii;
        }
        else
        {
             i=datasel.list(ii);
        }
        j=patno(i);
        k=selectobsno(i);
        m=thetano(i);
        o=scaleno(i);
        r=selectbetano(i);
        s=betanono(i);
        t=wno(i);
        u=obsselno(i);
        v=betaselno(i);
        if(betasel(v).all)
        {
             q=p;
             gamma.set_size(p);
             gamma=beta;
        }
        else
        {
             q=betasel(v).list.n_elem;
             gamma.set_size(q);
             gamma=beta.elem(betasel(v).list);
        }      
        if(order>0)cresults.grad.set_size(q);
        if(order>1)cresults.hess.set_size(q,q);
        cresults=irtm(order, patterns, patternnumber(j), selectobs(k), 
             thetas(m),  scale(o), obsscale(i),
             data(i), selectbeta(r), betanumber(s), w(t),
             obssel(u), gamma);
        addsel(order,betasel(v),cresults,results,obsweight(i));
    }
    return results;
}

