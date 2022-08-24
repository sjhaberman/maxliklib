//Find log likelihood
//and corresponding gradient and Hessian for
//generalized IRT model.  The component involves
//the response variables specified in obsdata,
//the prior distribution specified in obsthetamaps, the
//the weights from obsweight, and the
//sampling procedure specified in obsthetas.  Adaptive
//quadrature information is in obsscale.
//The log likelihood is always given.  If order>0, the
//gradient is found.  If order=2, the Hessian is computed.
//If order=3, then the approximate Hessian is found.
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
// The choice to use is indicated by adapt, xselect shows the elements involved, tran shows the
// transformation.
struct adq
{
    bool adapt;
    xsel xselect;
    vecmat tran;
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
struct obsthetamap
{
    std::vector<thetamap>thetamaps;
};
struct obstheta
{
    std::vector<pwr> thetas;
};
void addsel(const int & , const xsel & , const f2v & , f2v & , const double & );
f2v irtm (const int & , const std::vector<dat> & ,
    const std::vector<thetamap> & ,adq & , const std::vector<pwr> & , const vec & );
f2v irtms (const int & order, const vec & obsweight,
    const std::vector<obs> & obsdata,
    const std::vector<obsthetamap> & obsthetamaps, std::vector<adq> & obsscale,
    const std::vector<obstheta> & obsthetas, const std::vector<xsel> & betasel, const vec & beta)
{
    f2v cresults, results;
    int i, p, q, n;
    vec gamma;
    p=beta.n_elem;
    n=obsdata.size();
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
//Enter individual observations and add.
    for(i=0;i<n;i++)
    {
        if(betasel[i].all)
        {
             q=p;
             gamma.resize(p);
             gamma=beta;
        }
        else
        {
             q=betasel[i].list.n_elem;
             gamma.resize(q);
             gamma=beta.elem(betasel[i].list);
        }      
        if(order>0)cresults.grad.set_size(q);
        if(order>1)cresults.hess.set_size(q,q);
        cresults=irtm(order,obsdata[i].data,obsthetamaps[i].thetamaps, obsscale[i],
            obsthetas[i].thetas,gamma);
        addsel(order,betasel[i],cresults,results,obsweight(i));
    }
    return results;
}

