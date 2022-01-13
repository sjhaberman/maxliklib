//Find log likelihood
//and corresponding gradient and Hessian for
//generalized IRT model.  The component involves
//the response variables specified in obsdata,
//the prior distribution specified in thetamaps, the
//the weights from obsweight, and the
//sampling procedure specified in obsthetas.
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
f2v irtm (const int & , const std::vector<dat> & ,
    const std::vector<thetamap> & ,const std::vector<pwr> & , const vec & );
f2v irtms (const int & order, const vec & obsweight,
    const std::vector<obs> & obsdata,
    const std::vector<obsthetamap> & obsthetamaps,
    const std::vector<obstheta> & obsthetas, const vec & beta)
{
    f2v cresults, results;
    int i, p, n;
    p=beta.n_elem;
    n=obsdata.size();
    results.value=0.0;
    if(order>0)
    {
        results.grad.set_size(p);
        cresults.grad.set_size(p);
        results.grad.zeros();
    }
    if(order>1)
    {
        results.hess.set_size(p,p);
        cresults.hess.set_size(p,p);
        results.hess.zeros();
    }
//Enter individual observations and add.
    for(i=0;i<n;i++)
    {
        cresults=irtm(order,obsdata[i].data,obsthetamaps[i].thetamaps,
            obsthetas[i].thetas,beta);
        results.value=results.value+obsweight(i)*cresults.value;
        if(order>0)results.grad=results.grad+obsweight(i)*cresults.grad;
        if(order>1)results.hess=results.hess+obsweight(i)*cresults.hess;
    }
    return results;
}

