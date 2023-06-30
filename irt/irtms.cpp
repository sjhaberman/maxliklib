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
//datasel selects how to decompose individual responses.
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
    double mult;
    xsel xselect;
    vecmat tran;
};
struct pwr
{
    double weight;
    resp theta;
};
void addsel(const int & , const xsel & , const f2v & , f2v & , const double & );
f2v irtm (const int & , const vector<dat> & ,
    const vector<thetamap> & , const xsel & , adq & , const vector<pwr> & , const vec & );
f2v irtms (const int & order, const vec & obsweight,
    const vector<vector<dat>> & obsdata,
    const vector<vector<thetamap>> & obsthetamaps, const vector<xsel> & datasel,
    const xsel & obssel,  vector<adq> & obsscale,
    const vector<vector<pwr>> & obsthetas,
    const vector<xsel> & betasel, const vec & beta)
{
    f2v cresults, results;
    int i, ii, p, q, n, nn;
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
    n=obsdata.size();
    if(obssel.all)
    {
        nn=n;
    }
    else
    {
        nn=obssel.list.size();
    }
    if(nn==0) return results;
//Enter individual observations and add.
    for(ii=0;ii<nn;ii++)
    {
        if(obssel.all)
        {
             i=ii;
        }
        else
        {
             i=obssel.list(ii);
        }
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
        cresults=irtm(order,obsdata[i],obsthetamaps[i], datasel[i], obsscale[i],
            obsthetas[i],gamma);
        addsel(order,betasel[i],cresults,results,obsweight(i));
    }
    return results;
}

