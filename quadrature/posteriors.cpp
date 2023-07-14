//Find quadrature points and weights for posterior distributions of
//latent vector for generalized IRT model.  The input in irtms.cpp
//applies here except that order is not needed.
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
// Weights and points for prior.
struct pwr
{
    double weight;
    resp theta;
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
    double step;
};
struct rescale
{
    
    double mult;
    
    vecmat tran;
};
pwr adaptpwr(const pwr &, const adq & , const rescale &);
vecmat posterior (const vector<dat> & ,
    const vector<thetamap> & , const xsel & , const adq & , const rescale & ,
    const vector<pwr> & , const vec &  beta);
vector<vecmat> posteriors(const vector<vector<dat>> & obsdata,
    const vector<vector<thetamap>> & obsthetamaps, const vector<xsel> & datasel, 
    const xsel & obssel, const adq & scale,  const vector<rescale> & obsscale,
    const vector<pwr> & thetas, const vector<xsel> & betasel, const vec &  beta)
{
    vector<vecmat> results;
    int i,ii,n,nn,p,q,r,s;
    vec gamma;
    p=beta.n_elem;
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
    results.resize(n);
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
        r=thetas[0].theta.dresp.n_elem;
        s=thetas.size();      
        results[i].v.set_size(s);
        results[i].m.set_size(r,s);
        results[i]=posterior(obsdata[i],obsthetamaps[i],datasel[i],scale, 
             obsscale[i],thetas,gamma);
    }
    return results;
}

