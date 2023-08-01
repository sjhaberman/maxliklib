//Find components of log likelihood
//and corresponding gradient and Hessian for
//generalized IRT model.  The component involve
//the inidividual observations specified in obsdata and
//the prior distribution specified in obsthetamaps, and the
//sampling procedure specified in obsthetas.  Adaptive
//quadrature information is in scale and obsscale.
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
// The choice to use is indicated by adapt, xselect shows the elements involved, steo is the step
// size, tran shows the
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
struct pwr
{
    double weight;
    resp theta;
};

f2v irtm (const int & , const vector<dat> & ,
    const vector<thetamap> & , const xsel & , const adq & , const params & ,
    rescale & , const vector<pwr> & , const vec & );
vector<f2v> irtcomps (const int & order, 
    const vector<vector<dat>> & obsdata,
    const vector<vector<thetamap>> & obsthetamaps, const vector<xsel> & datasel,
    const xsel & obssel,  const adq & scale, vector<rescale> & obsscale,
    const vector<pwr> & thetas,
    const vector<xsel> & betasel, const vec & beta)
{
    vector<f2v> results;
    int i, ii, p, q, n, nn;
    n=obsdata.size();
    results.resize(n);
    params mparamsn;
    mparamsn.maxit=0;
    p=beta.n_elem;

    vec gamma;
    
// Observations to use.
    
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
             gamma.set_size(p);
             gamma=beta;
        }
        else
        {
             q=betasel[i].list.n_elem;
             gamma.set_size(q);
             gamma=beta.elem(betasel[i].list);
        }

        if(order>0)results[i].grad.set_size(q);
        if(order>1)results[i].hess.set_size(q,q);     
        results[i]=irtm(order,obsdata[i],obsthetamaps[i], datasel[i], scale, mparamsn, obsscale[i],
            thetas,gamma);
        
    }
    return results;
}

