//Find quadrature points and weights for posterior distribution of
//latent vector for generalized IRT model.  The input in irtm.cpp
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
    vecmat tran;
};
pwr adaptpwr(const pwr &, const adq &);
f2v irtc (const int & , const vector<dat> & ,
    const vector<thetamap> & , const xsel & , const resp & ,
    const vec &  beta);
vecmat posterior(const vector<dat> & data,
    const vector<thetamap> & thetamaps, const xsel & datasel, adq & scale,
    const vector<pwr> & thetas, const vec &  beta)
{
    vecmat results;
    double avelog,sumprob;
    int d,i,j,k,m,order=0, q;
    pwr newtheta;    
    newtheta.theta.iresp.copy_size(thetas[0].theta.iresp);
    newtheta.theta.dresp.copy_size(thetas[0].theta.dresp);
    m=beta.n_elem;
    q=thetas.size();
    k=thetas[0].theta.dresp.n_elem;
    mat points(k,q);   
    vector<f2v> cresults(q);
    vec prob(q),weights(q);
    results.v.set_size(q);
    results.m.set_size(k,q);
    for(i=0;i<q;i++)
    {
        newtheta=adaptpwr(thetas[i],scale);
        weights(i)=newtheta.weight;
        points.col(i)=newtheta.theta.dresp; 
        cresults[i]=irtc(order,data,thetamaps,datasel,newtheta.theta,beta);
        prob(i)=cresults[i].value;
    }
    avelog=mean(prob);
    prob=exp(prob-avelog)%weights;
    sumprob=sum(prob);    
    prob=prob/sumprob;     
    results.v=prob;
    results.m=points;
    return results;
}

