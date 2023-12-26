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
//Specify a model.
//choice is model distribution.
//o is constant vector.
//x is tranformation from beta elements used to lambda that does not involve theta.
//c is transformation from beta elements used and theta double elements  used to lambda.
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
int intsel(const xsel & , const int & );
ivec ivecsel(const xsel & , const ivec & );
vec vecsel(const xsel & , const vec & );
int sintsel(const xsel & , const int & );
int sivecsel(const xsel & , const ivec & );
int svecsel(const xsel & , const vec & );
//Components  for a single observation.
f2v irtm (const int & , const field<pattern> & , 
    const xsel & , const field<resp> & , const field <pwr> & ,  
    const adq & , dovecmat & , 
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const field<xsel> & , const xsel & ,
    const vec & , const xsel & , const vec &  );
//order is 0 if only values are found, 1 if gradient vectors are found, 2 if Hessian
//matrices found, and 3 if Louis approximation is used.
//patterns are possible response definitions.
//patternnumber is vector of pattern numbers.
//ipatno is number patno of patternnumber vector for observation i.
//selectobs(k) is data selection pattern k.
//selectobsno(i) is data selection pattern number of observation i.
//data(i) is responses for observation i.
//thetas(k) is latent vector pattern k.
//thetano(i) is theta pattern number for observation i.
//scale(o) is scale pattern o.
//scaleno(i) is scale pattern number for observation i.
//obsscale(i) is scaling for observation i.
//selectbeta gives selection pattern of beta parameters.
//selectbetano gives number of selection pattern for beta parameters.
//selbetano gives number of number of selection pattern for beta parameters.
//selectbeta gives selection pattern of beta parameters.
//selectbetano gives number of selection pattern for beta parameters.
//selbetano gives number of number of selection pattern for beta parameters.
//selectbetac gives selection pattern of beta parameters.
//selectbetacno gives number of selection pattern for beta parameters.
//selbetacno gives number of number of selection pattern for beta parameters.
//selectthetai gives selection pattern of beta parameters.
//selectthetaino gives number of selection pattern for beta parameters.
//selthetaino gives number of number of selection pattern for beta parameters.
//selectthetad gives selection pattern of beta parameters.
//selectthetadno gives number of selection pattern for beta parameters.
//selthetadno gives number of number of selection pattern for beta parameters.
//w is vector of item weights.
//wno is number of vector of item weights.
//obssel specifies pattern of items to use.
//obsselno is number of pattern of items to use.
//obsweight(i) is weight of observation i.
//datasel is selection of observations.
//betasel is selection pattern of the parameter beta.
//betaselno is the selection pattern of beta for observation i.
//beta is the parameter vector.
f2v irtms (const int & order, const field<pattern> & patterns, 
    const field<xsel> & patternnumber, const xsel & patno,
    const field<field<resp>> & data, const field<field <pwr>> & thetas, const xsel & thetano, 
    const field<adq> & scale, const xsel & scaleno, field<dovecmat> & obsscale, 
    const field<xsel> & selectbeta, const field<xsel> & selectbetano, const xsel & selbetano,
    const field<xsel> & selectbetac, const field<xsel> & selectbetacno,
    const xsel & selbetacno,
    const field<xsel> & selectthetai, const field<xsel> & selectthetaino,
    const xsel & selthetaino,
    const field<xsel> & selectthetad, const field<xsel> & selectthetadno,
    const xsel & selthetadno,
    const field<xsel> & selectthetac, const field<xsel> & selectthetacno,
    const xsel & selthetacno,
    const field<vec> & w, const xsel & wno, const field<xsel> & obssel, const xsel & obsselno, 
    const vec & obsweight, const xsel & datasel,
    const field<xsel> & betasel, const xsel & betaselno, const vec &  beta)
{
    f2v cresults, results;
    int i, ibetasel, ii, iobssel, ipat, iscale, iselbeta,
        iselbetac, iselthetai, iselthetad, iselthetac, itheta, iw, n, nn, p, q;
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
    nn=sintsel(datasel,n);
    if(nn==0) return results;
//Enter individual observations and add.
    for(ii=0;ii<nn;ii++)
    {
        i=intsel(datasel,ii);
        ipat=intsel(patno,i);
        itheta=intsel(thetano,i);
        iscale=intsel(scaleno,i);
        iselbeta=intsel(selbetano,i);
        iselbetac=intsel(selbetacno,i);
        iselthetai=intsel(selthetaino,i);
        iselthetad=intsel(selthetadno,i);
        iw=intsel(wno,i);
        iobssel=intsel(obsselno,i);
        ibetasel=intsel(betaselno,i);
        if(betasel(ibetasel).all)
        {
             q=p;
             gamma.set_size(p);
             gamma=beta;
        }
        else
        {
             q=betasel(ibetasel).list.n_elem;
             gamma.set_size(q);
             gamma=beta.elem(betasel(ibetasel).list);
        }     
        if(order>0)cresults.grad.set_size(q);
        if(order>1)cresults.hess.set_size(q,q);
        cresults=irtm(order, patterns, patternnumber(ipat), data(i),
             thetas(itheta), scale(iscale), obsscale(i),
             selectbeta,selectbetano(iselbeta),
             selectbetac,selectbetacno(iselbetac), 
             selectthetai,selectthetaino(iselthetai),
             selectthetad,selectthetadno(iselthetad),
             selectthetac,selectthetacno(iselthetac),
             w(iw),  obssel(iobssel), gamma);
        if(isnan(cresults.value))
        {
            results.value=datum::nan;
            if(order>0)results.grad.fill(datum::nan);
            if(order>1)results.hess.fill(datum::nan);
            return results;
        }
        addsel(order,betasel(ibetasel),cresults,results,obsweight(i));
    }
    return results;
}

