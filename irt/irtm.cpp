//Find log likelihood component
//and corresponding gradient and Hessian for
//generalized IRT model.  The component involves
//the response variables specified in data,
//the correspondence specified by thetamaps
//of the latent response and the predictors and of the latent response
//and the latent distribution, the prior distribution specified in prior, the
//adaptive quadrature information in scale, and the model parameter beta.
//datasel selects responses to use.
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
// The choice to use is indicated by adapt, xselect shows the elements involved,
// step shows the step size,tran shows the
// transformation.   mparams and step are used by nrvn.cpp, and mult is the weight multiplier.

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
pwr adaptpwr(const pwr &, const adq & ,const rescale & );
f2v irtc (const int & , const vector<dat> & ,
    const vector<thetamap> & , const xsel & , const resp & ,
    const vec &  beta);
maxf2v nrvn(const int & , const double & , const params & , const vec & ,
    const function<f2v(const int &, const vec &)> );
f2v irtm (const int & order, const vector<dat> & data,
    const vector<thetamap> & thetamaps, const xsel & datasel,
    const adq & scale, const params & mparamsn, rescale & newscale,
    const vector<pwr> & thetas, const vec &  beta)
{
    f2v results;
    
    bool flag;
    double avelog,sumprob;
    int d,i,j,k,m,order1, order0=0, order2=2,q;
    vec  th;
    pwr newtheta;  
    resp theta; 
    auto f = [&order0,&scale,&data,&thetamaps,&datasel,&theta,&beta]
    (const int & order2, const vec & th) mutable
    {
         if(scale.xselect.all)
         {
              theta.dresp=th;
         }
         else
         {
              theta.dresp(scale.xselect.list)=th;
         }
         return irtc(order0,data,thetamaps,datasel,theta,beta);
    };
    maxf2v scaleset;
    newtheta.theta.iresp.copy_size(thetas[0].theta.iresp);
    newtheta.theta.dresp.copy_size(thetas[0].theta.dresp);
    flag=scale.adapt;
    m=beta.n_elem;
    if(order==3)
    {
         order1=1;
    }
    else
    {
         order1=order;
    }
    q=thetas.size();
    vecmat pv;   
    vector<f2v> cresults(q);
    vec prob(q),weights(q);
    if(scale.adapt)
    {
        
        d=newscale.tran.v.n_elem;
        if(d==0)
        {
             flag=false;
        }
        else
        {
             th.set_size(d);
             theta.iresp.copy_size(thetas[0].theta.iresp);
             theta.dresp.copy_size(thetas[0].theta.dresp);
             scaleset.grad.set_size(d);
             scaleset.locmax.set_size(d);
             scaleset.hess.set_size(d,d);
             scaleset=nrvn(order2, scale.step,mparamsn,newscale.tran.v,f);
             newscale.tran.v=scaleset.locmax;
             if((-scaleset.hess).is_sympd())
             {
                  newscale.tran.v=scaleset.locmax;
                  newscale.tran.m=inv_sympd(sqrtmat_sympd(-scaleset.hess));
                  newscale.mult=det(newscale.tran.m);
             }
        }          
    }
    if(order>0)
    {
        results.grad.set_size(m);
        results.grad.zeros();      
    }
    if(order>1)
    {
        results.hess.set_size(m,m);
        if(order1>1)
        {
            results.hess.zeros();       
        }
    }
    results.value=0.0;
//Enter prior points and add.
    for(i=0;i<q;i++)
    {
        if(order>0)cresults[i].grad.set_size(m);
        if(order1>1)cresults[i].hess.set_size(m,m);



        newtheta=adaptpwr(thetas[i],scale,newscale);
        weights(i)=newtheta.weight; 
        cresults[i]=irtc(order1,data,thetamaps,datasel,newtheta.theta,beta);
        if(order1>1)cresults[i].hess=cresults[i].hess+cresults[i].grad*cresults[i].grad.t();
        prob(i)=cresults[i].value;   
    }
    avelog=mean(prob);
    prob=prob-avelog*ones(q);
    prob=exp(prob)%weights;
    sumprob=sum(prob);  
    results.value=log(sumprob)+avelog;
    if(order>0)prob=prob/sumprob;
    if(order>0)
    {    
         for(i=0;i<q;i++)
         {
             results.grad=results.grad+prob(i)*cresults[i].grad;
             if(order1>1)results.hess=results.hess+prob(i)*cresults[i].hess;
         }
    }
    if(order1>1)results.hess=results.hess-results.grad*results.grad.t();
    if(order==3)results.hess=-results.grad*results.grad.t();
    return results;
}

