//Log likelihood and its gradient and Hessian
//for response model with response y and predictor vector x.
//Weight w is used.
//Model codes are found in genresp.cpp.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.  If order is 2,
//then the function, gradient, and Hessian are returned.   If order is 3,
//then the Hessian is replaced by the approximate Hessian
//of the Louis method.
//patterns indicates full specification of models used for responses.
//patternnumber connects data to model pattern.
//selectobs connects responses for use in pattern.
//theta provides variable replacement values for selected elements of data.
//data indicates data used.
//selectbeta connects beta elements to patterns.
//betanumber connects data to selectbeta elements.
//obssel indicates which observations to consider.
//beta is parameter vector.
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
//all is for selection of all elements (true) or some or no elements (false).
//If all is false, list indicates element numbers to use.
struct xsel
{
  bool all;
  uvec list;
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
//ddim is dimension of do
//bdim is dimension of used beta elements.
//lcomps indicates constant components.
//lxcomps indicates components only dependent on the predictors.
//lxocomps indicates components only dependent on predictors from other variables.
//ltcomps indicates components only dependent on theta.
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
f2v genresp(const int & , const model &, const resp &, const vec &);
void addsel(const int & , const xsel & , const f2v & , f2v & , const double & );
f2v linsel(const int & , const f2v & , const mat & );
vec vecsel(const xsel & , const vec & );
f2v genresplik(const int & order, const field<pattern> & patterns, const uvec & patternnumber,
    const field<uvec> & selectobs, const resp & theta, const field<dat> & data,
    const field<xsel> & selectbeta, const uvec & betanumber, const vec & w, const xsel & obssel,
    const vec & beta)
{
//Generic real value.
    double z;
//Model parameter.
    vec gamma,lambda;
//Matrix of all interactions of lambda and beta.
    mat indepx;
//Observation contributions to log likelihood and gradient and Hessian.
    f2v obsresults, linresults;
//Log likelihood, gradient, and Hessian.
    f2v results;
//Counters and sizes.
    int h, hh, hhh, i, ii, j, jj, jjj, jjjj, k, kk,  m, n, nn, order0 = 0, 
         order1, p, pp, q, r, rr, rrr, s, t, tt, ttt, u, uu, uuu, v, vv;
//Response.
    resp response;
    results.value = 0.0;
//Number of elements of beta.
    p=beta.n_elem;
//Number of observations.
    n=data.n_elem;
//Set up results elements.
    if(order>0)results.grad.zeros(p);
    if(order>1)results.hess.zeros(p,p);
//Order specification if Louis approach used.
    order1=order;
    if(order>2)order1=1;
//Select observations.
    if(obssel.all)
    {
         nn=n;
    }
    else
    {
        nn=obssel.list.size();
    }
//If nothing selected, nothing to do.
    if(nn==0) return results;
//Cycle through observations.
    for (ii=0;ii<nn;ii++)
    {
        if(obssel.all)
        {
             i=ii;
        }
        else
        {
             i=obssel.list(ii);
        }
//Pattern to use.
        j=patternnumber(i);
//Beta selection to use
        jj=betanumber(i);
//pp is lambda dimension.
//q is number of elements of lcomps.
//r is number of elements of lxcomps.
//rr is number of elements of lxocomps.
//rrr is number of elements of ltcomps.
//s is number of elements of lbcomps.
//t is number of elements of lxbcomps.
//tt is number of elements of lxobcomps.
//ttt is number of elements of ltbcomps.
//u is number of elements of beta to use.
//v is dimension of integer part of response.
//vv is dimension of double part of response.
        pp=patterns(j).dim;
        q=patterns(j).lcomps.n_elem;
        r=patterns(j).lxcomps.n_elem;
        rr=patterns(j).lxocomps.n_elem;
        rrr=patterns(j).ltcomps.n_elem;
        s=patterns(j).lbcomps.n_elem;
        t=patterns(j).lxbcomps.n_elem;
        tt=patterns(j).lxobcomps.n_elem;
        ttt=patterns(j).ltbcomps.n_elem;
        uu=patterns(j).ithetas.n_elem;
        uuu=patterns(j).dthetas.n_elem;
        v=patterns(j).idim;
        vv=patterns(j).ddim;
        if(selectbeta(jj).all)
        {
             u=p;
        }
        else
        {
             u=selectbeta(jj).list.n_elem;
        }
        if(u>0)
        {
             gamma.set_size(u);
             gamma=vecsel(selectbeta(jj),beta);
        }
        lambda.set_size(pp);
        lambda.zeros();
        response.iresp.set_size(v);
        response.dresp.set_size(vv);
        if(u>0)
        {
             indepx.set_size(pp,u);
             indepx.zeros();
        }

        if(q>0)
        {
             for(k=0;k<q;k++)
             {
                  kk=patterns(j).lcomps(k).li;
                  lambda(kk)=patterns(j).lcomps(k).value;
             }
        }
        if(r>0)
        {
             for(k=0;k<r;k++)
             {
                  kk=patterns(j).lxcomps(k).li;
                  h=patterns(j).lxcomps(k).pi;
                  lambda(kk)=lambda(kk)+patterns(j).lxcomps(k).value*data(i).x(h);
             }
        }
        if(rr>0)
        {
             for(k=0;k<rr;k++)
             {
                  kk=patterns(j).lxocomps(k).li;
                  h=patterns(j).lxocomps(k).pi;
                  jjj=patterns(j).lxocomps(k).ob;
                  jjjj=selectobs(i)(jjj);
                  lambda(kk)=lambda(kk)+patterns(j).lxcomps(k).value*data(jjjj).x(h);
             }
        }
        if(rrr>0)
        {
             for(k=0;k<rrr;k++)
             {
                  kk=patterns(j).ltcomps(k).li;
                  h=patterns(j).ltcomps(k).th;
                  lambda(kk)=lambda(kk)+patterns(j).ltcomps(k).value*theta.dresp(h);
             }
        }
        if(s>0)
        {
             for(k=0;k<s;k++)
             {
                  kk=patterns(j).lbcomps(k).li;
                  h=patterns(j).lbcomps(k).bi;
                  z=patterns(j).lbcomps(k).value;
                  lambda(kk)=lambda(kk)+z*gamma(h);
                  indepx(kk,h)=indepx(kk,h)+z;
             }
        }
        if(t>0)
        {
             for(k=0;k<t;k++)
             {
                  kk=patterns(j).lxbcomps(k).li;
                  hh=patterns(j).lxbcomps(k).pi;
                  h=patterns(j).lxbcomps(k).bi;
                  z=patterns(j).lxbcomps(k).value*data(i).x(hh);
                  lambda(kk)=lambda(kk)+z*gamma(h);
                  indepx(kk,h)=indepx(kk,h)+z;
             }
        }
        if(tt>0)
        {
             for(k=0;k<tt;k++)
             {
                  kk=patterns(j).lxobcomps(k).li;
                  hh=patterns(j).lxobcomps(k).pi;
                  h=patterns(j).lxobcomps(k).bi;
                  jjj=patterns(j).lxobcomps(k).ob;
                  jjjj=selectobs(i)(jjj);
                  z=patterns(j).lxobcomps(k).value*data(jjjj).x(hh);
                  lambda(kk)=lambda(kk)+z*gamma(h);
                  indepx(kk,h)=indepx(kk,h)+z;
             }
        }
        if(ttt>0)
        {
             for(k=0;k<ttt;k++)
             {
                  kk=patterns(j).ltbcomps(k).li;
                  hh=patterns(j).ltbcomps(k).th;
                  h=patterns(j).ltbcomps(k).bi;
                  z=patterns(j).ltbcomps(k).value*theta.dresp(hh);
                  lambda(kk)=lambda(kk)+z*gamma(h);
                  indepx(kk,h)=indepx(kk,h)+z;
             }
        }
        
        response.iresp=data(i).y.iresp;
        if(uu>0)response.iresp.elem(patterns(j).ithetas)=theta.iresp.elem(patterns(j).ithetas);
        response.dresp=data(i).y.dresp;
        if(uuu>0)response.dresp.elem(patterns(j).dthetas)=theta.dresp.elem(patterns(j).dthetas);
        if(order>0&&q>0)obsresults.grad.set_size(r);
        if(order>1&&q>0)obsresults.hess.set_size(r,r);
        if(u>0)
        {    
            obsresults=genresp(order1, patterns(j).choice, response, lambda);
        }
        else
        {
            obsresults=genresp(order0, patterns(j).choice, response, lambda);
        }
        if(isnan(obsresults.value))
        {
            results.value=datum::nan;
            if(order>0)results.grad.fill(datum::nan);
            if(order>1)results.hess.fill(datum::nan);
            return results;
        }    
        if((u==0)||order==0)
        {
            addsel(order0,selectbeta(jj),obsresults,results,w(i));
        }
        else 
        {
            if(order>2)obsresults.hess=-obsresults.grad*obsresults.grad.t();
            linresults.grad.set_size(q);
            if(order>1) linresults.hess.set_size(q,q);
            linresults=linsel(order,obsresults,indepx);
            addsel(order,selectbeta(jj),linresults,results,w(i));  
        }
    }
    return results;
}
