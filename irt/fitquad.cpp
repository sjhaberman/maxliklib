//Fit location and scale transformation.
//Vector p has r observed weights with sum 1.
//thetas are original quadrature points and weights.
//newthetas are new quadrature points and weights.
//Linear terms in transformation are given by linselect.
//Quadratic terms in transformation are given by quadselect.
#include<armadillo>
using namespace arma;
//Function, gradient, and Hessian.
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
//Adaptive quadrature transformation structure.
struct dovecmat
{
    double s;
    vec v;
    mat m;
};
//Quadrature points, weights, and kernels.
struct resp
{
  ivec iresp;
  vec dresp;
};
struct pwr
{
    double weight;
    double kernel;
    resp theta;
};
struct xsel
{
    bool all;
    uvec list;
};
struct xselv
{
    bool all;
    umat list;
};
// Adaptive quadrature specifications.
// The choice to use is indicated by adapt, xselect shows the elements involved.
struct adq
{
    bool adapt;
    xsel linselect;
    xselv quadselect;
};
//Adaptive quadrature transformation.

dovecmat fitquad(const field<f2v> & cresults, const field<pwr> & newthetas,
    const adq & scale, dovecmat & obsscale)
{
    bool flag;
    int h, i, j, k, n, r, rr, s, t, u;
    uvec g(2,fill::zeros);
    dovecmat results;
    rr = newthetas(0).theta.dresp.n_elem;
    results.v.copy_size(obsscale.v);
    results.m.copy_size(obsscale.m);
    results.v=obsscale.v;
    results.m=obsscale.m;
    results.s=obsscale.s;
//Quit if no adaptive quadrature.
    if(!scale.adapt)return results;
    r=rr;
//Transformation dimension.
    if(!scale.linselect.all) r=scale.linselect.list.n_elem;
    s=rr*(rr+1)/2;
    if(!scale.quadselect.all) s=scale.quadselect.list.n_cols;
//s=0 means no quadratic component.
    if(s==0) return results;
//Gamma dimension
    t=1+r+s;
    u=cresults.n_elem;
//Not enough data.
    if(u<t) return results;
    vec gamma(t),q(u),lin(rr,fill::zeros);
    mat T(u,t),W(rr,rr,fill::eye);
    k=1;
//Set matrices;
    T.col(0)=ones(u);
    if(r>0){
    for(i=0;i<r;i++)
    {
         if(scale.linselect.all)
         {
              j=i;
         }
         else
         {
              j=scale.linselect.list(i);
         }
         for(h=0;h<u;h++) T(h,k)=newthetas(h).theta.dresp(j);
         k=k+1;
    }}
    for(i=0;i<s;i++)
    {
         if(!scale.quadselect.all)g=scale.quadselect.list.col(i);
         for(h=0;h<u;h++) T(h,k)=newthetas(h).theta.dresp(g(0))*newthetas(h).theta.dresp(g(1));
         k=k+1;
         if(scale.quadselect.all)
         {
              if(g(0)<g(1))
              {
                   g(0)=g(0)+1;
              }
              else
              {
                   g(0)=0;
                   g(1)=g(1)+1;
              }
         }
    }
//Dependent variables
    for(h=0;h<u;h++)q(h)=cresults(h).value;
//Regression.
    flag=solve(gamma,T,q);
    if(!flag)return results;
    k=1;
//Fit maximum and minus inverse Hessian.
    if(r>0){
    for(i=0;i<r;i++)
    {
         if(scale.linselect.all)
         {
              j=i;
         }
         else
         {
              j=scale.linselect.list(i);
         }
         lin(j)=gamma(k);
         k=k+1;
    }}
    g.zeros();
    for(i=0;i<s;i++)
    {
         if(!scale.quadselect.all)g=scale.quadselect.list.col(i);
         if(g(0)<g(1))
         {
              W(g(0),g(1))=-gamma(k);
              W(g(1),g(0))=-gamma(k);
         }
         else
         {
              W(g(0),g(0))=-2.0*gamma(k);
         }
         k=k+1;
         if(scale.quadselect.all)
         {
              if(g(0)<g(1))
              {
                   g(0)=g(0)+1;
              }
              else
              {
                   g(0)=0;
                   g(1)=g(1)+1;
              }
         }
    }
//If not positive definite, quit.   
    if(!(W.is_sympd()))return results;
//Otherwise find new results;
    results.m=inv_sympd(W);
    results.v=results.m*lin;
    results.m=sqrtmat_sympd(results.m);
    results.s=det(results.m);
    return results;
}
            
            
            
