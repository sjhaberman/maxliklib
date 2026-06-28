//Fit location and scale transformation.
//Vector p has r observed weights with sum 1.
//thetas are original quadrature points and weights.
//newthetas are new quadrature points and weights.
//Linear terms in transformation are given by linselect.
//Quadratic terms in transformation are given by quadselect.
#include<armadillo>
using namespace arma;
using namespace std;
//Function, gradient, and Hessian.
struct f2v{double value; vec grad; mat hess;};
//Quadrature points, weights, and kernels.
struct pwr{double weight; double kernel; vec theta;};
struct xsel{bool all; uvec list;};
struct xselv{bool all; umat list;};
// Adaptive quadrature specifications.
// The choice to use is indicated by adapt, xselect shows the elements involved.
struct adq{bool adapt; xsel linselect; xselv quadselect;};
//Adaptive quadrature transformation.
f2v fitquad(const vector<f2v> & cresults, const vector<pwr> & newthetas,
    const adq & scale, f2v & obsscale){
    bool flag;
    uword h, i, j, k, n, r, rr, s, t, u;
    uvec g(2);
    f2v results;
    rr = newthetas[0].theta.n_elem;
    results.grad.copy_size(obsscale.grad);
    results.hess.copy_size(obsscale.hess);
    results.value=obsscale.value;
    results.hess=obsscale.hess;
    results.grad=obsscale.grad;
//Quit if no adaptive quadrature.
    if(!scale.adapt)return results;
    r=rr;
//Transformation dimension.
    if(!scale.linselect.all) r=scale.linselect.list.n_elem;
    s=rr*(rr+1)/2;
    if(!scale.quadselect.all) s=scale.quadselect.list.n_cols;
//s=0 means no quadratic component.
    if(s==0) return results;
//Gamma dimension.
    t=1+r+s;
    u=cresults.size();
//Not enough data.
    if(u<t) return results;
    vec gamma(t),q(u),lin(rr);
    mat T(u,t),W(rr,rr,fill::eye);
    k=1;
//Set matrices;
    T.col(0)=ones(u);
    for(i=0;i<r;i++){
        if(scale.linselect.all)j=i;
        else j=scale.linselect.list(i);
        for(h=0;h<u;h++) T(h,k)=newthetas[h].theta(j);
        k=k+1;
    }
    for(i=0;i<s;i++){
        if(!scale.quadselect.all)g=scale.quadselect.list.col(i);
        for(h=0;h<u;h++){
            T(h,k)=newthetas[h].theta(g(0))*newthetas[h].theta(g(1));
        }
        k=k+1;
        if(scale.quadselect.all){
            if(g(0)<g(1))g(0)=g(0)+1;
            else{
                g(0)=0;
                g(1)=g(1)+1;
            }
        }
    }
//Dependent variables
    for(h=0;h<u;h++)q(h)=cresults[h].value;
//Regression.
    flag=solve(gamma,T,q);
    if(!flag)return results;
    k=1;
//Fit maximum and minus inverse Hessian.
    for(i=0;i<r;i++){
        if(scale.linselect.all)j=i;
        else j=scale.linselect.list(i);
        lin(j)=gamma(k);
        k=k+1;
    }
    g.zeros();
    for(i=0;i<s;i++){
        if(!scale.quadselect.all)g=scale.quadselect.list.col(i);
        if(g(0)<g(1)){
            W(g(0),g(1))=-gamma(k);
            W(g(1),g(0))=-gamma(k);
        }
        else W(g(0),g(0))=-2.0*gamma(k);
        k=k+1;
        if(scale.quadselect.all){
            if(g(0)<g(1)) g(0)=g(0)+1;
            else{
                g(0)=0;
                g(1)=g(1)+1;
            }
        }
    }
//If not positive definite, quit.   
    if(!(W.is_sympd()))return results;
//Otherwise find new results;
    flag=inv_sympd(results.hess,W);
    if(!flag)return results;
    results.grad=results.hess*lin;
    results.hess=sqrtmat_sympd(results.hess);
    results.value=det(results.hess);
    return results;
}
