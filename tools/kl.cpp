//Function to maximize when finding minimum discriminant information
//adjustment of a polytomous variable with vector p of s positive probabilities
//to a vector q with s positive probabilities such that u=sum(T*q) for an r by s matrix T
//and a given vector u of dimension r.
#include<armadillo>
using namespace arma;
//Function, gradient, and Hessian.
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
// Combination of vector and matrix.
struct vecmat
{
    vec v;
    mat m;
};
//Weighted mean and covariance matrix.
//Data matrix is wx.m and data weight is wx.v.
vecmat wmc(const int & , const vecmat & );
f2v kl(const int & order, const vec & p, const mat & T, const vec & u, const vec & gamma)
{
    f2v results;
    double c;
    int r, s;
    r=p.n_elem;
    s=gamma.n_elem;
    if(order>0) results.grad.set_size(s);
    if(order>1) results.hess.set_size(s,s);
    vec q(r);
    q=T*gamma;
    q=p%exp(q);
    c=sum(q);
    q=q/c;
    results.value=dot(gamma,u)-log(c);
    if(order>0)
    {
        vecmat wx,wy;
        wx.v.set_size(r);
        wx.v=q;
        wy.v.set_size(s);
        wx.m.set_size(r,s);
        wx.m=T;
        if(order>1)wy.m.set_size(s,s);
        wy=wmc(order,wx);
        results.grad=u-wy.v;
        if(order>1)results.hess=-wy.m;
    }
    return results;
}
