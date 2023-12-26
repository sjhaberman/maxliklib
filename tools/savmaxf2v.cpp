//Save value, location, gradient, Hessian, asymptotic covariance matrix, and asymptotic
//standard errors from maximization results in struct maxf2v.  fflag indicates file saving,
//and pflag indicates printing.
#include<armadillo>
using namespace arma;
using namespace std;
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
void savmaxf2v(const int & order , const maxf2v & vlm, const string & out,
     const bool & fflag, const bool & pflag)
{
    
    int o=5, p, q=2;
    if(order<2)
    {
        o=3;
        q=1;
    }
    field<mat>result(o);
    p=vlm.locmax.n_elem;
    result(0).set_size(1,1);
    result(0)(0,0)=vlm.max;
    result(1).set_size(p,q);
    result(1).col(0)=vlm.locmax;
    result(2).set_size(p,1);
    result(2).col(0)=vlm.grad;
    if(order>1)
    {
        result(3).set_size(p,p);
        result(3)=vlm.hess;
        result(4).set_size(p,p);
        result(4)=inv_sympd(-vlm.hess);
        result(1).col(1)=sqrt(diagvec(result(4)));
    }

    if(fflag)result.save(out);
    if(pflag)result.print();
    return;
}

