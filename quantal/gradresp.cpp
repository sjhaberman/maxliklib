//Log likelihood component, gradient, and Hessian
//for modified graded response model with response y with integer values
//Y=y.iresp(0) from 0 to  to n and parameter vector beta of dimension n.
//The distribution of Y involves an
//unobserved continuous variable Z with positive twice continuously differentiable density,
//distribution function F, quantile function T specified by transform, and n cut points t(k),
//k a nonnegative integer less than n.
//The cut points are strictly increasing in k.
//The observed variable y.iresp(0) is the number of cut points
//that are not less than Z. Let q be the largest integer no greater
//than n/2.  The initial element beta(0) of beta is t(q), and
//for k greater than 0 and less than n,
//log(t(k))-t(k-1))) is beta(k).  Thus for k>q, t(k) is
//beta(0) plus the sum for j from q+1 to k of exp(beta(j)).
//For k<q, t(k) is beta(0) minus the sum for j from k+1 to q
//of exp(beta(j)).
//Choice of T has 'G' for the
//complementary log-log transformation, 'H' for log-log,
//'L' for logit, and 'N' for probit.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f2v berresp(const int & , const char & , const resp & , const vec & );
f2v gradresp(const int & order, const char & transform ,
    const resp & y, const vec & beta)
{
    int n;
    n=beta.n_elem;
//Bernoulli case.
    if(n==1)return berresp(order, transform, y, beta);
//result is for t(k) for k=n-y.iresp(0), and result1 is for
//t(k-1).
//results is for final result.
    f2v result, result1, results;
    if(order>0)results.grad.zeros(n);
    if(order>1)results.hess.zeros(n,n);
    int k;
    k=n-y.iresp(0);
    resp z;
    z.iresp={1};
//lb and ub give bounds used for t(k) if k<n is not q, while lb1 and ub1 are for
//bounds used for t(k-1) if k>0 is not q+1.
//Bounds are 0 if not used.  sp is a range variable.
    int lb=0, lb1=0, q, sp, ub=0, ub1=0;
    q=n/2;
//gamma is the parameter vector for t(k).
//gamma1 is the parameter vector for t(k-1).
//ebeta is used to find t(k).
//ebeta1 is for t(k-1).
    vec ebeta, ebeta1, gamma(1), gamma1(1);
//Now for bounds.
    if(k==n){
        if(n==2)result1=berresp(order, transform, y, beta);
//Note that n-q>1 if n>2.
        else{
            lb1=q+1;
            ub1=n-1;
            ebeta1=exp(beta.subvec(lb1,ub1));
            gamma1(0)=beta(0)+sum(ebeta1);
            result1=berresp(order, transform, y, gamma1);
        }
        results.value=result1.value;
        if(order==0)return results;
        results.grad(0)=result1.grad(0);
        if(n>2)results.grad.subvec(lb1,ub1)=result1.grad(0)*ebeta1;
        if(order==1)return results;
        results.hess(0,0)=result1.hess(0,0);
        if(n>2){
            (results.hess.col(0)).subvec(lb1,ub1)=result1.hess(0,0)*ebeta1;
            (results.hess.row(0)).subvec(lb1,ub1)=((results.hess.col(0)).subvec(lb1,ub1)).t();
            results.hess.submat(lb1,lb1,ub1,ub1)=result1.hess(0,0)*ebeta1*ebeta1.t()+result1.grad(0)*diagmat(ebeta1);
        }
        return results;
    }
    else{
        if(k==0){
            lb=1;
            ub=q;
            ebeta=-exp(beta.subvec(lb,ub));
            gamma(0)=beta(0)+sum(ebeta);
            result=berresp(order, transform, z, gamma);
            results.value=result.value;
            if(order==0)return results;
            results.grad(0)=result.grad(0);
            results.grad.subvec(lb,ub)=result.grad(0)*ebeta;
            if(order==1)return results;
            results.hess(0,0)=result.hess(0,0);
            (results.hess.col(0)).subvec(lb,ub)=result.hess(0,0)*ebeta;
            (results.hess.row(0)).subvec(lb,ub)=((results.hess.col(0)).subvec(lb,ub)).t();
            results.hess.submat(lb,lb,ub,ub)=result.hess(0,0)*ebeta*ebeta.t()+result.grad(0)*diagmat(ebeta);
            return results;
        }
        else{
            if(k==q){
                result=berresp(order, transform, z, beta);
                lb1=q;
                ub1=lb1;
                ebeta1=-exp(beta.subvec(lb1,ub1));
                gamma1(0)=beta(0)+ebeta1(0);
                result1=berresp(order, transform, z, gamma1);
            }
            else{
                if(k==q+1){
                    lb=q+1;
                    ub=lb;
                    ebeta=exp(beta.subvec(lb,ub));
                    gamma(0)=beta(0)+ebeta(0);
                    result=berresp(order, transform, z, gamma);
                    result1=berresp(order, transform, z, beta);
                }
                else{
                    if(k>q+1){
                        lb=q+1;
                        ub=k;
                        ebeta=exp(beta.subvec(lb,ub));
                        gamma(0)=beta(0)+sum(ebeta);
                        result=berresp(order, transform, z, gamma);
                        lb1=lb;
                        ub1=ub-1;
                        sp=ub-lb;
                        ebeta1=ebeta.head(sp);
                        gamma1(0)=gamma(0)-ebeta(sp);
                        result1=berresp(order, transform, z, gamma1);
                    }
                    else{
                        ub=q;
                        lb=k+1;
                        ub1=ub;
                        lb1=k;
                        ebeta1=-exp(beta.subvec(lb1,ub1));
                        gamma1(0)=beta(0)+sum(ebeta1);
                        result1=berresp(order, transform, z, gamma1);
                        sp=ub1-lb1;
                        ebeta=ebeta1.tail(sp);
                        gamma(0)=gamma1(0)-ebeta1(0);
                        result=berresp(order, transform, z, gamma);
                    }
                }
            }
        }
    }
//Remaining cases with results not yet returned.
    double d, d1, e, f, f1, g=0.0, g1=0.0, gr=0.0, gr1=0.0, gg, h=0.0, h1=0.0;
    d=exp(result.value);
    d1=exp(result1.value);
    e=d-d1;
    results.value=log(e);
    if(order==0)return results;
    f=d/e;
    f1=d1/e;
    gr=f*result.grad(0);
    gr1=-f1*result1.grad(0);
    results.grad(0)=gr+gr1;
    if(lb>0)results.grad.subvec(lb,ub)=gr*ebeta;
    if(lb1>0)results.grad.subvec(lb1,ub1)=results.grad.subvec(lb1,ub1)+gr1*ebeta1;
    if(order==1) return results;
    gg=results.grad(0);
    g=gr*result.grad(0);
    h=f*result.hess(0,0);
    g1=gr1*result1.grad(0);
    h1=-f1*result1.hess(0,0);
    results.hess(0,0)=g+h+g1+h1-gg*gg;
//Just upper bound.
    if(lb>0&&lb1==0){
        (results.hess.col(0)).subvec(lb,ub)=(g+h-gg*gr)*ebeta;
        (results.hess.row(0)).subvec(lb,ub)=((results.hess.col(0)).subvec(lb,ub)).t();
        results.hess.submat(lb,lb,ub,ub)=gr*diagmat(ebeta)+(g+h-gr*gr)*ebeta*ebeta.t();
        return results;
    }
//Just lower bound.
    else{
        if(lb1>0&&lb==0){
            (results.hess.col(0)).subvec(lb1,ub1)=(g1+h1-gg*gr1)*ebeta1;
            (results.hess.row(0)).subvec(lb1,ub1)=((results.hess.col(0)).subvec(lb1,ub1)).t();
            results.hess.submat(lb1,lb1,ub1,ub1)=gr1*diagmat(ebeta1)+(g1+h1-gr1*gr1)*ebeta1*ebeta1.t();
            return results;
        }
//Upper and lower bounds.
        else{
            (results.hess.col(0)).subvec(lb,ub)=(g+h-gg*gr)*ebeta;
            (results.hess.row(0)).subvec(lb,ub)=((results.hess.col(0)).subvec(lb,ub)).t();
            (results.hess.col(0)).subvec(lb1,ub1)=(results.hess.col(0)).subvec(lb1,ub1)+(g1+h1-gg*gr1)*ebeta1;
            (results.hess.row(0)).subvec(lb1,ub1)=((results.hess.col(0)).subvec(lb1,ub1)).t();
            results.hess.submat(lb,lb,ub,ub)=gr*diagmat(ebeta)+(g+h-gr*gr)*ebeta*ebeta.t();
            results.hess.submat(lb1,lb1,ub1,ub1)=results.hess.submat(lb1,lb1,ub1,ub1)+gr1*diagmat(ebeta1)
                +(g1+h1-gr1*gr1)*ebeta1*ebeta1.t();
            results.hess.submat(lb,lb1,ub,ub1)=results.hess.submat(lb,lb1,ub,ub1)-gr*gr1*ebeta*ebeta1.t();
            results.hess.submat(lb1,lb,ub1,ub)=results.hess.submat(lb1,lb,ub1,ub)-gr1*gr*ebeta1*ebeta.t();
            return results;
        }
    }
}
