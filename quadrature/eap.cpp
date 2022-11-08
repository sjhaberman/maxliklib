//Find EAP and corresponding conditional covariance matrix.
//prob has n elements that are conditional probabilities.
//x is m by n with columns that correspond to elements of prob.
//eap.v is mean and eap.m is covariance.
#include<armadillo>
using namespace arma;
struct vecmat
{
    vec v;
    mat m;
};
mat wcrossprod(const mat & ,const vec & );
vecmat eap(const vec & prob, const mat & x)
{ 
    int m,n;
    m=x.n_rows;
    n=x.n_cols;
    vecmat result;
    result.v.set_size(n);
    result.m.set_size(n,n);
    result.v=x*prob;
    result.m=wcrossprod(trans(x)-ones(n)*trans(result.v),prob);
    return result;
}
