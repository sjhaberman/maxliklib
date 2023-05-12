//Find EAP and corresponding conditional covariance matrix.
//post.v has n elements that are conditional probabilities.
//post.m is m by n with columns that correspond to elements of post.v.
//eap.v is mean and eap.m is covariance.
#include<armadillo>
using namespace arma;
struct vecmat
{
    vec v;
    mat m;
};
mat wcrossprod(const vecmat & );
vecmat eap(const vecmat & post)
{ 
    int k,q;
    q=post.v.n_elem;
    k=post.m.n_rows;
    vecmat postm,result;
    postm.v.copy_size(post.v);
    postm.m.copy_size(post.m);
    result.v.set_size(k);
    result.m.set_size(k,k);
    result.v=post.m*post.v;
    postm.v=post.v;
    postm.m=post.m-result.v*ones<rowvec>(q);
    result.m=wcrossprod(postm);
    return result;
}
