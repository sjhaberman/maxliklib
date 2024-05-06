//Find EAP and corresponding conditional covariance matrix for all n observations.
//For each observation j, posts(j) has a vector of m probabilities and
//a m by k matrix of m responses of dimension k.
//eap.v is mean and eap.m is covariance.
#include<armadillo>
using namespace std;
using namespace arma;
// Combination of vector and matrix.
struct vecmat
{
    vec v;
    mat m;
};
vecmat wmc(const vecmat &  );
field<vecmat> eaps(const field<vecmat> & posts)
{ 
    int i,k,n;
    n=posts.size();
    field<vecmat> result(n);
    for(i=0;i<n;i++)
    {
         k=posts(i).m.n_cols;
         if(posts(i).v.n_elem>0&&k>0)
         {
              result(i).v.set_size(k);
              result(i).m.set_size(k,k);
              result(i)=wmc(posts(i));
         }
         
    }   
    return result;
}
