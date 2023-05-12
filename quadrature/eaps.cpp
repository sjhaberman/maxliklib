//Find EAP and corresponding conditional covariance matrix for all observations.
//prob has n elements that are conditional probabilities.
//x is m by n with columns that correspond to elements of prob.
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
vecmat eap(const vecmat &  );
vector<vecmat> eaps(const vector<vecmat> & posts)
{ 
    int i,k,n;
    n=posts.size();
    vector<vecmat> result(n);
    for(i=0;i<n;i++)
    {
         k=posts[i].m.n_cols;
         if(posts[i].v.n_elem>0&&k>0)
         {
              result[i].v.set_size(k);
              result[i].m.set_size(k,k);
              result[i]=eap(posts[i]);
         }
         
    }   
    return result;
}
