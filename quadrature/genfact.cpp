//Generate all cartesian products of finite sets of integers.  Set sizes are in sizes.
#include<armadillo>
using namespace arma;
using namespace std;


imat genfact(ivec sizes)

{
    int i,j,n;
    imat results;
    ivec comb=zeros<ivec>(sizes.n_elem);
    n=prod(sizes);
    results.set_size(sizes.n_elem,n);
    for(j=0;j<n;j++)
    {
        results.col(j)=comb;
        for(i=0;i<sizes.n_elem;i++)
        {
           if(comb(i)<sizes(i)-1)
           {
               comb(i)=comb(i)+1;
               break;
           }
           else
           {
               comb(i)=0;
           }
        }
    }
    
    
    
    return results;
}

