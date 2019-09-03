//Unpack vector beta of dimension d(d+3)/2 into a vector and a triangular matix.


#include<armadillo>
using namespace arma;
using namespace std;
struct vecmat
{
    vec v;
    mat m;
};

vecmat unpack(int d,vec beta)
{
    int i,j,k;
    
    vecmat result;
    result.v=beta.subvec(0,d-1);
    result.m=zeros(d,d);
    k=d;
    for(i=0;i<d;i++)
    {
        
        
        for(j=0;j<=i;j++)
        {
            
            result.m(i,j)=beta(k);
            k=k+1;
        }
        
    }
    
    return result;
}
