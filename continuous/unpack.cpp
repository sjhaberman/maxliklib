//Unpack vector beta of dimension d(d+3)/2 into a vector and a symmetric matrix.
#include<armadillo>
using namespace arma;
struct vecmat
{
    vec v;
    mat m;
};
vecmat unpack(const int & d, const vec & beta)
{
    int i,j,k;
    vecmat result;
    result.v=beta.subvec(0,d-1);
    result.m.set_size(d,d);
    k=d;
    for(i=0;i<d;i++)
    {
        for(j=0;j<=i;j++)
        {
            if(j==i)
            {
                 result.m(i,j)=beta(k);
            }
            else
            {
                 result.m(i,j)=0.5*beta(k);
                 result.m(j,i)=result.m(i,j);
            }
            k=k+1;
        }
    }
    return result;
}
