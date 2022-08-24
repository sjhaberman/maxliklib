//Unpack vector beta of dimension (d+1)(d+2)/2 into a value, a vector, and a symmetric matix.
#include<armadillo>
using namespace arma;
struct fvecmat
{
    double value;
    vec v;
    mat m;
};
fvecmat trisym(const int & d, const vec & beta)
{
    int i,j,k;
    fvecmat result;
    result.value=beta(0);
    result.v.set_size(d);
    result.v=beta.subvec(1,d);
    result.m.set_size(d,d);
    k=d+1;
    for(i=0;i<d;i++)
    {
        for(j=0;j<=i;j++)
        {
            if(i==j)
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
