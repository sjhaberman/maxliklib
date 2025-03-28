//Pack vector of dimension d and a d by d symmetric matrix into a
//vector of dimension d(d+3)/2.
#include<armadillo>
using namespace arma;
struct vecmat
{
    vec v;
    mat m;
};
vec pack(const vecmat & u)
{
    int d,i,j,k;
    vec result;
    d=u.v.n_elem;
    result.set_size(d*(d+3)/2);
    result.subvec(0,d-1)=u.v;
    k=d;
    for(i=0;i<d;i++)
    {
        for(j=0;j<=i;j++)
        {
            if(j==i)
            {
                result(k)=u.m(i,j);
            }
            else
            {
                result(k)=2.0*u.m(i,j);
            }
            k=k+1;
        }
    }
    return result;
}
