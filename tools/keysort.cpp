//Sort control table by keys.
#include<armadillo>
using namespace arma;
using namespace std;

field<string> keysort(field<string> & source)
{
    int i, i1, j, n, r;
    r=source.n_cols;
    n=source.n_rows;
    imat ranks(n,n);
    ivec rsum(n);
    uvec orderind(n);
    field<string> result(n,r);
    ranks(0,0)=0;
    for(i=1;i<n;i++)
    {
        for(j=0;j<i;j++)
        {
             ranks(i,j)=
                 sign(source(i,0).compare(source(j,0)));
             ranks(j,i)=-ranks(i,j); 
        }
    }
    rsum=sum(ranks,1);
    orderind=stable_sort_index(rsum);
    for(i=0;i<n;i++)
    {
         j=orderind(i);
         result.row(i)=source.row(j);
    }
    return result;
}