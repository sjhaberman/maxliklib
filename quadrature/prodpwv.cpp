//Compute product points and weights.
#include<armadillo>
using namespace std;
using namespace arma;
struct vecmat
{
    vec v;
    mat m;
};
struct pw
{
    vec points;
    vec weights;
};
vecmat prodpwv(const vector<pw> & pws)
{
    double x;
    int i,j,m,n;
    n=pws.size();
    uvec coord(n),ncoord(n);
    for(i=0;i<n;i++)
    {
        ncoord(i)=pws[i].points.size();
    }
    m=prod(ncoord);
    vecmat results;
    results.m.set_size(n,m);
    results.v.set_size(m);
    coord.zeros();
    for(j=0;j<m;j++)
    {
         x=1.0;
         for(i=0;i<n;i++)
         {
              results.m(i,j)=pws[i].points(coord(i));
              x=x*pws[i].weights(coord(i));
         }
         results.v(j)=x;
         for(i=0;i<n;i++)
         {
             coord(i)=coord(i)+1;
             if(coord(i)<ncoord(i)) break;
             coord(i)=0;
         }
    }
    return results;
}
            
            
            
