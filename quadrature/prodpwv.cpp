//Compute product points and weights.
#include<armadillo>
using namespace std;
using namespace arma;
struct pwv
{
    mat points;
    vec weights;
};
struct pw
{
    vec points;
    vec weights;
};
pwv prodpwv(const vector<pw> & pws)
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
    pwv results;
    results.points.set_size(n,m);
    results.weights.set_size(m);
    coord.zeros();
    for(j=0;j<m;j++)
    {
         x=1.0;
         for(i=0;i<n;i++)
         {
              results.points(i,j)=pws[i].points(coord(i));
              x=x*pws[i].weights(coord(i));
         }
         results.weights(j)=x;
         for(i=0;i<n;i++)
         {
             coord(i)=coord(i)+1;
             if(coord(i)<ncoord(i)) break;
             coord(i)=0;
         }
    }
    return results;
}
            
            
            
