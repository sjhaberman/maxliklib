//Find vartable entry in variable table based on name.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs
{
      string varname;
      imat locs;
};
int varlookup(string & name, field<varlocs>&vartab)
{
    int c,i,lower,n,pivot,upper;

    n=vartab.n_elem;
    lower=0;
    upper=n-1;
    while(lower<upper)
    {
         pivot=(lower+upper)/2;
         c=name.compare(vartab(pivot).varname);
         if(c==0)return pivot;
         if(c<0)
         {
              upper=pivot-1;
              continue;
         }
         lower=pivot+1;
    }
    return lower;
}
