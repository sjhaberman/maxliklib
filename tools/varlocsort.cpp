//Sorting varlocs by varname.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs{vector<uword> forms; vector<uword> positions; string varname;};
bool varlocsort(const varlocs & a,const varlocs & b)
    {return(a.varname<b.varname);}
