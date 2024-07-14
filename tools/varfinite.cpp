//Find if variable is always finite and present.  Variable name is assumed known to be valid.
#include<armadillo>
using namespace arma;
using namespace std;
struct vn
{
     field<string>varnames;
     mat vars;
};
struct model 
{
     char type;
     char transform;
};
struct varlocs
{
     string varname;
     vector<int>forms;
     vector<int>positions;
     model choice;
     int ncat;
};
int varlookup(const string & , const vector<varlocs> &);
bool varfinite(const string & name, const vector<varlocs> & vartab, const field<vn> & dataf)
{
     int i, j, k, m, n;
     i=varlookup(name,vartab);
     n=vartab[i].forms.size();
//See if variable on all forms.
     if(n!=dataf.n_elem)return false;
//See if always finite.   
     for(j=0;j<n;j++)
     {
         k=vartab[i].forms[j];
         m=vartab[i].positions[j];
         if(!dataf(k).vars.col(m).is_finite())return false;
     }
     return true;
}