//Find variable, form, and position table from array of forms.
#include<armadillo>
using namespace arma;
using namespace std;
struct vn
{
     field<string>varnames;
     mat vars;
};
struct varforpos
{
     string varname;
     int form;
     int position;
};
struct varlocs
{
     string varname;
     vector<int>forms;
     vector<int>positions;
     char type;
     char transform;
     int ncat;
};
bool vtabsort(const varforpos & a,const varforpos & b)
     {return(a.varname<b.varname);};

bool vtabchk(const varforpos & a, const varforpos & b)
     {return((a.varname==b.varname)&&(a.form==b.form));};
vector<varlocs> vfp(const field<vn> & dataf)
{
     int h, i, j, k, numforms, nvar, varcounts;
     varforpos u;
     numforms=dataf.n_elem;
     varcounts=0;
     for(i=0;i<numforms;i++)varcounts+=dataf(i).varnames.n_elem;
     vector<varforpos>vtab(varcounts);
     vector<varlocs>result;
     k=0;
     for(i=0;i<numforms;i++)
     {
         for(j=0;j<dataf(i).varnames.n_elem;j++)
         {
               vtab[k].varname=dataf(i).varnames(j);
               vtab[k].form=i;
               vtab[k].position=j;
               k++;
          }
     }
     stable_sort(vtab.begin(),vtab.end(),vtabsort);
//Check for variable duplication on form.
     if(adjacent_find(vtab.begin(),vtab.end(),vtabchk)<vtab.end())
     {
          cout<<"Duplicate variable name on a form "<<endl;
          return result;
     }
//Check for where variables in vtab start and end.
     vector<int>varcum(varcounts);
     h=0;
     for(i=0;i<varcounts;i++)
     {
          u=vtab[h];    
          const function <bool(const varforpos & v)>f=[&u](const varforpos & v)
          {return vtabsort(u,v);};
          varcum[i]=find_if(vtab.begin()+h,vtab.end(),f)-vtab.begin();
          h=varcum[i];
          if(h==vtab.size())break;
     }
     nvar=h++;
//Now for desired table.
     result.resize(nvar);
     h=0;
     for(k=0;k<nvar;k++)
     {
//Number of entries for variable k.
          i=varcum[k]-h;
          result[k].varname=vtab[h].varname;
          result[k].ncat=0;
          result[k].forms.resize(i);
          result[k].positions.resize(i);
          for(j=0;j<i;j++)
          {
               result[k].forms[j]=vtab[h+j].form;
               result[k].positions[j]=vtab[h+j].position;
          }
          result[k].type='.';
          result[k].transform='.';
          h=varcum[k];
     }
     return result; 
}
