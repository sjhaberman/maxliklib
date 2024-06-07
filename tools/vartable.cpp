//Table of variables and where they appear.
#include<armadillo>
using namespace arma;
using namespace std;
struct vn
{
    field<string>varnames;
    mat vars;

};
struct varlocs
{
      string varname;
      imat locs;
};
field<varlocs>vartable(field<vn> & forms)
{
    int i,i1,j,j1,k,nforms,nvar,vnum=0;
    nforms=forms.n_elem;
    for(i=0;i<nforms;i++)
    {
         vnum=vnum+forms(i).varnames.n_elem;
    }
    imat ranks(vnum,vnum);
    imat vtab(vnum,2);
    ivec rsum(vnum);
    i1=0;
    for(i=0;i<nforms;i++)
    {
        for(j=0;j<forms(i).varnames.n_elem;j++)
        {
             vtab(i1,0)=i;
             vtab(i1,1)=j;
             i1=i1+1;
        }
    }
    ranks(0,0)=0;
    for(i=1;i<vnum;i++)
    {
        ranks(i,i)=0;
        for(j=0;j<i;j++)
        {
             ranks(i,j)=
                 sign(forms(vtab(i,0)).varnames(vtab(i,1)).
                 compare(forms(vtab(j,0)).varnames(vtab(j,1))));
             ranks(j,i)=-ranks(i,j); 
        }
    }
    rsum=sum(ranks,1);
    uvec uniqueind;
    uniqueind=find_unique(rsum);
    nvar=uniqueind.n_elem;
    ivec rsumuni(nvar);
    uvec orderind(nvar);
    for(i=0;i<nvar;i++) rsumuni(i)=rsum(uniqueind(i));
    orderind=sort_index(rsumuni);
    field<varlocs>result(nvar);
    field<uvec>finds(nvar);
    for(i=0;i<nvar;i++)
    {
         j=orderind(i);
         j=uniqueind(j);
         finds(i)=find(rsum==rsum(j));
         result(i).varname=forms(vtab(j,0)).varnames(vtab(j,1));
         j1=finds(i).n_elem;
         cout<<i<<" "<<j<<" "<<j1<<endl;
         result(i).locs.set_size(j1,2);
         for(i1=0;i1<j1;i1++)
         {
              k=finds(i)(i1);
              result(i).locs.row(i1)=vtab.row(k);
         }
    }
    return result;
             
    
    
    

}
