//Set up type of model for group of variables.
#include<armadillo>
#include<string.h>
using namespace std;
using namespace arma;
struct modeltype
{
     string name;
     char type;
     char transform;
     vector<string> members;
};
struct varlocs
{
     string varname;
     vector<int> forms;
     vector<int> positions;
     char type;
     char transform;
     vector<string>catnames;
};
vector<varlocs>::iterator varlookup(const string & , vector<varlocs> & );

vector<string>parse(const string & );
bool typeset(const modeltype & mt, vector<varlocs> & vartab)
{
     size_t fl;
     int fcount;
     vector<varlocs>::iterator varno;
     char p='.',q='|',r=',';
     string str1,str2;
     if(!mt.members.empty())
     {

          for(string v:mt.members)
          {
              fl=v.find(q);
              if(fl==string::npos)
              {
                    str1=v;
                    str2="";
              }
              else
              {
                   if(v.back()==q)
                   {
                        cout<<"Variable name misspecified."<<endl;
                        return false;
                   }
                   str1=v.substr(0,fl);
                   str2=v.substr(fl+1);
              }

              varno=varlookup(str1,vartab);
              if(varno==vartab.end())
              {
                   cout<<"Variable name misspecified."<<endl;
                   return false;
              }
              if(varno->type!=p)
              {
                   cout<<"Model type redefined"<<endl;
                   return false;
              }
              varno->type=mt.type;
              if(varno->transform!=p)
              {
                   cout<<"Model transform redefined"<<endl;
                   return false;
              }
              varno->transform=mt.transform;
//Categories.
              if(str2!="")
              {
                   varno->catnames=parse(str2);
                   fcount=varno->catnames.size();
                   if(fcount==1)
                   {
                        cout<<"Cannot have 1 category."<<endl;
                        return false;
                   }
                   if(varno->type=='S'&&fcount>2)
                   {
                        cout<<"Dichotomous variable cannot have more than 2 categories."<<endl;
                        return false;
                   }
              }              
         }
    }
    return true;
}
