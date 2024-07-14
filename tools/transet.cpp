//Set up type of model for group of variables.
#include<armadillo>
using namespace std;
using namespace arma;
struct varlocs
{
     string varname;
     vector<int> forms;
     vector<int>positions;
     char type;
     char transform;
     int ncat;
};
int varlookup(const string & , const vector<varlocs> & );
bool transet(const field<string> & c, const char & b, vector<varlocs> & vartab)
{
     int varno;
     char p='.';
     if(!c.empty())
     {
          for(string v:c)
          {
              int varno;
              varno=varlookup(v,vartab);
              if(varno>=vartab.size())
              {
                   cout<<"Variable misspecified"<<endl;
                   return false;
              }
              if(vartab[varno].type!=p)
              {
                   cout<<"Model transform redefined"<<endl;
                   return false;
              }
              vartab[varno].transform=b; 
          }
    }
    return true;
}
