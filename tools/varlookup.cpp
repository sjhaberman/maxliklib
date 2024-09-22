//Find vartable entry in variable table based on name.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs
{
     string varname;
     vector<int> forms;
     vector<int> positions;
     char type;
     char transform;
     vector<string>catnames;
};
vector<varlocs>::iterator varlookup(const string & name, vector<varlocs>&vartab)
{
    const function <bool(const varlocs & v)>f=[&name](const varlocs & v)
         {return (v.varname==name);};
    vector<varlocs>::iterator it;
    it=find_if(vartab.begin(),vartab.end(),f);
    return it;
}
