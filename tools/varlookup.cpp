//Find vartable entry in variable table based on name.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs
{
    vector<int> forms;
    vector<int> positions;
    string varname;
};
bool varlocsort(const varlocs & a, const varlocs & b);
vector<varlocs>::iterator varlookup(const string & name,
    vector<varlocs>&vartab)
{
    varlocs vaname;
    vaname.varname=name;
    pair<vector<varlocs>::iterator,vector<varlocs>::iterator> it;
    it=equal_range(vartab.begin(),vartab.end(),vaname,varlocsort);
    if(it.first==vartab.end()) return it.first;
    if(next(it.first,1)!=it.second) return vartab.end();
    return it.first;
}
