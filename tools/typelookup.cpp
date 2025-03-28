//Find vartable entry in variable table based on type.
#include<armadillo>
//Look for variable of given type.
using namespace arma;
using namespace std;
struct varlocs
{
    vector<string>catnames;
    vector<int> forms;
    vec offset;
    vector<int> positions;
    vector<string> preds;
    char type;
    char transform;
    mat transition;
    string varname;
};
pair<vector<varlocs>::iterator,vector<varlocs>::iterator>
     typelookup(const char & t,  vector<varlocs>&vartab)
{
    const function <bool(const varlocs & v)>f=[&t](const varlocs & v)
        {return (v.type==t);};
    pair<vector<varlocs>::iterator,vector<varlocs>::iterator> it;
    it.first=find_if(vartab.begin(),vartab.end(),f);
    it.second=vartab.end();
    if(it.first>=next(vartab.end(),-1))return it;
    it.second=find_if(next(it.first,1),vartab.end(),f);
    return it;
}
