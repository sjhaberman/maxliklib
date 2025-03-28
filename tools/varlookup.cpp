//Find vartable entry in variable table based on name.
#include<armadillo>
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
vector<varlocs>::iterator varlookup(const string & name, vector<varlocs>&vartab)
{
    const function <bool(const varlocs & v)>f=[&name](const varlocs & v)
       {return (v.varname==name);};
    vector<varlocs>::iterator it;
    it=find_if(vartab.begin(),vartab.end(),f);
    return it;
}
