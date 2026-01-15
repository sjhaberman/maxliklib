//Sorting vardefs by vardefname.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs
{
    vector<int> forms;
    vector<int> positions;
    string varname;
};
struct vardef
{
    vector<string>catnames;
    vec o;
    bool obs;
    vector<vector<varlocs>::iterator> predits;
    vector<string> preds;
    vector<mat> predweights;
    char transform;
    char type;
    string vardefname;;
    vector<string>varname;
    vector<vector<varlocs>::iterator> varnameit;
};
bool vecsort(const vec & ,const vec & );
bool vardefsort(const vardef & a,const vardef & b)
{
    int i;
    if(a.catnames!=b.catnames)return(a.catnames<b.catnames);
    if(!((a.o-b.o).is_zero()))return vecsort(a.o,b.o);
    if(a.obs!=b.obs)return(a.obs<b.obs);
    if(a.preds!=b.preds)return(a.preds<b.preds);
    for(i=0;i<a.predweights.size();i++){
        if(!((a.predweights[i]-b.predweights[i]).is_zero())) return
            vecsort(vectorise(a.predweights[i]),vectorise(b.predweights[i]));
    }
    if(a.transform!=b.transform)return(a.transform<b.transform);
    if(a.type!=b.type)return(a.type<b.type);
    return false;
}
