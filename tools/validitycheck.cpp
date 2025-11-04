//Check if observation valid.  type is type of variable.  x is its value.
#include<armadillo>
using namespace arma;
using namespace std;
struct xsel
{
    bool all;
    uvec list;
};
struct varlocs
{
    vector<int> forms;
    vector<int> positions;
    string varname;
};
struct vardef
{
    vector<string>catnames;
    xsel constant;
    xsel deg1;
    xsel full;
    vec o;
    bool obs;
    vector<string> preds;
    vector<vector<varlocs>::iterator> predits;
    char transform;
    char type;
    string vardefname;
    vector<string>varname;
    vector<vector<varlocs>::iterator> varnameit;
};
bool validitycheck(const vardef & vardefs, const vector<double> & x)
{
//First issue is whether item observed at all.
    char type;
    if(any_of(x.begin(),x.end(),[](double y){return !isfinite(y);}))return false;
    double z;
    z=x[0];
    type=vardefs.type;
    vector<double>u;
    if(type=='R')
    {
        u.resize(x.size());
        copy(x.begin(),x.end(),u.begin());
        sort(u.begin(),u.end());
    }
    switch(type)
    {
        case 'B': return (z>0.0&z<1.0)? true: false;
            
        case 'C': return
            ((z==floor(z))&(z>=0.0)&(z<(double)vardefs.catnames.size()))?
            true:false;
            
        case 'D': return true;
            
        case 'E':  return ((all_of(x.begin(),x.end(),[](double y)
            {return (y>0.0);}))
            &(fabs(accumulate(x.begin(),x.end(),-1.0))
            <datum::eps*(double)x.size()))?
            true:false;
            
        case 'F':  return true;
            
        case 'G': return
            ((z==floor(z))&(z>=0.0)&(z<(double)vardefs.catnames.size()))?
            true:false;
            
        case 'H': return (z>0.0)? true:false;
            
        case 'L':return
            ((z==floor(z))&(z>=0.0)&(z<(double)vardefs.catnames.size()))?
            true:false;
            
        case 'M': return ((z==0.0)|(z==1.0))? true: false;
            
        case 'P': return ((z==floor(z))&(z>=0.0))?true:false;
        
        case 'R': return ((all_of)(u.begin(),u.end(),[vardefs](double y)
            {return ((y>=0.0)
            &(y<(double)vardefs.catnames.size()));})
            &(adjacent_find(u.begin(),u.end())>=u.end()))?
            true:false;
            
        case 'S': return ((z==0.0)|(z==1.0))? true: false;
            
        default: return ((x[1]==0.0)|(x[1]==1.0))? true: false;
    }
}
