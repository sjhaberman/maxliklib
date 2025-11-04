//dupname checks if all elements of names are distinct.  If so, true is returned.
//Otherwise false is returned.
#include<armadillo>
#include<string>
using namespace arma;
using namespace std;
bool dupname(const vector<string> & names)
{
    if(names.size()<2)return true;
    vector<string> sortnames(names.size());
    copy(names.cbegin(),names.cend(),sortnames.begin());
    sort(sortnames.begin(),sortnames.end());
    if(adjacent_find(sortnames.begin(),sortnames.end())>=sortnames.end())
        return true;
    return false;
}
