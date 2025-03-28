//dupname checks if all elements of names are distinct.  If so, true is returned.
//Otherwise false is returned.
#include<armadillo>
#include<string>
using namespace arma;
using namespace std;
bool dupname(const vector<string> & names)
{
    vector<string>::const_iterator namesit1,namesit2;
    if(names.size()<2)return true;
    for(namesit1=next(names.cbegin(),1);namesit1!=names.cend();++namesit1)
    {
        for(namesit2=names.cbegin();namesit2!=namesit1;++namesit2)
        {
            if(*namesit1==*namesit2)return false;
        }
    }
    return true;
}
