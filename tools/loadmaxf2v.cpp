//Print results from file saved by savmaxf2v.
#include<armadillo>
using namespace arma;
using namespace std;
struct maxf2v{vec locmax; double max; vec grad; mat hess;};
void loadmaxf2v(const int & order , string & out){
    int o=6, p;
    if(order<2) o=3;
    field<mat>result(o);
    result.load(out);
    result.print();
}

