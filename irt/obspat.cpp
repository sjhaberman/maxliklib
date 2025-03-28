//Produce vectors of missing data.
#include<armadillo>
using namespace arma;
using namespace std;
struct vn
{
    vector<string>varnames;
    mat vars;
};
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
//Result is vector of vectors of integers with one vector of integers equal 0 or 1
//for each observation, and that vector of integers with one element for each
//variable used, with 0 for missing and 1 for present.
vector<vector<int>>obspat(const vector<vn>&dataf,const vector<varlocs>&vartab,
    const vector<int> & obsvar )
{
//n is number of observations.
//m is number of variables.
//f is number of forms.
//i is an index for observations.
//j is an index for variables.
//k is an index for forms.
//q is an index for observations within forms.
//h is a variable position.
//r is a number of categories.
    int f, h, i, j, k, m, n, q, r;
//Result is vector of vectors of integers with 0 for missing and 1 for present.
    m=vartab.size();
    n=0;
    f=dataf.size();
    vector<vn>::const_iterator datafit;
    for(datafit=dataf.cbegin();datafit!=dataf.cend();++datafit)n+=datafit->vars.n_rows;
    vector<vector<int>> result(n);
    vector<vector<int>>::iterator resultit;
    vector<varlocs>::const_iterator vartabit;
    vector<int>::const_iterator obsvarit;
    vector<int>::const_iterator formit;
    vector<int>::const_iterator posit;
    vector<int>::iterator presentit;
    vector<vector<int>>formtable(m),postable(m);
    vector<vector<int>>::iterator formtableit,postableit;
    string name;
    double x;
//Go over forms.
    formtableit=formtable.begin();
    postableit=postable.begin();
    obsvarit=obsvar.cbegin();
    for(vartabit=vartab.begin();vartabit!=vartab.end();++vartabit)
    {
        formtableit->resize(f);
        postableit->resize(f);
        std::fill(formtableit->begin(),formtableit->end(),0);
        posit=vartabit->positions.cbegin();
        for(formit=vartabit->forms.cbegin();formit!=vartabit->forms.cend();++formit)
        {
            (*formtableit)[*formit]=*obsvarit;
            (*postableit)[*formit]=*posit;
            ++posit;
        }
        ++formtableit;
        ++postableit;
        ++obsvarit;
    }
    k=0;
    resultit=result.begin();
    i=0;        
    for(datafit=dataf.cbegin();datafit!=dataf.cend();++datafit)
    {
//Skip if form has no data.
        if(datafit->vars.n_rows==0)continue;
//Within forms, go over observations.
        for(q=0;q<datafit->vars.n_rows;q++)
        {
            resultit->resize(m+1); 
//Initialize as all zeros.
            std::fill(resultit->begin(),resultit->end(),0);
//Add observation number.
            (*resultit)[m]=i;
//Set if item on form and used.
            formtableit=formtable.begin();
            postableit=postable.begin();
            presentit=resultit->begin();
            j=0;
            for(vartabit=vartab.cbegin();vartabit!=vartab.cend();++vartabit)
            {
                if((*formtableit)[k]==1)
                {
                    h=(*postableit)[k];
//If item on form and used, see if observed and valid.
                    x=datafit->vars(q,h);
//First issue is whether item observed at all.
                    if(isfinite(x))
                    {
//See if item is categorical but not within bounds.
                        r=vartabit->catnames.size();
                        if(r==0||(x==floor(x)&&x<(double)r&&x>=0.0))
                        {
//See if item is Poisson but not a nonnegative integer.
                            if(vartabit->type!='P'||(x==floor(x)&&x>=0.0))
                            {
//Mark item as observed and valid.
                                *presentit=1;
                            }
                        }
                    }
                }                    
                ++presentit;
                ++formtableit;
                ++postableit;
                j++;     
            }
//Update observation.
            ++resultit;
            i++;
        }
        k++;
    }
    return result;
}
