//Produce vectors of available observations.
//dataf gives observations.
//vltab gives basic data on individual variables,
//and vardefs gives data on final variables for analysis.
//observed variables are used.
#include<armadillo>
using namespace arma;
using namespace std;
struct xsel
{
    bool all;
    uvec list;
};
struct vn
{
    vector<string>varnames;
    mat vars;
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
    string vardefname;;
    vector<string>varname;
    vector<vector<varlocs>::iterator> varnameit;
};
bool vdefsort(pair<vector<vardef>::const_iterator,bool> & a,
    pair<vector<vardef>::const_iterator,bool> & b) {return ((a.first)->vardefname<(b.first)->vardefname);};
bool validitycheck(const vardef & , const vector<double> & );
//Result is vector of pairs with first element a vector that provides the
//sequence numbers of variables presented and the the second element the sequence number of the observation.
vector<pair<vector<int>,int>> obspat(const vector<vn> & dataf, const vector<varlocs> & vltab, const vector<vardef> & vardefs)
{
//n is number of observations.
//m is number of observed variables.
//mm ia maximum  possible number of variables on a form.
//f is number of forms.
//i is an index for observations.
//j is an index for variables.
//k is an index for position.
//q is an index for observations within forms.
//r is an index for index within forms vector.
//s and t are distance measures
    int formno=0, f, h, i, j, k, m, n=0, q, r, s, t;
    vector<vardef>::const_iterator vardefit, vardefit1;
    pair<vector<pair<vector<vardef>::const_iterator,bool>>::iterator,
        vector<pair<vector<vardef>::const_iterator,bool>>::iterator> resultedp;
    vector<varlocs>::iterator vartabit;
    vector<vector<varlocs>::iterator>::const_iterator vartabit1;
//Maximum variable count on a form.
    f=dataf.size();
//mm is maximum possible number of observed variables on a form.
    vector<int>mm(f,0);
//mmit is used to iterate over mm.
    vector<int>::iterator mmit;
//formit iterates over forms within a variable table.
    vector<int>::const_iterator formit;
//vtabdef provides a mapping from vartab  to vardefs for observed variables.
    map<vector<varlocs>::iterator,vector<vardef>::const_iterator> vtabdef;
    for(vardefit=vardefs.cbegin();vardefit!=vardefs.cend();++vardefit)
    {
        vartabit=*vardefit->varnameit.cbegin();
        for(vartabit1=vardefit->varnameit.cbegin();vartabit1!=vardefit->varnameit.cend();++vartabit1)
            vtabdef.insert({*vartabit1,vardefit});
        if((vardefit->obs)&(vardefit->transform!='W')&(vardefit->transform!='I'))
        {
            for(formit=vartabit->forms.cbegin();formit!=vartabit->forms.cend();++formit)mm[*formit]++;
        }
        else{
            for(mmit=mm.begin();mmit!=mm.end();++mmit)(*mmit)++;
        }
    }
//datafit is used to iterate over forms.
    vector<vn>::const_iterator datafit;
    for(datafit=dataf.cbegin();datafit!=dataf.cend();++datafit)n+=datafit->vars.n_rows;
//result contains results of analysis.
    vector<pair<vector<int>,int>> result(n);
//resultit is used to iterate over result.
    vector<pair<vector<int>,int>>::iterator resultit;
//resulted is used for tentative results.
    vector<pair<vector<vardef>::const_iterator,bool>> resulted;
    pair<vector<vardef>::const_iterator,bool> resulted1;
    vector<pair<vector<vardef>::const_iterator,bool>>:: iterator resultedit, resultedit1;
    i=0;
    for(resultit=result.begin();resultit!=result.end();++resultit)
    {
        resultit->second=i;
        i++;
    }
//resultit1 iterates over first component of result.
    vector<int>::iterator resultit1;
//Variable name used in search.
    string name;
//Value of observed variable.
    vector<double> x;
    vector<double>::iterator xit;
//Control flags.
    bool flag, flag1;
//Go over observations to indicate which variables are observed and valid.
    resultit=result.begin();
    pair<vector<int>::const_iterator,vector<int>::const_iterator> formp;
//Loop over forms.
    mmit=mm.begin();
    for(datafit=dataf.cbegin();datafit!=dataf.cend();++datafit)
    {
//Within forms, go over observations.  q is position of observation within form.
        for(q=0;q<datafit->vars.n_rows;q++)
        {
//Maximum number of observations on form.
            resulted.resize(*mmit);
            resultedit=resulted.begin();
            for(vardefit=vardefs.cbegin();vardefit!=vardefs.cend();++vardefit)
            {
                if((vardefit->transform=='I')|(vardefit->transform=='W'))continue;
                if(vardefit->obs)
                {
                    vartabit=*vardefit->varnameit.cbegin();
                    formp=equal_range(vartabit->forms.cbegin(),vartabit->forms.cend(),formno);
                    if(formp.first<vartabit->forms.cend())
                    {
                        r=distance(vartabit->forms.cbegin(),formp.first);
                        x.resize(vardefit->varnameit.size());
                        xit=x.begin();
                        for(vartabit1=vardefit->varnameit.cbegin();vartabit1!=vardefit->varnameit.cend();++vartabit1)
                        {
                            vartabit=*vartabit1;
                            k=vartabit->positions[r];
                            *xit=datafit->vars(q,k);
                            ++xit;
                        }
                        flag=validitycheck(*vardefit, x);
//Break if any component invalid.
                        if(!flag)continue;
                    }
                }
                (*resultedit).first=vardefit;
                (*resultedit).second=true;
                ++resultedit;
            }
            s=distance(resulted.begin(),resultedit);
            resulted.resize(s);
//Eliminate variables with missing predictors.
            for(j=0;j<=s;j++)
            {
                flag1=true;
                t=0;
                for(resultedit=resulted.begin();resultedit!=resulted.end();++resultedit)
                {
                    vardefit=(*resultedit).first;
                    if((*resultedit).second)t++;
                    if(!vardefit->predits.empty()&(*resultedit).second)
                    {
                        for(vartabit1=vardefit->predits.cbegin();vartabit1!=vardefit->predits.cend();++vartabit1)
                        {
                            vardefit1=vtabdef[*vartabit1];
                            resulted1.first=vardefit1;
                            resulted1.second=true;
                            if(binary_search(resulted.begin(),resulted.end(),resulted1))continue;
                            (*resultedit).second=false;
                            flag1=false;
                            break;
                        }
                    }
                }
                ++resultedit;
                if(flag1)break;
            }
//All is stable.
            resultit->first.resize(t);
            if(t>0)
            {
                resultit1=resultit->first.begin();
                for(resultedit=resulted.begin();resultedit!=resulted.end();++resultedit)
                {
                    vardefit=(*resultedit).first;
                    if((*resultedit).second)
                    {
                        *resultit1=distance(vardefs.cbegin(),vardefit);
                        ++resultit1;
                    }
                }
            }
            ++resultit;
        }
        ++mmit;
    }
    return result;
}
