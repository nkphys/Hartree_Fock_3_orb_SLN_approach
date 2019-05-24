#ifndef Parameters_class
#define Parameters_class
#include "tensor_type.h"

class Parameters{

public:
    int lx, ly, ns, IterMax, RandomSeed;
    double Convergence_Error;
    int TBC_mx, TBC_my;
    int TBC_cellsX, TBC_cellsY;
    int lx_cluster, ly_cluster;
    double mus,Total_Particles,pi, mu_old;
    double J_Hund;
    double U_onsite;
    double Lambda_SOC;
    double Disorder_Strength, RandomDisorderSeed;
    bool PBC;
    bool PNICTIDES_HOPPING;

    bool Read_OPs;
    string File_OPs_in, File_OPs_out;

    Matrix<double> t2g_hopping_NN;
    Mat_1_doub Crystal_Field;

    bool Simple_Mixing;
    bool Broyden_Mixing;
    bool BroydenSecondMethodMixing;
    double w_minus1,wn;
    int BroydenSecondMethodCounter;
    double alpha_OP;

    double Temperature,beta,Eav,maxmoment;

    char Dflag;

    void Initialize(string inputfile_);
    double matchstring(string file,string match);
    string matchstring2(string file,string match);

};


void Parameters::Initialize(string inputfile_){

    double Simple_Mixing_double, Broyden_Mixing_double, BroydenSecondMethodMixing_double;
    double Read_OPs_double;
    string PBC_string, Pnictides_Hopping_string;

    cout << "____________________________________" << endl;
    cout << "Reading the inputfile: " << inputfile_ << endl;
    cout << "____________________________________" << endl;


    lx = int(matchstring(inputfile_,"Xsite"));
    ly = int(matchstring(inputfile_,"Ysite"));
    TBC_mx = int(matchstring(inputfile_,"TwistedBoundaryCond_mx"));
    TBC_my = int(matchstring(inputfile_,"TwistedBoundaryCond_my"));
    TBC_cellsX = int(matchstring(inputfile_,"TBC_cellsX"));
    TBC_cellsY = int(matchstring(inputfile_,"TBC_cellsY"));
    BroydenSecondMethodCounter = int(matchstring(inputfile_,"BroydenSecondMethodCounter"));

    ns = lx*ly;
    cout << "TotalNumberOfSites = "<< ns << endl;

    Total_Particles = matchstring(inputfile_,"Total No. of particles");
    cout << "TotalNumberOfParticles = "<< Total_Particles << endl;

    IterMax = int(matchstring(inputfile_,"No_of_SelfConsistency_iters"));
    Convergence_Error=matchstring(inputfile_,"Convergence_Error");
    RandomSeed = matchstring(inputfile_,"RandomSeed");
    RandomDisorderSeed = matchstring(inputfile_,"RandomDisorderSeed");
    Disorder_Strength = matchstring(inputfile_,"Disorder_Strength");
    J_Hund = matchstring(inputfile_,"J_HUND");
    U_onsite = matchstring(inputfile_,"U_Onsite");
    Lambda_SOC = matchstring(inputfile_, "Lambda_SOC");

    alpha_OP = matchstring(inputfile_,"alpha_OP");
    w_minus1 = matchstring(inputfile_,"w_minus1");
    wn = matchstring(inputfile_,"wn");



    Dflag = 'N';

    Simple_Mixing_double=double(matchstring(inputfile_,"Simple_Mixing"));
    Broyden_Mixing_double=double(matchstring(inputfile_,"Broyden_Mixing"));
    BroydenSecondMethodMixing_double=double(matchstring(inputfile_,"Broyden_Second_Method_Mixing"));


    if(BroydenSecondMethodMixing_double==1.0){
         BroydenSecondMethodMixing=true;
        Broyden_Mixing=false;
        Simple_Mixing=false;
    }
    else{
         BroydenSecondMethodMixing=false;
        if(Broyden_Mixing_double==1.0){
            Broyden_Mixing=true;
            Simple_Mixing=false;

        }
        else if(Broyden_Mixing_double==0.0){
            Broyden_Mixing=false;
            Simple_Mixing=true;
            cout<<"Broyden_Mixing and  BroydenSecondMethodMixing, both are 0(false). So Simple mixing is used"<<endl;

        }

    }



    Read_OPs_double=double(matchstring(inputfile_,"Read_initial_OPvalues"));
    if(Read_OPs_double==1.0){
        Read_OPs=true;
    }
    else{
        Read_OPs=false;
    }


    PBC_string=matchstring2(inputfile_,"PBC");
    if(PBC_string=="true"){
        PBC=true;
    }
    else{
        PBC=false;
    }

    Pnictides_Hopping_string=matchstring2(inputfile_,"Pnictides_Hopping");
    if(Pnictides_Hopping_string=="true"){
        PNICTIDES_HOPPING=true;
    }
    else{
        PNICTIDES_HOPPING=false;
    }


    File_OPs_in=matchstring2(inputfile_,"Read_initial_OPvalues_file");
    File_OPs_out=matchstring2(inputfile_,"Write_Final_OPvalues_file");

    string Nearest_Neigh_Hopping_t2g_basis_row0;
    string Nearest_Neigh_Hopping_t2g_basis_row1;
    string Nearest_Neigh_Hopping_t2g_basis_row2;

    Nearest_Neigh_Hopping_t2g_basis_row0=matchstring2(inputfile_, "Nearest_Neigh_Hopping_t2g_basis_row0");
    Nearest_Neigh_Hopping_t2g_basis_row1=matchstring2(inputfile_, "Nearest_Neigh_Hopping_t2g_basis_row1");
    Nearest_Neigh_Hopping_t2g_basis_row2=matchstring2(inputfile_, "Nearest_Neigh_Hopping_t2g_basis_row2");

    stringstream t2g_row0_stream(Nearest_Neigh_Hopping_t2g_basis_row0);
    stringstream t2g_row1_stream(Nearest_Neigh_Hopping_t2g_basis_row1);
    stringstream t2g_row2_stream(Nearest_Neigh_Hopping_t2g_basis_row2);

    t2g_hopping_NN.resize(3,3);
    for(int n=0;n<3;n++){
        t2g_row0_stream >> t2g_hopping_NN(0,n);
        t2g_row1_stream >> t2g_hopping_NN(1,n);
        t2g_row2_stream >> t2g_hopping_NN(2,n);
    }


    string Crystal_Field_t2g;
    Crystal_Field_t2g=matchstring2(inputfile_, "Crystal_Field_t2g");

    stringstream Crystal_Field_t2g_stream(Crystal_Field_t2g);


    Crystal_Field.resize(3);
    for(int n=0;n<3;n++){
        Crystal_Field_t2g_stream >> Crystal_Field[n];
    }

    pi=4.00*atan(double(1.0));
    Eav=0.0;

    Temperature=0.0001;
    //beta=(11605.0/Temperature);
     beta=(1.0/Temperature);

    mus=0.25;
    mu_old=0.25;
    cout << "____________________________________" << endl;
}


double Parameters::matchstring(string file,string match) {
    string test;
    string line;
    ifstream readFile(file);
    double amount;
    bool pass=false;
    while (std::getline(readFile, line)) {
        std::istringstream iss(line);
        if (std::getline(iss, test, '=') && pass==false) {
            // ---------------------------------
            if (iss >> amount && test==match) {
                // cout << amount << endl;
                pass=true;
            }
            else {
                pass=false;
            }
            // ---------------------------------
            if(pass) break;
        }
    }
    if (pass==false) {
        string errorout=match;
        errorout+="= argument is missing in the input file!";
        throw std::invalid_argument(errorout);
    }
    cout << match << " = " << amount << endl;
    return amount;
}

string Parameters::matchstring2(string file,string match) {

    string line;
    ifstream readFile(file);
    string amount;
    int offset;

    if(readFile.is_open())
    {
        while(!readFile.eof())
        {
            getline(readFile,line);

            if ((offset = line.find(match, 0)) != string::npos) {
                amount = line.substr (offset+match.length()+1);				}

        }
        readFile.close();
    }
    else
    {cout<<"Unable to open input file while in the Parameters class."<<endl;}




    cout << match << " = " << amount << endl;
    return amount;
}

#endif



