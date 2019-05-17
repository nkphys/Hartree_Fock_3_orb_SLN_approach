#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <random>
#include <complex>
#include <cmath>
#include <cassert>
using namespace std;
#include "Matrix.h"
#include "ParametersEngine.h"
#include "Coordinates.h"
#include "MFParams.h"
#include "Hamiltonian.h"
#include "Observables.h"
#include "SelfConsistencyEngine.h"
#include "random"


int main(int argc, char *argv[]) {
    if (argc<2) { throw std::invalid_argument("USE:: executable inputfile"); }
    string inputfile = argv[1];

    bool check_Non_Int=false;


    Parameters Parameters_;
    Parameters_.Initialize(inputfile);

    Coordinates Coordinates_(Parameters_.lx, Parameters_.ly);

    mt19937_64 Generator_(Parameters_.RandomSeed); //for random fields
    mt19937_64 Generator2_(Parameters_.RandomDisorderSeed); //for random disorder


    MFParams MFParams_(Parameters_,Coordinates_,Generator_, Generator2_);


    Hamiltonian Hamiltonian_(Parameters_,Coordinates_,MFParams_);


    Observables Observables_(Parameters_,Coordinates_,MFParams_,Hamiltonian_);




    if(check_Non_Int==true){

        //Parameters_.J_HUND=0.0;
        // Hamiltonian_.InteractionsCreate();
        //  Hamiltonian_.Ham_.print();
        // Hamiltonian_.Check_up_down_symmetry();
        //Hamiltonian_.Check_Hermiticity();
        // Hamiltonian_.Diagonalize('V');
        // int temp=Parameters_.Total_Particles;
        //cout<<"mu for n=4 = "<<0.5*(Hamiltonian_.eigs_[temp-1] + Hamiltonian_.eigs_[temp])<<"   "<<
        // Hamiltonian_.eigs_[temp-1]<<"   "<<Hamiltonian_.eigs_[temp]<<endl;

        // double mu_temp = 0.5*(Hamiltonian_.eigs_[temp-1] + Hamiltonian_.eigs_[temp]);
        // Parameters_.mus = Hamiltonian_.chemicalpotential(mu_temp, Parameters_.Total_Particles);
        // cout<<"Energies near the Fermi level:"<<Hamiltonian_.eigs_[temp-1]<<"   "<<Hamiltonian_.eigs_[temp]<<endl;
        //  cout<<"Chemical potential = "<<Parameters_.mus<<endl;
        //  double Quantum_E=Hamiltonian_.E_QM();
        // double Classical_E=Hamiltonian_.GetCLEnergy();
        // cout<<setprecision(9);
        //  cout<<"Total_Energy = "<<Quantum_E+Classical_E<<endl;
        //double mu = chemicalpotential(0.5, temp);
        // Observables_.Get_Non_Interacting_dispersion();
        //Hamiltonian_.Ham_.print();
        //Observables_.Calculate_Akw();
        //Observables_.Calculate_Akw_at_w(mu);
        // Observables_.Calculate_Nw();


    }

    else{

        cout<<setprecision(9);
        SelfConsistencyEngine SelfConsistencyEngine_(Parameters_,Coordinates_,MFParams_,Hamiltonian_,Observables_);
        SelfConsistencyEngine_.RUN_SelfConsistencyEngine();


        Observables_.Calculate_Nw_t2g();
        Observables_.Calculate_SpinSpincorrelations_Smartly();
        Observables_.Calculate_Orbitalcorrelations_Smartly();
       Observables_.Calculate_Akw_t2g();
        //Observables_.Calculate_Optical_Conductivity();


    }




    cout << "--------THE END--------" << endl;
} // main
