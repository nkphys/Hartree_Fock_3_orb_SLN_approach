#include "ParametersEngine.h"
#include "Coordinates.h"
#include "MFParams.h"
#include "Hamiltonian.h"
#include "Observables.h"
#include "tensor_type.h"
#include <iomanip>

#ifndef SelfConsistencyEngine_H
#define SelfConsistencyEngine_H


class SelfConsistencyEngine{
public:
    SelfConsistencyEngine(Parameters& Parameters__, Coordinates& Coordinates__,
                          MFParams& MFParams__, Hamiltonian& Hamiltonian__,
                          Observables& Observables__)
        : Parameters_(Parameters__),Coordinates_(Coordinates__),
          MFParams_(MFParams__), Hamiltonian_(Hamiltonian__),
          Observables_(Observables__),
          lx_(Parameters_.lx), ly_(Parameters_.ly), ns_(Parameters_.ns)
    {

    }

    void RUN_SelfConsistencyEngine();
    double Prob (double muu, double mu_new);
    double ProbCluster (double muu, double mu_new);
    Parameters &Parameters_;
    Coordinates &Coordinates_;
    MFParams &MFParams_;
    Hamiltonian &Hamiltonian_;
    Observables &Observables_;
    const int lx_, ly_, ns_;

};

/*
 * ***********
 *  Functions in Class SelfConsistencyEngine ------
 *  ***********
*/

void SelfConsistencyEngine::RUN_SelfConsistencyEngine(){

    complex<double> zero(0.0,0.0);

    double muu_prev;
    double Curr_QuantE;
    double Prev_QuantE;
    double Curr_ClassicalE;
    double Prev_ClassicalE;

    cout << "Temperature = " << Parameters_.Temperature<<" is being done"<<endl;


    string File_Out_progress;

    double initial_mu_guess;
    int n_states_occupied_zeroT;

    //starting with a random guess
    File_Out_progress = "output.txt";
    ofstream file_out_progress(File_Out_progress.c_str());


    file_out_progress<< "Maximum no of self consistency iterations = "<<Parameters_.IterMax<<"."<<endl;
    file_out_progress<<"Convergence error targetted = "<<Parameters_.Convergence_Error<<endl;

    Parameters_.Dflag='V'; // flag to calculate only Eigenvalue


    file_out_progress<<"Iter"<<setw(15)<<
                       "Error_OP"<<setw(15)<<
                       "mu"<<setw(17)<<
                       "E_CL"<<setw(17)<<"E_Quantum"<<endl;



    /*
    Prev_ClassicalE = Hamiltonian_.GetCLEnergy();
    Hamiltonian_.InteractionsCreate();
    Hamiltonian_.Diagonalize(Parameters_.Dflag);
    n_states_occupied_zeroT=Parameters_.ns*Parameters_.Fill*2.0;
    initial_mu_guess=0.5*(Hamiltonian_.eigs_[n_states_occupied_zeroT-1] + Hamiltonian_.eigs_[n_states_occupied_zeroT]);
    Parameters_.mus=Hamiltonian_.chemicalpotential(initial_mu_guess,Parameters_.Fill);
    Prev_QuantE = Hamiltonian_.E_QM();
    Hamiltonian_.copy_eigs(1);
    cout<<"Initial Classical Energy = "<<Prev_ClassicalE<<endl;
    cout<<"Initial Quantum Energy = "<<Prev_QuantE<<endl;
    cout<<"Initial mu="<<Parameters_.mus<<endl;
    */


    double Error_OP=100;

    for(int count=0;count<Parameters_.IterMax;count++){
        if(
                Error_OP>Parameters_.Convergence_Error
                ){



            Hamiltonian_.InteractionsCreate();
            // Hamiltonian_.Ham_.print();
            //Hamiltonian_.Check_Hermiticity();
            Hamiltonian_.Diagonalize(Parameters_.Dflag);
//             if(count==0){
//                 n_states_occupied_zeroT=Parameters_.Total_Particles;
//                 initial_mu_guess=0.5*(Hamiltonian_.eigs_[n_states_occupied_zeroT-1] + Hamiltonian_.eigs_[n_states_occupied_zeroT]);
//                 Parameters_.mus=Hamiltonian_.chemicalpotential(initial_mu_guess,Parameters_.Total_Particles);
//             }




            //calculating mu
            n_states_occupied_zeroT=Parameters_.Total_Particles;
            initial_mu_guess=0.5*(Hamiltonian_.eigs_[n_states_occupied_zeroT-1] + Hamiltonian_.eigs_[n_states_occupied_zeroT]);
            Parameters_.mus=Hamiltonian_.chemicalpotential(initial_mu_guess,Parameters_.Total_Particles);
            Parameters_.mu_old=Parameters_.mus;

            //Getting order params using mu for N particles
            Observables_.Calculate_Order_Params();




            Curr_QuantE = Hamiltonian_.E_QM();
            Curr_ClassicalE = Hamiltonian_.GetCLEnergy();


            Observables_.Get_OrderParameters_diffs();

            Error_OP=Observables_.Error_OP_;



            file_out_progress<<setprecision(15)<<count<<setw(30)<<
                               Error_OP<<setw(30)<<
                               Parameters_.mus<<setw(32)<<
                               Curr_ClassicalE<<setw(32)<<Curr_QuantE<<endl;


            if(Parameters_.BroydenSecondMethodMixing==true){
                Observables_.Update_OrderParameters_Modified_Broyden_in_ComplexSpace(count);
            }
            else{
                Observables_.Update_OrderParameters(count);
            }


        }

    }


    //Printing eigenvalues
    /*string File_Out_eigen = "Eigenvalues.txt";
    ofstream file_eigen_out(File_Out_eigen.c_str());
    for(int n=0;n<Hamiltonian_.eigs_.size();n++){
        file_eigen_out<<n<<"    "<<Hamiltonian_.eigs_[n]<<endl;
    }*/


    Observables_.Calculate_Local_n_orb_resolved();

    string File_Out_Local_orb_densities = "Local_orb_spin_resolved_densities.txt";
    ofstream file_out_Local_orb_densities(File_Out_Local_orb_densities.c_str());
    file_out_Local_orb_densities<<"#site     ix    iy    yz(orb=0),up   yz(orb=0),dn   xz(orb=1),up   xz(orb=1),dn   xy(orb=2),up   xy(orb=2),dn"<<endl;
    int c1;
    int site_i;
    for(int iy=0;iy<ly_;iy++){
        for(int ix=0;ix<lx_;ix++){
            site_i=Coordinates_.Nc(ix,iy);
            file_out_Local_orb_densities<<site_i<<setw(15)<<ix<<setw(15)<<iy;
            for(int orb=0;orb<3;orb++){
                for(int spin=0;spin<2;spin++){

                    c1=Coordinates_.Nc_dof(site_i,orb+3*spin);
                    file_out_Local_orb_densities<<setw(15)<<Observables_.Local_n_orb_resolved[c1];
                }
            }
            file_out_Local_orb_densities<<endl;

        }
    file_out_Local_orb_densities<<endl;
    }




    string File_Out_Local_OP = Parameters_.File_OPs_out;
    ofstream file_out_Local_OP(File_Out_Local_OP.c_str());
    file_out_Local_OP<<"#site     ix    iy    Sz[site]   Sy[site]   Sx[site]    Lz[site]   Ly[site]   Lx[site]   n[site]"<<endl;




    for(int iy=0;iy<ly_;iy++){
        for(int ix=0;ix<lx_;ix++){
            site_i=Coordinates_.Nc(ix,iy);
            file_out_Local_OP<<site_i<<setw(15)<<ix<<setw(15)<<iy<<setw(15)<<
                               MFParams_.Local_Sz[site_i]<<setw(15)<<MFParams_.Local_Sx[site_i]<<setw(15)<<
                               MFParams_.Local_Sy[site_i]<<setw(15)
                            <<MFParams_.Local_Lz[site_i]<<setw(15)<<MFParams_.Local_Lx[site_i]<<setw(15)<<
                              MFParams_.Local_Ly[site_i]<<setw(15)
                           <<MFParams_.Local_n[site_i]<<endl;
        }
        file_out_Local_OP<<endl;
    }




} // ---------



#endif // SelfConsistencyEngine_H
