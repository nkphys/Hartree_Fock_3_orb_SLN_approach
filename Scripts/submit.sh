mkdir Runs
cd Runs

for J_exchange in 1.15 #1.5 2.0 1.0
do
mkdir J_${J_exchange}
cd J_${J_exchange}

for Dis in 1.5 #0.0001 0.0005 #0.001 0.002 0.005 0.01 #1.1 0.75 0.85 1.3 0.05 0.025 0.075 #0.1 0.2 0.3 0.4 0.5 0.6 0.65 0.7 0.8 0.9 1.0 1.2 1.4 2.0 3.0 4.0 #0.1 0.2 0.3 0.4 0.5 0.6 0.65 0.7 0.8 0.9 1.0 1.2 1.4

#1.0 #0.5 1.2 1.4  #0.1 0.3 0.6 0.7 0.8 0.9 #0.1 0.5 1.0 2.0 4.0 6.0 8.0 10.0 #1.0 2.0 4.0 8.0 10.0
do
mkdir Dis_${Dis}
cd Dis_${Dis}

for unbiased_OP_seed in 101
do
mkdir unbiased_OP_seed_${unbiased_OP_seed}
cd unbiased_OP_seed_${unbiased_OP_seed}

for Disorder_seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 #21 22 23 24 25 26 27 28 29 30 31 32 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
do
mkdir Disorder_seed_${Disorder_seed}
cd Disorder_seed_${Disorder_seed}


input=input_run.inp

cp ../../../../../input_template.inp $input
cp ../../../../../SelfConsistent .
#cp ../../../../../Runs/J_${J_exchange}/Dis_${Dis}/unbiased_OP_seed_${unbiased_OP_seed}/Disorder_seed_${Disorder_seed}/output_Local_* .


########################################
sed -i -e "s/OP_SEED_VALUE/${unbiased_OP_seed}/g" $input
sed -i -e "s/DISORDER_SEED_VALUE/${Disorder_seed}/g" $input
sed -i -e "s/DISORDER_STRENGTH/${Dis}/g" $input
sed -i -e "s/VALUE_J_EXC/${J_exchange}/g" $input
########################################


time ./SelfConsistent input_run.inp > out_run.txt

cd ..
done

cd ..
done

cd ..
done

cd ..
done
