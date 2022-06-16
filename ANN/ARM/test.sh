#PBS -N test
#PBS -l nodes=4
pssh -h $PBS_NODEFILE -i "if [ ! -d \"/home/s2012174/MPI\" ];then mkdir -p \"/home/s2012174/MPI\"; fi" 1>&2
pscp -h $PBS_NODEFILE /home/s2012174/MPI/a.out /home/s2012174/MPI 1>&2
mpiexec -np 4 -machinefile $PBS_NODEFILE /home/s2012174/MPI/a.out
