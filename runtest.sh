#!/bin/bash
device0=0
device1=1
device2=2
device3=3
if [ $# -ne 0 -a $# -ne 4 ]; then
echo "Please provide 4 device IDs"
exit
fi
if [ $# -eq 4 ]; then
device0=$1
device1=$2
device2=$3
device3=$4
fi
function load() {
  echo "loading"
  module load slurm;
  module load cuda/11.2;
  module load anaconda
}


function setup() {
  if [ ! -d benchmark ]; then
  mkdir benchmark
  fi
  cd benchmark
  if [ ! -f benchmark_plot.py ]; then
  ln -s ../benchmark_plot.py  benchmark_plot.py
  fi
  if [ ! -d data ]; then
  mkdir data
  fi
  if [ ! -d pictures ]; then
  mkdir pictures
  fi
  rm -rf data/benchmark.csv
  touch data/benchmark.csv
  printf "Setting,Runtime\n" > data/benchmark.csv
  testname=conv-data2
  if [ ! -f ${testname}.timer.cu ]; then
  ln -s ../src/out/transformer/pipeline_conv/${testname}/${testname}.timer.cu  ${testname}.timer.cu
  fi
  testname=conv-data4
  if [ ! -f ${testname}.timer.cu ]; then
  ln -s ../src/out/transformer/pipeline_conv/${testname}/${testname}.timer.cu  ${testname}.timer.cu
  fi
  testname=conv-nopipeline
  if [ ! -f ${testname}.timer.cu ]; then
  ln -s ../src/out/transformer/pipeline_conv/${testname}/${testname}.timer.cu  ${testname}.timer.cu
  fi
  testname=conv-4pipeline
  if [ ! -f ${testname}.timer.cu ]; then
  ln -s ../src/out/transformer/pipeline_conv/${testname}/${testname}.timer.cu  ${testname}.timer.cu
  fi
  testname=conv-8pipeline
  if [ ! -f ${testname}.timer.cu ]; then
  ln -s ../src/out/transformer/pipeline_conv/${testname}/${testname}.timer.cu  ${testname}.timer.cu
  fi
  cd ..
}

function compile() {
  nvcc -o $1  $1.cu \
  -I ../src/main/resources/headers \
  -I /usr/lib/x86_64-linux-gnu/openmpi/include \
  -L /usr/lib/x86_64-linux-gnu/openmpi/lib64 \
  -l nccl \
  -l mpi \
  -l cudnn \
  -l cublas
}

function sexec() {
  mpiexec --oversubscribe -np $*
}

# compile and run the generated test code
function test() {
  echo "testing start"
  cd benchmark
  csvfile=data/benchmark.csv
  testname=conv-data2.timer
  outputname='Data Parallism (2 GPU)'
  echo "compiling -- ${outputname}"
  compile $testname
  if [ $? -eq 0 ]; then
    echo "testing   -- ${outputname}"
    #sexec $testname 0
    res=$(sexec 2 $testname $device0 $device1 $device2 $device3)
    printf "Training loop time: ${res} sec\n\n"
    printf "${outputname},${res}\n" >> ${csvfile}
  fi

  testname=conv-data4.timer
  outputname='Data Parallism (4 GPU)'
  echo "compiling -- ${outputname}"
  compile $testname
  if [ $? -eq 0 ]; then
    echo "testing   -- ${outputname}"
    #sexec $testname 0
    res=$(sexec 4 $testname $device0 $device1 $device2 $device3)
    printf "Training loop time: ${res} sec\n\n"
    printf "${outputname},${res}\n" >> ${csvfile}
  fi

  testname=conv-nopipeline.timer
  outputname='Mixed Parallism (no pipeline)'
  echo "compiling -- ${outputname}"
  compile $testname
  if [ $? -eq 0 ]; then
    echo "testing   -- ${outputname}"
    #sexec $testname 0
    res=$(sexec 4 $testname $device0 $device1 $device2 $device3)
    printf "Training loop time: ${res} sec\n\n"
    printf "${outputname},${res}\n" >> ${csvfile}
  fi

  testname=conv-4pipeline.timer
  outputname='Mixed Parallism (4 pipelines)'
  echo "compiling -- ${outputname}"
  compile $testname
  if [ $? -eq 0 ]; then
    echo "testing   -- ${outputname}"
    #sexec $testname 0
    res=$(sexec 4 $testname $device0 $device1 $device2 $device3)
    printf "Training loop time: ${res} sec\n\n"
    printf "${outputname},${res}\n" >> ${csvfile}
  fi

  testname=conv-8pipeline.timer
  outputname='Mixed Parallism (8 pipelines)'
  echo "compiling -- ${outputname}"
  compile $testname
  if [ $? -eq 0 ]; then
    echo "testing   -- ${outputname}"
    #sexec $testname 0
    res=$(sexec 4 $testname $device0 $device1 $device2 $device3)
    printf "Training loop time: ${res} sec\n\n"
    printf "${outputname},${res}" >> ${csvfile}
  fi

  cd ..
}

function plot() {
  conda activate
  cd benchmark
  python3 ./benchmark_plot.py
  cd ..
}
load
setup
test
plot