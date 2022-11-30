#!/bin/bash

export ANSIBLE_HOST_KEY_CHECKING=False

MEMCACHED_WORKER_THREADS=10
MEMCACHED_MEMORY_LIMIT_MB=16384
MEMCACHED_PIN_WORKER_THREADS=true

build_memcached () {
  if [[ -f "memcached/memcached" ]]
  then 
    return
  fi
  git clone --branch 1.6.12 https://github.com/memcached/memcached.git
  pushd memcached
  ./autogen.sh
  ./configure
  make -j4
  popd
}

build_mcperf () {
  if [[ -f "memcache-perf/mcperf" ]]
  then 
    return
  fi
  git clone https://github.com/shaygalon/memcache-perf
  pushd memcache-perf
  git checkout 0afbe9b
  make -j4
  popd
}

install_ansible_python () {
  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8
  pip3 install ansible
}

install_dep () {
  #sudo apt update
  sudo apt-add-repository ppa:ansible/ansible -y
  sudo apt update
  sudo apt install ansible -y
  ansible-playbook -i hosts ansible/install_dep.yml
}

build () {
  build_memcached
  build_mcperf
  pushd ~
  tar -czf mcperf.tgz mcperf
  popd
 }	

build_install () {
  install_dep
  build
  ansible-playbook -v -i hosts ansible/install.yml
}

run_profiler () {
  ansible-playbook -v -i hosts ansible/profiler.yml --tags "run_profiler"
}

kill_profiler () {
  ansible-playbook -v -i hosts ansible/profiler.yml --tags "kill_profiler"
}

run_remote () {
  vars="WORKER_THREADS=${MEMCACHED_WORKER_THREADS} MEMORY_LIMIT_MB=${MEMCACHED_MEMORY_LIMIT_MB} PIN_THREADS=${MEMCACHED_PIN_WORKER_THREADS}"
  ansible-playbook -v -i hosts ansible/mcperf.yml -e "$vars" --tags "run_memcached,run_agents"
}

kill_remote () {
  vars="WORKER_THREADS=${MEMCACHED_WORKER_THREADS} MEMORY_LIMIT_MB=${MEMCACHED_MEMORY_LIMIT_MB} PIN_THREADS=${MEMCACHED_PIN_WORKER_THREADS}"
  ansible-playbook -v -i hosts ansible/mcperf.yml -e "$vars" --tags "kill_memcached,kill_agents"
}

run_server () {
  vars="WORKER_THREADS=${MEMCACHED_WORKER_THREADS} MEMORY_LIMIT_MB=${MEMCACHED_MEMORY_LIMIT_MB} PIN_THREADS=${MEMCACHED_PIN_WORKER_THREADS}"
  ansible-playbook -v -i hosts ansible/mcperf.yml -e "$vars" --tags "run_memcached"
}

kill_server () {
  vars="WORKER_THREADS=${MEMCACHED_WORKER_THREADS} MEMORY_LIMIT_MB=${MEMCACHED_MEMORY_LIMIT_MB} PIN_THREADS=${MEMCACHED_PIN_WORKER_THREADS}"
  ansible-playbook -v -i hosts ansible/mcperf.yml -e "$vars" --tags "kill_memcached"
}

check_status () {
  ansible-playbook -v -i hosts ansible/mcperf.yml --tags "status"
}

"$@"
