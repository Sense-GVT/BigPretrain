PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --mpi=pmi2 -p $1 -n$2 --gpu --cpus-per-task=4 --job-type normal \
"python -u -m prototype.solver.soco_solver --config config.yaml"
