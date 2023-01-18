
#!/bin/sh

# start the dialogpt enroot container on the cluster
IMAGE=/netscratch/enroot/nvcr.io_nvidia_tensorflow_22.03-tf2-py3.sqsh

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,$HOME:$HOME \
  --container-workdir=$HOME \
  --container-image=$IMAGE \
  --ntasks=1 \
  --nodes=1 \
  $*

