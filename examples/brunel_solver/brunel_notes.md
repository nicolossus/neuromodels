To switch NEST into multi-threaded mode, you only have to add one line
to your simulation script:
nest.SetKernelStatus({'localnumthreads' : n})
