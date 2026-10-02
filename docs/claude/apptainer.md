## Opening an apptainer
To run with isaacsim, you need a gpu, and you need to use an apptainer. To enter an apptainer, you can use 
```/gscratch/weirdlab/will/polaris_overlay.sh.```


I'm not sure exactly how you would enter an apptainer if you weren't on an interactive session. Next time you figure that out, you should replace this section with that info.


To get gpus, you can use these commands (defined in ~\.bashrc):
show_weird_compute: outputs the compute
compute [flags]
- in general, you need to run on an l40 or l40s if you want isaac to work (preferably l40). other things (like general inference) can use a40
- prefer using -w (ie, weirdlab gpus). otherwise, ckpt gpus are alright, but they can be preempted. 

if possible, when running RL training, you should start pi0 on a separate GPU than the isaac env, so you have space for more environments and memory.