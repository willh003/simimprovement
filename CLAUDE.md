If you want documentation on a specific part of the implementation, check docs/, especially docs/claude

Whenever you do a large or important implementation, save information about it in organized markdown files within docs/claude. Make sure to keep this files clean and organized for yourself

You may not always be able to run commands using isaac sim python, because this requires that you are in an apptainer. Sometimes you will be in the apptainer, and will be capable of running commands. Always check first.

IMPORTANT: isaac sim does not crash or end gracefully (it will hang at the end of programs, requiring a "ctrl+]" command to terminate it). So if there haven't been any changes to the output for a while, it might be hanging. Make sure you periodically check its not hanging.

I have an alias in my bashrc that points isaacpy to /isaac-sim/python.sh, so in commands you give me you can use "isaacpy" instead.


If you need to run python (for quick scripts), you can use isaacpy if you are in an apptainer. If not, you should use this python: /gscratch/weirdlab/will/venvs/ogbench/bin/python (this one won't work for running isaac sim though).