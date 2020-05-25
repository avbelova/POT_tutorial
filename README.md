# OpenVINO Post-Training Optimization Toolkit (POT) Tutorial

## This tutorial helps to learn:
* What's POT configuration files structure
* How to run POT in simplified mode
* How to measure accuracy of FP32, INT8 models using POT config
* How to create your own POT config
* How to properly benchmark the workload

## Prerequisites:
1. [Intel(R) Distrubution of OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) 2020.2 for Linux.
2. Installed OpenVINO [Accuracy Checker](https://docs.openvinotoolkit.org/latest/_tools_accuracy_checker_README.html) Tool with all dependencies.
3. Installed OpenVINO [Post-Training Optimization Toolkit](https://docs.openvinotoolkit.org/latest/_README.html) with all dependencies.

## Getting Started:
#### You can run the titorial at your own machine:
1. Clone or download the repo:

`$ git clone https://github.com/avbelova/POT_tutorial.git`

2. Go to the tutorial directory:

`$ cd POT_tutorial`

3. Set up OprnVINO environment:

`$ source /opt/intel/openvino/bin/setupvars.sh`

4. Run Jypyter Notebook

`$ jupyter notebook`

   or Jupyter lab

`$ jupyter lab`

#### Other option is to run it on the DevCloud. In this case you don't need to install anything and source environment variables. You should just clone the repo, proceed to the directory and run Jupyter notebook/lab.
