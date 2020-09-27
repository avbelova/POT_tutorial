# OpenVINO Post-Training Optimization Toolkit (POT) Tutorial

[Post-Training Optimization Toolkit (POT)](https://docs.openvinotoolkit.org/latest/_README.html) is a part of [OpenVINO Toolkit](https://docs.openvinotoolkit.org/) which is responsible of applying different optimization techniques like quantization or sparsity. This repo helps you to easy undestand the tool in well-documented practical way.

## This tutorial consists of 2 parts:

#### [BASIC](https://github.com/avbelova/POT_tutorial/blob/master/POT_tutorial_BASICS.ipynb) which shows "standard" quantization workflow
* What's POT configuration files structure
* How to run POT in simplified mode
* How to measure accuracy of FP32, INT8 models using POT config
* How to create your own POT config
* How to properly benchmark the workload

#### [ADVANCED](https://github.com/avbelova/POT_tutorial/blob/master/POT_tutorial_ADVANCED.ipynb) covers complicated cases and focuses on custom functionality, discovering secrets of Accuracy Checker - the base of POT.
* Accuracy checker architecture
* How to analyze the model
* How to support custom dataset 
* How to add custom pre- and post- processing
* YoloV3 example
* DCSCN example

## Prerequisites:
1. Installed [Intel(R) Distrubution of OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) 2020.2 for Linux.
2. Installed OpenVINO [Accuracy Checker](https://docs.openvinotoolkit.org/latest/_tools_accuracy_checker_README.html) Tool with all dependencies.
3. Installed OpenVINO [Post-Training Optimization Toolkit](https://docs.openvinotoolkit.org/latest/_README.html) with all dependencies.

Note: the tutorial was created for Ubuntu OS but can also be adapted for other [supported OSes](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/system-requirements.html).

Note: the tutorial was checked for OpenVINO 2020.2 and might be not working with other versions.

## Getting Started:
#### You can run the tutorial at your own machine:
1. Clone or download the repo:

`$ git clone https://github.com/avbelova/POT_tutorial.git`

2. Go to the tutorial directory:

`$ cd POT_tutorial`

3. Set up OprnVINO environment:

`$ source /opt/intel/openvino/bin/setupvars.sh`

4. Run Jupyter Notebook

`$ jupyter notebook`

   or Jupyter lab

`$ jupyter lab`

#### The other option is to run it on the [Intel® DevCloud for the Edge](https://devcloud.intel.com/edge/). In this case you don't need to install anything and source environment variables. You should just clone the repo, proceed to the directory and run Jupyter notebook/lab.

As an addition, you can also [watch](https://www.youtube.com/watch?v=7XQAZBdA_wo&list=PLTseHiQLIfGM6ltiaeh9fL8qfxiE-u4fw&index=6) how to perform model quantization with POT via [Deep Learning Workbench](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Introduction.html).
