<p align="center">
  <img src="assets/logo-classicblue-800px.png?raw=true" alt="Intel Logo" width="250"/>
</p>

# Distributed Training on AWS with Pytorch 

© Copyright 2023, Intel Corporation


## Introduction
This repository shows how to fine-tune GPT2-small (124M) model on [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/) in a distributed setting on 3rd Gen or 4th Gen Xeon CPUs. [nanoGPT](https://github.com/karpathy/nanoGPT) was used for GPT2 implementation and [OpenWebText dataset](https://huggingface.co/datasets/Skylion007/openwebtext) was downloaded from HuggingFace Hub.  

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setting things up](#setup)
4. [Fine-tuning on Single CPU](#fine-tuning-on-single-cpu)
5. [Preparing for Distributed training](#preparing-for-distributed-training)
6. [Fine-tuning on Multiple CPUs](#fine-tuning-on-multiple-cpus)

## Overview
LLMs (Large Language Models) are everywhere, but in many cases, you don't need the full capability of the GPT-4 model. Additionally, when you have a specific task at hand, the performance of the GPT-4 model might not be optimal. Often, fine-tuning a small LLM on your dataset is sufficient. In this guide, you will learn how to fine-tune the GPT2-small model on a cluster of Xeon CPUs on AWS.

Here are the steps to accomplish the fine-tuning GPT2-small model on OpenWebtext dataset:
1. **Install dependencies**: Begin by installing the necessary dependencies and ensure that all required libraries and tools are setup correctly.
2. **Download and prepare the dataset**: Obtain the OpenWebText dataset from HuggingFace Hub. Preprocess and format the data appropriately for compatibility with nanoGPT implementation. Optionally, save both the raw and preprocessed data in S3 Bucket.
3. **Fine-tune on Single CPU**: Test the fine-tuning script on a single CPU to understand the basic workflow and to make all dependencies are installed correctly.
4. **Set things up for Distributed Training**: Configuring the necessary infrastructure like EC2 instances, Security Groups, etc. for distributed training on multiple CPUs. This is the most important step.
5. **Fine-tune on Multiple CPUs**: Once the distributed training environment is ready, perform fine-tuning on the cluster of Xeon CPUs to train the model quickly.

[Back to Table of Content](#table-of-contents)


## Prerequisites

Before proceeding, ensure you have an AWS account and the necessary permissions to create AMIs, security groups, and launch multiple EC2 instances.

For fine-tuning, in this guide, we used 3x [*m6i.4xlarge* instances](https://aws.amazon.com/ec2/instance-types/m6i/) with an Ubuntu 20.04 AMI and 250 GB Storage each. However, if you have access to 4th Gen Sapphire Rapids Xeon CPUs ([*R7iz*](https://aws.amazon.com/ec2/instance-types/r7iz/)) on AWS, you can also use those for fine-tuning. To maximize performance during fine-tuning, we recommend using `bfloat16` precision, especially when using 4th Gen Xeon CPUs.

To verify if the AMX instruction set is supported on your system, you can run the following command in the terminal:

```bash
lscpu | grep amx
```

If your system supports the AMX instruction set, you should see the following flags:

```
amx_bf16 amx_tile amx_int8
```

These flags indicate that the AMX instructions are available on your system, which is essential for leveraging AMX optimizations during fine-tuning or other computations.

By following this guide, you will be able to effectively fine-tune your model on either [*m6i.4xlarge* instances](https://aws.amazon.com/ec2/instance-types/m6i/) or the newer 4th Gen Xeon [*R7iz*](https://aws.amazon.com/ec2/instance-types/r7iz/) CPUs.

[Back to Table of Content](#table-of-contents)


## Setup

To set up the necessary environment for fine-tuning the GPT2-small model, follow these steps:

Update the package manager and install [tcmalloc](https://github.com/google/tcmalloc) for extra performance

```bash
sudo apt update
sudo apt install libgoogle-perftools-dev -y
```

Create a virtual environment and activate it

```bash
# Creating 
sudo apt-get install python3-pip -y
pip install pip --upgrade
export PATH=/home/ubuntu/.local/bin:$PATH
pip install virtualenv

# Activating
virtualenv cluster_env
source cluster_env/bin/activate
```

Install PyTorch, IPEX, and oneCCL in the virtual environment 

```bash
pip3 install torch==1.13.0+cpu -f https://download.pytorch.org/whl/cpu
pip3 install intel_extension_for_pytorch==1.13.0+cpu -f https://developer.intel.com/ipex-whl-stable-cpu
pip3 install oneccl_bind_pt==1.13.0+cpu -f https://developer.intel.com/ipex-whl-stable-cpu
```

> **Note**: If you encounter an error indicating that no compatible versions are found for the Intel packages, download the **appropriate** `whl` files from [here](https://www.intel.com/content/dam/develop/external/us/en/documents/ipex/whl-stable-cpu.html) and install them using the following commands:

```bash
# Installing IPEX
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/cpu/intel_extension_for_pytorch-1.13.0%2Bcpu-cp310-cp310-linux_x86_64.whl
pip3 install intel_extension_for_pytorch-1.13.0+cpu-cp310-cp310-linux_x86_64.whl
rm intel_extension_for_pytorch-1.13.0+cpu-cp310-cp310-linux_x86_64.whl

# Installing oncCCL
wget https://intel-extension-for-pytorch.s3.amazonaws.com/torch_ccl/cpu/oneccl_bind_pt-1.13.0%2Bcpu-cp310-cp310-linux_x86_64.whl
pip3 install oneccl_bind_pt-1.13.0+cpu-cp310-cp310-linux_x86_64.whl
rm oneccl_bind_pt-1.13.0+cpu-cp310-cp310-linux_x86_64.whl
```

(Optional) Install AWS CLI if you want to upload your dataset and processed files to S3

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -r awscliv2.zip aws
aws --version
aws configure
```

These steps will prepare the necessary environment and tools for the subsequent fine-tuning process.

[Back to Table of Content](#table-of-contents)


## Fine-tuning on Single CPU

To begin the fine-tuning process in a single CPU, follow these steps:

Clone this repo and install its dependencies:

```bash
git clone https://github.com/intel-innersource/frameworks.ai.infrastructure.distributed-training-aws-pytorch.git distributed-training-aws-pytorch
cd distributed-training-aws-pytorch
pip3 install -r requirements.txt
```

Next, download and process the OpenWebText dataset by running the following command:

> **Note**: The script also uploads both the raw data and processed files to S3. Ensure that you have prepared a S3 bucket before running this script. You also need to updated the bucket name in the code. To disable the upload, set `UPLOAD_TO_S3 = False` in `prepare.py` file.

```bash
python data/openwebtext/prepare.py
```

This script will download the OpenWebText dataset from Hugging Face Hub, tokenize the text using `tiktoken`, and save the processed data (token index) as BIN files. It can take anywhere around 1-3 hours depending on your CPU.


 For future use on other systems, you can directly download the processed BIN files from S3 by executing `download.py` script. Follow the previous steps, and instead of running `prepare.py`, execute the `download.py` script. Make sure to update the S3 bucket name and processed files path in `download.py` file.

Once, you have the BIN files ready, generate the `accelerate` config file by running the following command:

```bash
$ accelerate config --config_file ./single_config.yaml
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:yes
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU? [yes/NO]:yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: NO
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
```
You will be prompted to answer a series of questions (just like the ones you see above). Choose the appropriate option based on your compute environment and preferences. For example, select `bf16` on 4th Gen Xeon CPUs otherwise select `fp16`. This will generate a config file and save it as `single_config.yaml` in your CWD. 

We are now ready to start fine-tuning the GPT-2 Small model. Use the `accelerate launch` command along with the generated config file. For maximum performance, set `OMP_NUM_THREADS` to the number of cores in your system.

```bash
OMP_NUM_THREADS=16 accelerate launch --config_file ./single_config.yaml main.py
```

This command will initiate the fine-tuning process with 16 threads, utilizing the settings specified in the `single_config.yaml` file.

[Back to Table of Content](#table-of-contents)


## Preparing for Distributed training

Now that we have this running in a single system, lets try to run it on multiple systems. To prepare for distributed training and ensure a consistent setup across all systems, follow these steps:

1. **Create an AMI**: Start by creating an Amazon Machine Image (AMI) from the existing instance where you have successfully run the fine-tuning on a single system. This AMI will capture the entire setup, including the dependencies, configurations, codebase, and dataset. To create an AMI, refer [Create a Linux AMI from an instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html#:~:text=EBS%20volumes.-,Create%20a%20Linux%20AMI%20from%20an%20instance,-You%20can%20create).

2. **Security Group**: While waiting for the AMI creation, let's continue by creating a security group that enables communication among the member nodes. This security group should be configured to allow inbound and outbound traffic on the necessary ports for effective communication between the master node and the worker nodes.

    In the security group configuration, ensure that you have allowed *all* traffic originating from the security group itself. This setting allows seamless communication between the instances within the security group.

    Please refer to the following screenshot as an example:

    ![Security Group Configuration](assets/sg.png)

    By setting up the security group in this manner, you ensure that all necessary traffic can flow between the master node and the worker nodes during distributed training.

3. **Launch new instances**: Use the created AMI to launch new instances, specifying the desired number of instances based on the number of systems you want to use for distributed training. This ensures that all the instances have the same environment and setup.

4. **Passwordless SSH**: Set up passwordless SSH from the master node to all the worker nodes. To enable password-less SSH, configure the master instance's SSH public key to be authorized on all other nodes. This will ensure SSH access without prompts between the master and worker nodes. To enable passwordless SSH, follow these steps:

    1. **Verify SSH Access**: First, check if you can SSH into the other nodes from the master node. Use the private IP address and the appropriate username for each node.

        ```bash
        ssh <username>@<ip-address>
        ```

        Successful SSH connections will indicate that the inbound rules of the security group are correctly set up. In case of any issues, check the network settings.

    2. **Generate SSH Key Pair**: On the master node, run the following command to generate an SSH key pair:

        ```bash
        ssh-keygen
        ```

        You will be prompted to enter a passphrase for the key. You can choose to enter a passphrase or leave it blank for no passphrase. For simplicity in this guide, it is recommended to leave it blank. The key pair will be generated and saved in the `~/.ssh` directory, with two files: `~/.ssh/id_rsa` (private key) and `~/.ssh/id_rsa.pub` (public key). For security, set appropriate permissions on the private key:

        ```bash
        chmod 600 ~/.ssh/id_rsa
        ```

    3. **Propagate the Public Key to Remote Systems**: To transfer the public key to the remote hosts, use the `ssh-copy-id` command. If password authentication is currently enabled, this is the easiest way to copy the public key:

          ```bash
          ssh-copy-id <username>@<private-ip-address>
          ```

          This command will copy the public key to the specified remote host. You will have to run this command from the master node to copy the public key to all other nodes.

    4. **Verify Passwordless SSH**: After copying the public key to all nodes, verify that you can connect using the key pair:

          ```bash
          ssh <username>@<private-ip-address>
          ```

          If you can successfully log in without entering a password, it means passwordless SSH is set up correctly.

    By following above steps, you will establish passwordless SSH between the master node and all worker nodes, ensuring smooth communication and coordination during distributed training.

Next, to continue setting up the cluster, you will need to edit the SSH configuration file located at `~/.ssh/config` on the master node. The configuration file should look like this:

```plaintext
Host 10.0.*.*
   StrictHostKeyChecking no

Host node1
    HostName 10.0.10.251
    User ubuntu

Host node2
    HostName 10.0.11.189
    User ubuntu
```

The `StrictHostKeyChecking no` line disables strict host key checking, allowing the master node to SSH into the worker nodes without prompting for verification.

With these settings, you can now use `ssh node1` or `ssh node2` to connect to any node without any additional prompts.

Additionally, on the master node, you will create a host file (`hosts`) that includes the names of all the nodes you want to include in the training process, as defined in the SSH configuration above. Use `localhost` for the master node itself as you will launch the training script from the master node. The `hosts` file will look like this:

```plaintext
localhost
node1
node2
```

This setup will allow you to seamlessly connect to any node in the cluster for distributed training.

[Back to Table of Content](#table-of-contents)


## Fine-tuning on Multiple CPUs


Next, we need to generate a new `accelerate`` config for multi-CPU setup. Run the following command:
```bash
$ accelerate config --config_file ./multi_config.yaml
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-CPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 3
-----------------------------------------------------------------------------------------------------------------------------------------------------------
What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 36.112.23.24
What is the port you will use to communicate with the main process? 29500
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: yes
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU? [yes/NO]:yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
How many CPU(s) should be used for distributed training? [1]:1
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16

```

Just like before, you will be prompted with several question to configure the multi-CPU setup. Select the appropriate answers based on your environment. This will generate a new config file named `multi_config.yaml` in your current working directory.

To train PyTorch models in a distributed setting on Intel hardware, we utilize Intel's MPI (Message Passing Interface) implementation. This implementation provides flexible, efficient, and scalable cluster messaging on Intel® architecture. The Intel® oneAPI HPC Toolkit includes all the necessary components, including `oneccl_bindings_for_pytorch`, which is installed alongside the MPI toolset.

To use `oneccl_bindings_for_pytorch`, you simply need to source the environment by running the following command:

```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

This command sets up the environment variables required for utilizing `oneccl_bindings_for_pytorch` and enables distributed training using Intel MPI. 

> **Note**: In a distributed setting, `mpirun` can be used to run any program, not just for distributed training. It allows you to execute parallel applications across multiple nodes or machines, leveraging the capabilities of MPI (Message Passing Interface).

Finally, it's time to run the fine-tuning process on multi-CPU setup. Execute the following command:
```bash
mpirun -f ~/hosts -n 3 -ppn 1 -genv OMP_NUM_THREADS=16 -genv LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so" accelerate launch --config_file ./multi_config.yaml main.py 
```

The `mpirun` command runs the fine-tuning process on multiple-machine (-n 3) with one process per machine (-ppn 1). We set the number of OpenMP threads to 16 (-genv OMP_NUM_THREADS=16) and used tcmalloc for improved performance (-genv LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so"). 

The `mpirun` command you provided runs the fine-tuning process on multiple machines with specific configurations. Here are some important points to consider when deciding the values for various `mpirun` parameters:
- `-n`: This parameter represents the number of CPUs or nodes. Typically, it is set to the number of nodes you are using. However, in the case of bare metal instances with two CPUs per board, you would use `2n` to account for the multiple CPUs on each node.
- `-ppn`: The "process per node" parameter determines how many training jobs you want to start on each CPU or node. If you set `-ppn` to 3, for example, it means that three instances of the command `accelerate launch --config_file ./multi_config.yaml main.py` will be executed on each CPU or node.
- `-genv`: This argument allows you to set an environment variable that will be applied to all processes. In your case, you used it to set the `OMP_NUM_THREADS` environment variable to control the number of cores that PyTorch will utilize.
- `OMP_NUM_THREADS`: The `OMP_NUM_THREADS` environment variable specifies the number of CPU cores that PyTorch will utilize. If you have set `-ppn` to 2 (indicating two training jobs per CPU or node), you may want to divide the total number of cores available by two and use that value as the number of threads for each training job.

Consider these factors when determining the values for `mpirun` parameters to ensure optimal performance and resource utilization in your distributed training setup.

Unfortunately, `mpirun` does not print any output (in particular `print` statements) until the finetuning process is complete. So, to monitor the progress, you can SSH into each system and use tools like `htop` to view CPU, memory and network utilization. Alternatively, you can check the AWS monitoring tab for each EC2 instance or create a cloudwatch dashboard for easier monitoring.

[Back to Table of Content](#table-of-contents)

