# FormulaZero
<b>Coming soon!</b>
<p align="center">
  <img src="assets/action.gif"/>
</p>


This is the reference implementation for our paper:

<em><b>FormulaZero: Distributionally Robust Online Adaptation via Offline Population Synthesis</b></em>
[PDF](https://arxiv.org/pdf/2003.03900.pdf)

[Aman Sinha*](http://amansinha.org), [Matthew O'Kelly*](http://www.mokelly.net/),[Hongrui Zheng*](https://hongruizheng.com), [Rahul Mangharam](), [John Duchi](http://stanford.edu/~jduchi/), [Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)

<em><b>Abstract:</b></em> Balancing performance and safety is crucial to deploying autonomous vehicles
in multi-agent environments. In particular, autonomous racing is a domain 
that penalizes safe but conservative policies, highlighting the need for
robust, adaptive strategies. Current approaches either make simplifying
assumptions about other agents or lack robust mechanisms for online
adaptation. This work makes algorithmic contributions to both
challenges. First, to generate a realistic, diverse set of opponents, we
develop a novel method for self-play based on replica-exchange Markov chain
Monte Carlo. Second, we propose a distributionally robust bandit
optimization procedure that adaptively adjusts risk aversion
relative to uncertainty in beliefs about opponents’ behaviors. We rigorously
quantify the tradeoffs in performance and robustness when approximating
these computations in real-time motion-planning, and we
demonstrate our methods experimentally on autonomous vehicles
that achieve scaled speeds comparable to Formula One racecars..

#### Citing

If you find this code useful in your work, please consider citing:

```
	@article{sinha2020formulazero,
	title={FormulaZero: Distributionally Robust Online Adaptation via Offline Population Synthesis},
	author={Sinha, Aman and O'Kelly, Matthew and Zheng, Hongrui and Mangharam, Rahul and Duchi, John and Tedrake, Russ},
	journal={arXiv preprint arXiv:2003.03900},
	year={2020}
	}

```

# Dependencies
Requires:
* docker
* nvidia-docker2
* A recent Nvidia GPU *e.g.* GTX980 or better.
* Ubuntu

# Installation
The docker image essentially packages all dependencies in a safe environment.  The scripts we provide will externally mount our source code, and our models, into the docker environment.

## Docker and Nvidia-Docker

The following is all of the steps to build a docker image for `FormulaZero` from a fresh Ubuntu installation:

1) Install [Docker for Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/). Make sure to `sudo usermod -aG docker your-user` and then not run below docker scripts as `sudo`
2) Install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker). Make sure to use `nvidia-docker2` not `nvidia-docker1`.
```
sudo apt-get install -y nvidia-docker2
```
You can test that your nvidia-docker installation is working by running
```
nvidia-docker run --rm nvidia/cuda nvidia-smi
```
If you get errors about nvidia-modprobe not being installed, install it by running
```
sudo apt-get install nvidia-modprobe
```
and then restart your machine.

## Building the FormulaZero Docker Image
To run the demo, do the following. You will need to have docker installed and have an Nvidia GPU. Currently, this only works on Linux machines (in particular the visualization does not work on Mac).

* In your terminal, navigate to the Simulator folder.

* Run `./download_files.sh`

* If this script does not work for some reason, you can manually download the files here: https://drive.google.com/drive/folders/1cBRKoQ31lGhFYXkCiBOdl34JZvHwBQh0
And then place them manually in the following locations:
Unzip `flow_weights.zip` and then place in `/python`
Place `map1_speed.msgpack` and `map1_range.msgpack` in `/python`
Place `lut_inuse.npz` in `python/mpc`


* Run `./build_docker_ui.sh`
Note you may need to use sudo with this command depending on the way you installed docker.

* If you have docker 19.03 or later, install Nvidia Container Toolkit following the instructions here: (https://github.com/NVIDIA/nvidia-docker), and run `./docker_ui.sh`. Note that you may need to use sudo with this command depending on the way you installed docker.

* If you have an older version of docker and are using nvidia-docker 1.0, run `./docker_ui_nvidia_docker_1.0.sh`. Note that you may need to use sudo with this command depending on the way you installed docker.

* If you have an older version of docker and are using nvidia-docker 2.0, run `./docker_ui_nvidia_docker_2.0.sh`. Note that you may need to use sudo with this command depending on the way you installed docker.

The code will take up to 5 minutes to compile upon startup depending on your machine architecture.
