# FormulaZero
<b>Coming soon!</b>
<p align="center">
  <img src="assets/action.gif"/>
</p>


This is the reference implementation for our paper:

<em><b>FormulaZero: Distributionally Robust Online Adaptation via Offline Population Synthesis</b></em>
[PDF](https://arxiv.org/pdf/2003.03900.pdf)

[Aman Sinha*](http://amansinha.org), [Matthew O'Kelly*](http://www.mokelly.net/),[Hongrui Zheng*](), [Rahul Mangharam](), [John Duchi](http://stanford.edu/~jduchi/), [Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)

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
	author={Sinha, Aman and O'Kelly, Matthew and Zheng, Hongrui and Mangharam, Rahul
	journal={arXiv preprint arXiv:2003.03900},
	year={2020}
	}

```

# Dependencies
Requires:
* docker
* nvidia-docker2
* A recent Nvidia GPU *e.g.* GTX980 or better.

# Installation
The docker image essentially packages all dependencies in a safe environment.  The scripts we provide will externally mount our source code, and our models, into the docker environment.

Most source code for this project is in Python and so once the docker image is built we won't need any compiling.

## Docker and Nvidia-Docker

The following is all of the steps to build a docker image for `RareSim` from a fresh Ubuntu installation:

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
<b>Coming soon</b>
