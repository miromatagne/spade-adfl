# Framework for managing a network of SPADE agents implementing Asynchronous Decentralized Federated Learning (ADFL)

This project is an application that allows to launch SPADE agents that implement the ADFL algorithm, and allows the user to setup, control, manage and monitor the execution. Each agent trains a local model on a given dataset, and the agents communicate between themselves in order to perform the asynchronous consensus and obtain asymptotically a very efficient model.

## Installation and execution

First of all, in order to execute this program, the device should be connected to a UPV network, or to a UPV VPN. Indeed, the XMPP server used in the solution is ```gtirouter.dsic.upv.es``` and is only accessible when connected to a UPV network. 

### Using the Docker repository

A Docker image of the application was created and uploaded to Docker Hub. It is publically accessible, and the application can be launched as follows :

```bash
docker pull miromatagne2103/tfm-spade-agents:latest
docker run -it --net=host miromatagne2103/tfm-spade-agents:latest --interface-port <port>
```

where ```port```corresponds to the port where the user interface of the launcher agent will be available (at the adress ```127.0.0.1:<port>/agent```).

### Using the source code

Install the dependencies :

```bash
pip install -m requirements.txt
```

Launch the program :

```bash
python3 launcher_main.py
```
