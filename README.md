# RL in Depth (in Simulation)

Welcome to the Reinforcement Learning (RL) workshop! In this class, you will embark on an exciting three-day journey to deepen your understanding of RL by applying advanced concepts to a simulated robotics challenge. Our goal is to design and deploy an RL-based solution that commands a two-wheeled simulated robot, the **Duckiebot**, to navigate a real road environment successfully. This project will challenge your creativity and your problem-solving skills.

<div align="center">
  <img src="img/duckiebot.jpg" alt="Duckiebot" width="400"/>
</div>


Through this hands-on experience, you will gain practical insights into:
- Refining your understanding of Reinforcement Learning.
- Designing an RL agent suitable for controlling a two-wheeled robot.


## Organization

### Prerequisites (Docker)
First, you need to install docker (if you haven't already) by following the instructions in the [Docker installation guide](https://docs.docker.com/get-docker/).
Then follow the instructions in the [Docker initialization](#docker-initialization) section to build and run the docker image.

### Day 1: Discovery and Exploration
- Students will familiarize themselves with the simulator that models the Duckiebot and train an RL agent using the provided instructions in [duckiesim.md](duckiesim/duckiesim.md) file. This file explains how to set up the environment and guides students through the first steps of training.

### Days 2 and 3: Development and Competition
- Over the next two days, the focus will shift to the development and refinement of the RL-based solution. Students will test their agents, improve their models, and prepare for the final competition.

### Competition
At the end of Day 3, the competition will take place on a **circuit unknown to the students**. The objective is to **finish the circuit in minimal time**. The group with the best-performing agent will earn a grade of 20, while others will "strive harder next year" (humor).




## Course Schedule

Here is the planned schedule for the three days:

| **Day** | **Time**    | **Session**                           | **Key Goals**                                                   |
| ------- | ----------- | ------------------------------------- | --------------------------------------------------------------- |
| Day 1   | 08:30-10:00 | Intro                                 | Understand the objectives + Installation + Manual Control       |
|         | 10:00-10:30 | Munchausen                            | Understand Munchausen                                           |
|         | 10:30-11:45 | HP tuning + Reward shaping            | Prepare the environment for training                            |
|         | 13:30-13:45 | Policy evaluation                     | Make real robots work  Policy evaluation at the end of training |
|         | 13:45-15:00 | Real world setup                      | Control the real world robot                                    |
|         | 15:15-16:45 | Project presentation and Group set-up | The competition begins                                          |
| Day 2   | 08:30-10:00 | Brainstorming                         | Understand how to solve the problem                             |
|         | 10:15-11:45 | Implementation \& debug               | Make it work                                                    |
| Day 3   | 08:30-10:00 | Implementation \& debug               | Make it work                                                    |
|         | 10:15-11:45 | Implementation \& debug               | Make it work                                                    |
|         | 13:30-15:00 | Implementation \& debug               | Make it work                                                    |
|         | 15:15-16:00 | Preparing presentations               | Building the pitch                                              |
|         | 16:00-16:45 | Presentation                          | Evaluation                                                      |



## RL Robot Project Evaluation Grid

### Project Presentation and Implementation (18 points)

The project presentation will be a concise demonstration of your work and approach. While slides are not required, you should prepare a well-structured verbal explanation of your methodology, technical choices, and results. Be ready to clearly articulate your reasoning, explain the challenges you encountered and how you overcame them, and demonstrate your understanding of the RL algorithms used. What matters most is the quality of your argumentation and your ability to critically analyze your own work, not the visual support.

| Criterion                              | Insufficient                                  | Acceptable                                                | Good                                                      | Excellent                                                                                             |
| -------------------------------------- | --------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Clarity of approach**                | Approach is confusing and difficult to follow | Approach is generally understandable but lacks structure  | Approach is clear, logical and well-structured            | Approach is perfectly articulated with exemplary logical progression                                  |
| **Justification of technical choices** | Choices are not or poorly justified           | Choices are partially justified without in-depth analysis | Choices are well justified with relevant analysis         | Choices are perfectly justified with thorough critical analysis and considered alternatives           |
| **Understanding of RL algorithms**     | Superficial understanding of algorithms used  | Good understanding of the basics of algorithms used       | Strong mastery of algorithms with ability to explain      | Deep mastery with critical analysis and understanding of nuances                                      |
| **Experimentation and analysis**       | Few or no experiments                         | Some experiments without in-depth analysis                | Relevant experiments with results analysis                | Systematic experiments with comparative analyses and relevant conclusions                             |
| **Future directions**                  | Unable to identify meaningful future work     | Identifies basic next steps without clear justification   | Proposes well-reasoned future directions based on results | Presents comprehensive, strategic future directions with clear prioritization informed by conclusions |
| **Code documentation**                 | Code poorly documented or not at all          | Partially documented code                                 | Well documented and organized code                        | Exemplary, perfectly documented and structured code                                                   |

### Competition (2 points)

| Position  | Score    |
| --------- | -------- |
| 1st place | 2 points |
| 2nd place | 1 points |
| 3rd place | 0 points |


## AZURE MACHINE 


### SSH CONNECTION
Check your ssh IP at the following link: [Azure Portal](https://labs.azure.com/)
```bash
ssh -p PORT user1234@IP
```

### RDP CONNECTION (BUREAU A DISTANCE)

First install the RDP client on your machine by following the instructions in the [RDP Clients Installation Guide](#rdp-clients-installation-guide) section.


#### RDP Clients Installation Guide

##### Windows

**Microsoft Remote Desktop Client (built-in)**
```
Press Win + R, type mstsc and press Enter
```

##### Linux

**Remmina**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install remmina remmina-plugin-rdp

# Run from menu or terminal
remmina
```

##### macOS
1. Open App Store
2. Search for "Microsoft Remote Desktop"
3. Click "Get" to install
4. Launch from Launchpad


**Microsoft Remote Desktop**
```
1. Open App Store
2. Search for "Microsoft Remote Desktop"
3. Click "Get" to install
4. Launch from Launchpad
```


#### Download the RDP file
Download the RDP file from the following link: [Azure Portal](https://labs.azure.com/)




## Docker initialization

### Install nvidia-docker2
First, install nvidia-docker2 on your machine to use your gpu in the container:

```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Create docker image 

```bash
docker build -t indepthrl .
```

### Run docker image
Run the following command to make the script executable:
```bash
chmod +x launch_docker.sh
```
Run the following command to start the container:
```bash
./launch_docker.sh
```