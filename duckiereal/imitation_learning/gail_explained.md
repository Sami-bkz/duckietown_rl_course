# Generative Adversarial Imitation Learning (GAIL)

## Introduction
GAIL est un algorithme d'apprentissage par imitation qui combine les principes des GAN (Generative Adversarial Networks) avec l'apprentissage par renforcement. Il vise à reproduire le comportement d'un expert à partir d'un ensemble de démonstrations.

[Article original](https://arxiv.org/abs/1606.03476) | [Slides](https://efrosgans.eecs.berkeley.edu/CVPR18_slides/GAIL_by_Ermon.pdf)

## Principe général
L'algorithme fonctionne avec deux réseaux en opposition :
- Un **générateur** (la politique π) qui tente de reproduire le comportement expert
- Un **discriminateur** (D) qui tente de distinguer les trajectoires générées de celles de l'expert


### Objectif d'optimisation
```math
min_{π} max_{D} E_{π}[log(D(s,a))] + E_{πE}[log(1-D(s,a))] - λH(π)
