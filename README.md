# Introduction to Diffusion Magnetic Resonance Imaging

#### Lecture and practical: Tractography, From Diffusion-weighted MRI to brain anatomical connectivity.

UE 3.11b – Advanced NeuroImaging Data Modeling and Analysis of [Master BioMedical Engineering](http://www.bme-paris.com), Track BioImaging (BIM)

Pauline Roca

Research scientist in the team: Imaging Biomarkers for brain development and disorders, Centre de Psychiatrie et Neurosciences, INSERM U894
Centre Hospitalier Sainte-Anne, Paris, FR


## Objectives

This practical is an introduction to diffusion Magnetic Resonance Imaging.

You will manipulate diffusion MRI data using python and:
  * familiarize yourself with common neuroimaging modules: nibabel, dipy and nilearn
  * better understand the MR signal in diffusion MRI (by plotting diffusion MR signal values in different tissues)
  * apply classic local models to model the diffusion signal (diffusion tensor model and spherical model) and compare them
  * do a whole brain tractography using dipy

 There are also `BONUS` exercices about :
  * Correction for susceptibility-induced spatial distortions
  * Brain segmentation and diffusion weighted imaging
  * Local modeling using FSL
  * Tractography using FSL

## Notebook:
The notebook of the practical can be found here: [bm_dwi_practical.ipynb](bme_dwi_practical.ipynb)

## Requirements

  * nibabel
  * dipy
  * nilearn
  * jupyter

## Installation
For configuration on Telecom ParisTech computers, you can follow the steps in [python_setup.sh](python_setup.sh).

## Data
We will use data from FSL courses on diffusion MRI, some dipy datasets and some Sainte-Anne Hospital data.

The link towards the different datasets are in the notebook.

