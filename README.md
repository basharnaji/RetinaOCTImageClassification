
![Eye title graphic](/images/eye_title_graphic.JPG)

# Deep Learning-based OCT image classification


**This work is a group industrial project, considered as final assignment for FourthBrain Machine Learning Engineer program.**

[FourthBrain Machine Learning Engineer program website](https://www.fourthbrain.ai/machine-learning-engineer)  


## 🙎 Contributing members 

|Name |GitHub profile name |GitHub profile link|
|-----|-------|--|
|Christina Morgenstern|morgen01|[Link](https://github.com/morgen01)|
|Bashar Naji          |basharnaji|[Link](https://github.com/basharnaji)|
|Pawel Dymek   |pdymek|[Link](https://github.com/pdymek)|

## 📝 Project description

### Background

Retinal Optical Coherence Tomography (OCT) is a non-invasive imaging test. OCT uses light waves to take cross-section pictures of your retina. With OCT, your ophthalmologist can see each of the retina's distinctive layers. This allows your ophthalmologist to map and measure their thickness.  
It is estimated that 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.

### Problem description

In order to speed up this process we can utilize Machine Learning models to identify scans of patients that might have a disease that can allow ophthalmologists to focus on those patients first.  One of the challenges of having accurate Machine Learning models is the scarcity of well annotated data. In this project, we tackle this problem by utilizing the sinGAN model to generate realistic synthetic data that can be used to increase our training data and improve our prediction results.

Challenges in OCT image analysis:
- Availability of data  
- Labeling requires effort and expertise
- Class imbalance
- Speed of analysis is important
- Algorithms still produce erroneous results and require expert intervention

### 📖 Related work

- [SinGAN description](https://arxiv.org/abs/1905.00116)
- [sSinGAN with sinGAN-seg publication](https://arxiv.org/abs/2107.00471)
- [OpticNet-71 reference](https://github.com/SharifAmit/OpticNet-71)
- [simCLRv2 reference](https://github.com/anoopsanka/retinal_oct)


## ⚙️ Used technologies

- Python
- TensorFlow
- Flask
- AWS
- GoogleColab
- Kaggle
- GitHub


<p align="left">
<img src="https://camo.githubusercontent.com/aa96ee3a3352c9c3c2161d3e95698d0885a277ab85d617fe77912627d37a3959/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f707974686f6e2e737667" height="40" style="vertical-align:top; margin:4px">
<img src="https://camo.githubusercontent.com/981d48e57e23a4907cebc4eb481799b5882595ea978261f22a3e131dcd6ebee6/68747470733a2f2f70616e6461732e7079646174612e6f72672f7374617469632f696d672f70616e6461732e737667" height="40" style="vertical-align:top; margin:4px">
<img src="https://camo.githubusercontent.com/c04e16c05de80dadbdc990884672fc941fdcbbfbb02b31dd48c248d010861426/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f736f6369616c2e706e67" height="40" style="vertical-align:top; margin:4px">  
<img src="https://camo.githubusercontent.com/d626e9d547431bd83945c901088f0ff8b48bbf45ff074dd46272fdec5818c9c5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f636f6c61626f7261746f72792e737667" height="40" style="vertical-align:top; margin:4px">
 <img src="https://avatars.githubusercontent.com/u/2232217?s=200&v=4" height="40" style="vertical-align:top; margin:4px">  
<img src="https://flask.palletsprojects.com/en/1.1.x/_static/flask-icon.png" height="40" style="vertical-align:top; margin:4px">    
<img src="https://camo.githubusercontent.com/96313f84e4c257e753560f701e77c29697410d36bbd327294980f90451fcb1bc/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6b6167676c652e737667" height="40" style="vertical-align:top; margin:4px">   
<img src="https://camo.githubusercontent.com/72e5df59529a42423d671ba4c02bfb327d917517bfff18595c5e5dc17a5abece/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f68746d6c352e737667  " height="40" style="vertical-align:top; margin:4px">  
<img src="https://camo.githubusercontent.com/b788527f604d8e727fcc90d721984125bced85c8a1c9f8da69c6c4a3e51df3c5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f637373332e737667" height="40" style="vertical-align:top; margin:4px"> 
<img src="https://camo.githubusercontent.com/6ae487ec56908a6fea7e7f58bb04f09786fc25954ac2a41dceb69b6a2c61b5c5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6d61726b646f776e2e737667" height="40" style="vertical-align:top; margin:4px"> 
 <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" height="40" style="vertical-align:top; margin:4px">  
</p>


## 📊 Initial data

- [Kaggle OCT dataset](https://www.kaggle.com/paultimothymooney/kermany2018),
- [OCT Canada dataaset](https://dataverse.scholarsportal.info/dataverse/OCTID).

### Basic data summary

Dataset is imbalanced among different classes:

<details open><summary><b>🔍 Image class distribution</b></summary>
  
![class distribution](/images/image_class_distribution.JPG)

 </details>
 
 
 
## 🗺️ System design


<details open><summary><b>🔍 Project steps</b></summary>

![Project steps](/images/project_steps.JPG)

 </details>  
  
The system approach is summarized by the following:
- The technician/doctor will upload the OCT scan into our system
- Any Personally Identifiable Information (PII) is removed from the image in order to protect the privacy of the patient and adhere to local & medical laws.  A unique ID will be generated for the transaction
- The image is converted into the desired JPG format that the model requires
- The model will process the image and make a prediction on the patient’s status
- The result of the model and the image will be stored for future reference
- The result will be passed back to the ophthalmologist 
- The ophthalmologist will confirm the diagnosis or correct it and that result is fed back to our model storage
- The model should get retrained any time its performance (accuracy) drops below 92% and every time we acquire 500 new annotated images:
  - singGAN will be used to generate new synthetic data 
  - use Xception model to retrain on the new “training” data.
  - Once we reach 250,000 real annotated images we should consider turning off the sinGAN portion as we probably have acquired a significant amount of labelled images.

<details><summary><b>🔍 System design schema</b></summary>

![System desing](/images/system_design.JPG)
  
</details>
 
 
 
## 🔬 Explanation of Outcomes 


### Xception model

The Xception model was used for training on different randomly picked fractions of data (100%, 75%, 50%, 10%). For subsequent tuning of the model we decided to do it on this trained with 10% of data, supposing that it’s good base accuracy on a relatively small portion of data, which is an important factor for practical usage of the model. Other tasks will be related to maximizing the improvement of the 10% trained model.

|Training Data Fraction|Test Accuracy |Test Loss (cross entropy) |
|- |-|-|
|100%|98.97%|0.0463|
|75%|99.48%|0.0293|
|50%|98.14%|0.0587|
|10%|92.15%|0.2108|


### Experimenting with trainable layers

Different experimental approaches in setting trainable layers for all or selected ones.

|Training data fraction|Trainable layers|Test accuracy|Test loss|Training time|
|-|-|-|-|-|
|10%|all|92.15%|0.2108|2min 32s ± 1.95 s per loop (mean ± std. dev. of 7 runs, 1 loop each)|
|10%|Block 14 (last 6 layers)|85.74%|0.4135|2min 3s ± 809 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)|
|10%|Block 13 and 14 (last 16 layers)|98.14%|0.0662|2min 1s ± 875 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)|

### Final results after improvments

|Training data fraction / trainable layers|Test accuracy|Test Loss (Cross entropy)|
|-|-|-|
|10%|92.15%|0.2108|
|10% + SinGAN|98.86%|0.0328|

### Generating SinGAN images

Multiple new images are generated based on sample images.

<details><summary><b>🔍 New SinGAN image examples</b></summary>
  
![Normal class examples](/images/singan_generated_normal.JPG)
![Normal class examples](/images/singan_generated_cnv.JPG)
 
</details>

## 🚀 Deployment

The application is deployed on following address:  
http://octclf.eba-yntjrjbn.us-west-2.elasticbeanstalk.com/

![](/images/OCT-sample-test.gif)
