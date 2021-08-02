# Retina_OCT_Image_Classification

This is for the OCT2017 image set.
The approach is to start off with a DenseNet121 model and then extracting the features to train a SimCLRv2 model.
The results from the SimCLRv2 model will then be fed back to the DenseNet model and the improvements will be measured


#Colab Notebooks
01-OCTData_Dense --> Initial model using 1% of the training data and DenseNet121 Model with imagenet wweights.  We added a few convolutional layers on top with some skip connections to improve performance slightly
