# 1d_gradCAM

This repository provides a Class to perform grad-CAM [[1](https://arxiv.org/abs/1610.02391)] in Python with Keras library. 

## How to use

1. Object creation:\
The library is build following OOP paradigm. The first step is to create the object: 
```python

gCam = GradCam(testset, model)

```
where: 
* *testset* is the testset on which compute gradCAM.
* *model* is the model to explain with gradCAM. It must be a keras model. 

2. HeatMap computation:
```python

heatmap = gCam.make_gradcam_heatmap(conv_layer_name)

```
where:
* *conv_layer_name* is the name of the convolutional layer on which applies the gradCAM. It is the same obtaine with the summary function of Keras. 

3. Interpolation:\
It is possible to automatically interpolate the data to obtain the same dimension as the input feature. At the the moment there are two possible interpolation strategies: 
* *cubic spline interpolation*
  ```python

  interpolate_vector = gCam.cubic_spline_interpolation(vector, v_size, input_size)
  
  ```
  where:
    * *vector* the vector to interpolate.
    * *v_size* size of the flatted heatmap. You have to multiply the two dimensions of the heatmap. 
    * *input_size* size of the input
