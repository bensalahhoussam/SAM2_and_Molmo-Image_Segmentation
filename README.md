# SAM2_and_Molmo_Image_Segmentation

This work is partially automatic image segmentation using natural language with SAM2 and Molmo, will build an app where we start from a text prompt, and use Molmo and SAM2 for generating segmentation maps of objects in an image

There are two deep learning models involved:

Molmo: The Molmo VLM will help extract the coordinates of objects using natural language. Here, we will use the MolmoE-1B-7B model.

SAM2: Then we will feed the image and the point coordinates as prompts to the SAM2 model for automated segmentation. 

![sam2-molmo-image-segmentation-pipeline](https://github.com/user-attachments/assets/8c0e1710-9f66-412b-b7be-e420860c85cf)


## Results 

![ezgif-46695f6d71bc9c](https://github.com/user-attachments/assets/f027b24e-ca5f-412b-91d2-01bd65094f56)
