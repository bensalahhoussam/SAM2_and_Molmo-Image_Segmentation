# SAM2_and_Molmo_Image_Segmentation

This work is partially automatic image segmentation using natural language with SAM2 and Molmo, will build an app where we start from a text prompt, and use Molmo and SAM2 for generating segmentation maps of objects in an image

There are two deep learning models involved:

Molmo: The Molmo VLM will help extract the coordinates of objects using natural language. Here, we will use the MolmoE-1B-7B model.

SAM2: Then we will feed the image and the point coordinates as prompts to the SAM2 model for automated segmentation. 

![sam2-molmo-image-segmentation-pipeline](https://github.com/user-attachments/assets/8c0e1710-9f66-412b-b7be-e420860c85cf)


## Results 

outputs = get_output(image_path='input/image_2.jpg', prompt='Point where the people are.')

points x1="26.0" y1="67.5" x2="44.2" y2="40.5" alt="people">people</points

![téléchargement (1)](https://github.com/user-attachments/assets/ff21a667-a309-4c96-8bc5-2fe0b41a552d)

## Demo without Sam2

![ezgif-46695f6d71bc9c](https://github.com/user-attachments/assets/f027b24e-ca5f-412b-91d2-01bd65094f56)
