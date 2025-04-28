from transformers import (AutoModelForCausalLM,AutoProcessor, GenerationConfig,BitsAndBytesConfig)
from PIL import Image
import torch
import requests
import cv2
import matplotlib.pyplot as plt
import re
import gradio as gr




quant_config = BitsAndBytesConfig(load_in_4bit=True)

processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',trust_remote_code=True,device_map='auto',torch_dtype='auto')
model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    offload_folder='offload',
    quantization_config=quant_config,
    torch_dtype='auto'
)


def get_coords(output_string):
    if 'points' in output_string:
        # Handle multiple coordinates
        matches = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', output_string)
        coordinates = [(int(float(x_val)), int(float(y_val))) for _, x_val, _, y_val in matches]
    else:
        # Handle single coordinate
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            coordinates = [(int(float(match.group(1))), int(float(match.group(2))))]

    return coordinates


def get_output(image_path=None, prompt="Describe this image."):
    # process the image and text
    if image_path:
        inputs = processor.process(
            images=[Image.open(image_path)],
            text=prompt
        )
    else:
        inputs = processor.process(
            images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
            text=prompt
        )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    print(generated_text)
    return generated_text





def draw_point_and_return(image_path=None, points=None):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    for point in points:
        p1 = int((point[0]/100)*w)
        p2 = int((point[1]/100)*h)
        image = cv2.circle(
            image,
            (p1, p2),
            radius=5,
            color=(0, 255, 0),
            thickness=5,
            lineType=cv2.LINE_AA
        )

    # Instead of showing, we return the image
    return image[..., ::-1]  # Convert BGR to RGB
def process_image(image, prompt):
    # Save the uploaded PIL image temporarily
    temp_path = 'temp_uploaded_image.jpg'
    image.save(temp_path)

    # Get model output
    output_text = get_output(image_path=temp_path, prompt=prompt)
    coords = get_coords(output_text)

    # Draw points
    image_with_points = draw_point_and_return(temp_path, coords)

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(image_with_points)
    ax.axis('off')
    plt.tight_layout()

    return fig, output_text


iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type='pil', label='Upload Image'),
        gr.Textbox(label='Prompt', placeholder='e.g., Point where the people are.')
    ],
    outputs=[
        gr.Plot(label='Result with Points', format='png'),
        gr.Textbox(label='Model Output')
    ],
    title='Image Point Detection',
    description='Upload an image and provide a prompt to locate and point objects in the image.',
)

iface.launch(share=True)