import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import array
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import asyncio


def set_device():
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    return torch.device(dev)


def check_bottle(sender, value, user_data):
    classes = ['Beer Bottle', 'Plastic Bottle', 'Soda Bottle', 'Water Bottle', 'Wine Bottle']
    dpg.configure_item("valbut", enabled=False)
    dpg.set_value(user_data[0], "Working...")
    what = classify(user_data[1], user_data[2], user_data[3], classes)
    dpg.set_value(user_data[0], what)
    dpg.configure_item("valbut", enabled=True)


def classify(model, image_transforms, image, classes):
    model = model.eval()
    image = Image.fromarray(image, mode="RGB")
    image = image_transforms(image)
    image = image.unsqueeze(0)

    device = set_device()
    image = image.to(device)

    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return classes[predicted.item()]


async def load_model():
    model = torch.load('../Model/best_model.pth', map_location=set_device())
    print("Model loaded")
    return model


async def create_default_template(frame_width, frame_height):
    texture_data = []
    for i in range(0, int(frame_width * frame_height)):
        texture_data.append(1)
        texture_data.append(0)
        texture_data.append(1)
        texture_data.append(1)
    return list(array.array('f', texture_data))


async def main():
    model = await asyncio.create_task(load_model())
    mean_and_std = (torch.Tensor([0.4729, 0.4099, 0.3521]), torch.Tensor([0.1786, 0.1670, 0.1610]))
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean_and_std[0], mean_and_std[1])
    ])

    try:
        camera = cv2.VideoCapture(0)
    except:
        print("Can't reach camera")
        exit()

    frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps_info = camera.get(cv2.CAP_PROP_FPS)

    dpg.create_context()

    default_data = await asyncio.create_task(create_default_template(frame_width=frame_width, frame_height=frame_height))

    with dpg.texture_registry():
        dpg.add_raw_texture(height=frame_height, width=frame_width, default_value=default_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")

    with dpg.window(label="Camera", tag="cam", no_move=True, no_close=True):
        dpg.add_text(f"Camera info: {int(frame_width)}x{int(frame_height)} {fps_info}FPS")
        dpg.add_image("texture_tag")
        dpg.add_text("This is:")
        status = dpg.add_text("", tag="what_it_is")
        dpg.add_button(label="Validate bottle", tag="valbut", callback=check_bottle, user_data=[status, model, image_transforms, 0])

    dpg.create_viewport(title="Bootle segregation", height=int(frame_height+0.25*frame_height), width=int(frame_width), resizable=False)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("cam", True)
    while dpg.is_dearpygui_running():
        try:
            ret, frame = camera.read()
            if ret:
                data = np.flip(frame, 2)  # BGR -> RGB
                dpg.configure_item("valbut", user_data=[status, model, image_transforms, data])
                data = data.ravel()  # change frame data to 1D array
                data = np.asfarray(data, dtype='f')  # int to float
                data = np.true_divide(data, 255.0)
                dpg.set_value("texture_tag", data)
            else:
                dpg.set_value("texture_tag", default_data)
            dpg.render_dearpygui_frame()
        except:
            print("Camera lost")
            dpg.destroy_context()
            exit()

    camera.release()
    dpg.destroy_context()

asyncio.run(main())
