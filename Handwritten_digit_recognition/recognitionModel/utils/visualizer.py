import cv2
import numpy as np
import torch


def display_image(img, target_size=None):
    if target_size:
        ratio = target_size / max(img.shape[0], img.shape[1])
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=interp)
    cv2.imshow("", img)
    cv2.waitKey(0)


def demo_display_single_image(images, labels, predicted):
    image_to_display = images[0].cpu().numpy()
    image_to_display = image_to_display.transpose(1, 2, 0)

    # uint8 form
    image_to_display = (image_to_display * 255).astype(np.uint8)
    # resize 240x240
    image_to_display = cv2.resize(image_to_display, (240, 240))

    label = labels[0].item()
    predict = predicted[0].item()

    # add text
    cv2.putText(image_to_display, f'Label: {label}, Predicted: {predict}', (10, 20), 1, 1, (255, 255, 255), 1,
                cv2.LINE_AA)

    cv2.imshow("", image_to_display)
    cv2.waitKey(0)


def demo_display_specific_digit_combination(images, labels):
    assert images.shape[0] == labels.shape[0]

    # this is an example for retrieve digits image
    digit_image_dict = {}
    for i in range(0, labels.shape[0]):
        digit = int(labels[i].item())
        if digit not in digit_image_dict:
            digit_image_dict[digit] = images[i]

        if len(digit_image_dict.keys()) >= 10:
            break

    # Specify your student number as a list of digits
    student_number_digits = [2, 0, 2, 2, 0, 1, 2, 0, 8, 7]  # Replace with your actual student number

    # Retrieve the corresponding images from digit_image_dict
    digit_images = [digit_image_dict[digit] for digit in student_number_digits]

    # Concatenate images horizontally to form the target image
    target_img = torch.cat(digit_images, dim=2)  # Concatenate along the width (channels last format)

    target_img = target_img.numpy()
    target_img = target_img.transpose(1, 2, 0)
    display_image(target_img, target_size=240)
