from rembg import remove, new_session
import cv2
import matplotlib.pyplot as plt
def RemoveBackground(image):
    model_name = "isnet-general-use"
    session = new_session(model_name)
    result = remove(image,session=session,)
    return result
def show_images(images, titles=None):
    num_images = len(images)

    if titles is None:
        titles = ['Image {}'.format(i+1) for i in range(num_images)]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.show()
input = 'dataset/dataset1/cane/OIP-0cYdzGqi1lvZQkk0Hy0GGAHaIu.jpeg'
image = cv2.imread(input)
show_images([image, RemoveBackground(image)], ['original', 'RemoveBackground'])
