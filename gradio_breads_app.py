import gradio as gr
import cv2
import numpy as np


def histograms_of_masked_img(path):
    masked_image = mask_img(path)
    cielab_image = convert_to_cielab(masked_image)
    return histogram(cielab_image)


def histogram(cielab_image):
    mask = np.all(cielab_image != [0, 0, 0], axis=-1)
    histograms = []
    for i in range(3):  # For L*, a*, and b* channels
        channel = cielab_image[:, :, i][mask]
        hist, _ = np.histogram(channel, bins=256, range=(0, 256))
        histograms.append(hist)
    return histograms


def convert_to_cielab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


def mask_img(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 40)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    hist1 = [h.astype(np.float32) for h in hist1]
    hist2 = [h.astype(np.float32) for h in hist2]
    similarities = [cv2.compareHist(h1, h2, method) for h1, h2 in zip(hist1, hist2)]
    return sum(similarities) / len(similarities)


def find_most_similar(histograms, reference_histograms, method=cv2.HISTCMP_CORREL):
    similarities = [compare_histograms(histograms, ref_hist, method) for ref_hist in reference_histograms]
    most_similar_index = similarities.index(max(similarities))
    return most_similar_index


reference_images = [
    'toasted_bread_dataset/pan2 medio tostado.jpg',
    'toasted_bread_dataset/pan2 sin tostar.jpg',
    'toasted_bread_dataset/pan2 tostado.jpg'
]

reference_histograms = [histograms_of_masked_img(path) for path in reference_images]

user_images_choices = [
    'toasted_bread_dataset/pan1 sin tostar.jpg',
    'toasted_bread_dataset/pan1 tostado.jpg',
    'toasted_bread_dataset/pan1 tostado bis.jpg',
    'toasted_bread_dataset/pan1 medio tostado.jpg',
    'toasted_bread_dataset/pan2 tostado bis.jpg',
]

def classify_bread(selected_image_path):
    # Your existing logic to process the selected image and find the most similar reference image
    histograms = histograms_of_masked_img(selected_image_path)
    most_similar_index = find_most_similar(histograms, reference_histograms)
    most_similar_image = cv2.imread(reference_images[most_similar_index])
    most_similar_image_rgb = cv2.cvtColor(most_similar_image, cv2.COLOR_BGR2RGB)
    return most_similar_image_rgb

def gallery_selection_handler(selection):
    """Handles selection from the gallery to pass the correct image path to the classification function."""
    # Extract the first file path from the selection
    if not selection:  # Check if the selection is empty
        print("No image selected.")
        return None

    # Since the selection could be a list of tuples, extract the first path
    image_path = selection[0][0]  # Assuming single selection, get the first item's first element
    print("Selected image path:", image_path)

    # Now you can use the image path directly
    return classify_bread(image_path)


# Define the user interface
iface = gr.Interface(
    fn=gallery_selection_handler,
    inputs=gr.Gallery(label="Choose a Bread Image", value=user_images_choices, type="filepath"),
    outputs=gr.Image(label="Most Similar Reference Image"),
    title="Bread Toastiness Classifier",
    description="Select an image of bread to see which reference image it's most similar to."
)

# Launch the app
iface.launch()
