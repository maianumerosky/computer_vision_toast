import gradio as gr
import cv2
import numpy as np


def histograms_of_masked_image(path):
    masked_image = mask_image(path)
    cielab_image = convert_to_cielab(masked_image)
    return histograms(cielab_image)


def histograms(cielab_image):
    mask = np.all(cielab_image != [0, 0, 0], axis=-1)
    histograms_for_each_channel = []
    for i in range(3):  # For L*, a*, and b* channels
        channel = cielab_image[:, :, i][mask]
        hist, _ = np.histogram(channel, bins=256, range=(0, 256))
        histograms_for_each_channel.append(hist)
    return histograms_for_each_channel


def convert_to_cielab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


def mask_image(image_path):
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
    'toasted_bread_dataset/toasted.jpg',
    'toasted_bread_dataset/not_toasted.jpg',
    'toasted_bread_dataset/very_toasted.jpg'
]

reference_histograms = [histograms_of_masked_image(path) for path in reference_images]

user_images_choices = [
    'toasted_bread_dataset/pan1_sin_tostar.jpg',
    'toasted_bread_dataset/pan1_tostado.jpg',
    'toasted_bread_dataset/pan1_tostado_bis.jpg',
    'toasted_bread_dataset/pan1_medio_tostado.jpg',
    'toasted_bread_dataset/pan2_tostado_bis.jpg',
]


def classify_bread(selected_image_index):
    histograms = histograms_of_masked_image(user_images_choices[selected_image_index])
    most_similar_index = find_most_similar(histograms, reference_histograms)
    most_similar_image_path = reference_images[most_similar_index]
    most_similar_image = cv2.imread(most_similar_image_path)
    most_similar_image_rgb = cv2.cvtColor(most_similar_image, cv2.COLOR_BGR2RGB)
    cv2.putText(most_similar_image_rgb, f"Most similar: {clean_name(most_similar_image_path)}", (70, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return most_similar_image_rgb


def clean_name(image_path):
    removed_folder = image_path.split("/")[-1]
    removed_extension = removed_folder.split(".")[0]
    removed_underscore = removed_extension.replace("_", " ")
    return removed_underscore.capitalize()


with gr.Blocks(title='Bread toastedness classifier') as demo:
    gr.Markdown(
        """
        # Classify the toastedness of bread!
        Choose an image below.
        """)
    imgs = gr.State()
    gallery = gr.Gallery(allow_preview=False)


    def deselect_images():
        return gr.Gallery(selected_index=None)


    def generate_images():
        images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in user_images_choices]
        return images, images


    demo.load(generate_images, None, [gallery, imgs])

    with gr.Row():
        selected = gr.Number(show_label=False)
        classify_btn = gr.Button("Classify selected")
    deselect_button = gr.Button("Deselect")

    deselect_button.click(deselect_images, None, gallery)


    def get_select_index(evt: gr.SelectData):
        return evt.index


    gallery.select(get_select_index, None, selected)


    def classify_img(imgs, index):
        index = int(index)
        imgs[index] = classify_bread(index)
        return imgs, imgs


    classify_btn.click(classify_img, [imgs, selected], [imgs, gallery])

if __name__ == "__main__":
    demo.launch()
