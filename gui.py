import os
import glob
import threading
import gradio as gr
import matplotlib.pyplot as plt
import torch

from setup_data import create_demo_data
from utils.data_loader import get_data_loader
from gan import FingerprintGAN
from generate import generate_images
from find_best_checkpoint import find_best_checkpoints


# -------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------

def list_images(folder, limit=16):
    images = sorted(glob.glob(os.path.join(folder, "*.png")))
    return images[:limit]


def count_images(folder):
    return len(glob.glob(os.path.join(folder, "*.png")))


def list_checkpoints():
    return sorted(glob.glob("checkpoints/*.pth"))


# -------------------------
# ДАННЫЕ
# -------------------------

def load_dataset_preview(data_dir):
    import os
    import glob

    if not os.path.exists(data_dir):
        return "Папка не найдена", []

    # Ищем изображения рекурсивно
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        images.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    images = sorted(images)

    info = f"Найдено изображений: {len(images)}"
    preview = images[:16]  # первые 16 для предпросмотра

    return info, preview



def create_demo():
    create_demo_data(20)
    return "Демо-данные созданы ✅"


# -------------------------
# ОБУЧЕНИЕ
# -------------------------

training_log = []
gan_instance = None


def train_model(
    data_dir,
    batch_size,
    epochs,
    img_size,
    nz,
    lr_g,
    lr_d,
    advanced,
    augment,
):
    global gan_instance
    training_log.clear()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        augment=augment,
    )

    gan_instance = FingerprintGAN(
        img_size=img_size,
        nz=nz,
        lr_g=lr_g,
        lr_d=lr_d,
        device=device,
        advanced=advanced,
    )

    gan_instance.train(
        dataloader=dataloader,
        num_epochs=epochs,
        save_dir="checkpoints",
        samples_dir="samples",
    )

    return "✅ Обучение завершено"


def train_async(*args):
    thread = threading.Thread(target=train_model, args=args)
    thread.start()
    return "▶ Обучение запущено (в фоне)"


# -------------------------
# ГЕНЕРАЦИЯ
# -------------------------
def list_samples():
    import glob
    import os

    samples_dir = "samples"
    if not os.path.exists(samples_dir):
        return []

    return sorted(glob.glob(os.path.join(samples_dir, "*.png")))


def generate_from_sample(sample_file):
    """
    Псевдо-генерация:
    возвращает выбранный файл из папки samples
    и показывает его как результат генерации.
    """

    import os

    if not sample_file:
        return []

    if not os.path.exists(sample_file):
        return []

    # Gradio Gallery принимает список файлов
    return [sample_file]






# -------------------------
# АНАЛИЗ
# -------------------------

def find_best():
    results = find_best_checkpoints("checkpoints")
    if not results:
        return "Чекпоинты не найдены"
    best = results[0]
    return f"⭐ Лучшая модель: эпоха {best['epoch']}"


def plot_history():
    import os
    import torch
    import matplotlib.pyplot as plt

    history_path = "training_history.pt"

    if not os.path.exists(history_path):
        return None

    history = torch.load(history_path)

    plt.figure(figsize=(6, 4))
    plt.plot(history.get("d_loss", []), label="Discriminator loss")
    plt.plot(history.get("g_loss", []), label="Generator loss")
    plt.legend()
    plt.grid(True)

    output_path = "training_history.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path



# -------------------------
# GUI
# -------------------------

with gr.Blocks(title="Fingerprint GAN Studio", theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🧠 Fingerprint GAN Studio")

    with gr.Tab("📁 Данные"):
        data_dir = gr.Textbox(value="data/fingerprints", label="Папка с данными")
        data_info = gr.Textbox(label="Информация")
        data_gallery = gr.Gallery(label="Примеры", columns=4)
        load_btn = gr.Button("Загрузить данные")
        demo_btn = gr.Button("Создать демо-данные")

        load_btn.click(load_dataset_preview, data_dir, [data_info, data_gallery])
        demo_btn.click(create_demo, None, data_info)

    with gr.Tab("🧠 Обучение"):
        batch_size = gr.Slider(8, 128, value=64, step=8, label="Batch size")
        epochs = gr.Slider(1, 200, value=50, step=1, label="Epochs")
        img_size = gr.Slider(32, 128, value=64, step=16, label="Image size")
        nz = gr.Slider(50, 200, value=100, step=10, label="Noise dim")
        lr_g = gr.Number(value=0.0002, label="LR Generator")
        lr_d = gr.Number(value=0.0002, label="LR Discriminator")
        advanced = gr.Checkbox(label="Advanced architecture")
        augment = gr.Checkbox(label="Data augmentation")

        train_btn = gr.Button("▶ Запустить обучение")
        train_status = gr.Textbox(label="Статус")

        train_btn.click(
            train_async,
            [
                data_dir,
                batch_size,
                epochs,
                img_size,
                nz,
                lr_g,
                lr_d,
                advanced,
                augment,
            ],
            train_status,
        )

    with gr.Tab("🖼️ Образцы"):
        samples_gallery = gr.Gallery(label="Samples", columns=4)
        refresh_samples = gr.Button("Обновить")

        refresh_samples.click(lambda: list_images("samples"), None, samples_gallery)

    with gr.Tab("🎨 Генерация"):
        sample_selector = gr.Dropdown(
            choices=list_samples(),
            label="Выберите файл из папки samples",
            interactive=True
        )

        refresh_samples_btn = gr.Button("🔄 Обновить список samples")
        gen_btn = gr.Button("Показать образец")

        gen_gallery = gr.Gallery(
            label="Результат",
            columns=1,
            height=400
        )

        refresh_samples_btn.click(
            fn=list_samples,
            inputs=None,
            outputs=sample_selector
        )

        gen_btn.click(
            fn=generate_from_sample,
            inputs=sample_selector,
            outputs=gen_gallery
        )

    with gr.Tab("📊 Анализ"):
        gr.Markdown("### График обучения GAN")

        plot_btn = gr.Button("📈 Показать график обучения")
        plot_img = gr.Image(label="Loss G / D")

        plot_btn.click(
            fn=plot_history,
            inputs=None,
            outputs=plot_img
        )

app.launch()
