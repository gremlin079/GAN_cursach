"""
Скрипт для генерации изображений отпечатков пальцев с помощью обученной модели
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from gan import FingerprintGAN
from utils.data_loader import denormalize_image


def generate_images(gan, num_images, output_dir, grid_size=None):
    """
    Генерация и сохранение изображений отпечатков пальцев
    
    Args:
        gan: обученная модель GAN
        num_images: количество изображений для генерации
        output_dir: директория для сохранения
        grid_size: размер сетки для визуализации (если None, то сохраняются отдельные файлы)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Генерация {num_images} изображений...")
    
    # Генерация изображений
    generated_images = gan.generate(num_images=num_images)
    
    # Денормализация
    images = denormalize_image(generated_images)
    
    if grid_size is not None:
        # Сохранение в виде сетки
        n_rows = grid_size[0]
        n_cols = grid_size[1]
        total_grid = n_rows * n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
        fig.suptitle('Сгенерированные отпечатки пальцев', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < min(total_grid, len(images)):
                ax.imshow(images[i], cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'generated_grid.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Сетка изображений сохранена: {output_path}")
    
    # Сохранение отдельных изображений
    print(f"Сохранение отдельных изображений...")
    for i, img in enumerate(images):
        output_path = os.path.join(output_dir, f'fingerprint_{i+1:04d}.png')
        plt.imsave(output_path, img, cmap='gray', format='png')
    
    print(f"Все изображения сохранены в: {output_dir}")


def interpolate_between_images(gan, num_steps=10, output_dir='interpolations'):
    """
    Интерполяция между двумя случайными точками в пространстве шума
    
    Args:
        gan: обученная модель GAN
        num_steps: количество шагов интерполяции
        output_dir: директория для сохранения
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Генерация интерполяции ({num_steps} шагов)...")
    
    # Две случайные точки в пространстве шума
    z1 = torch.randn(1, gan.nz, 1, 1, device=gan.device)
    z2 = torch.randn(1, gan.nz, 1, 1, device=gan.device)
    
    # Интерполяция
    alphas = np.linspace(0, 1, num_steps)
    interpolated_images = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        img = gan.generate(num_images=1, noise=z_interp)
        interpolated_images.append(img)
    
    # Объединение в одно изображение
    all_images = torch.cat(interpolated_images, dim=0)
    images = denormalize_image(all_images)
    
    # Сохранение как сетка
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    fig.suptitle('Интерполяция между отпечатками пальцев', fontsize=14)
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'α={alphas[i]:.2f}')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'interpolation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Интерполяция сохранена: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Генерация изображений отпечатков пальцев')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Путь к чекпоинту обученной модели')
    parser.add_argument('--num_images', type=int, default=16,
                       help='Количество изображений для генерации')
    parser.add_argument('--output_dir', type=str, default='generated',
                       help='Директория для сохранения сгенерированных изображений')
    parser.add_argument('--grid', type=int, nargs=2, default=None, metavar=('ROWS', 'COLS'),
                       help='Создать сетку изображений (например: --grid 4 4)')
    parser.add_argument('--interpolate', action='store_true',
                       help='Создать интерполяцию между двумя изображениями')
    parser.add_argument('--img_size', type=int, default=64,
                       help='Размер изображений (должен соответствовать обученной модели)')
    parser.add_argument('--nz', type=int, default=100,
                       help='Размер вектора шума (должен соответствовать обученной модели)')
    parser.add_argument('--advanced', action='store_true',
                       help='Использовать улучшенную архитектуру')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ГЕНЕРАЦИЯ ОТПЕЧАТКОВ ПАЛЬЦЕВ")
    print("=" * 50)
    
    # Проверка существования чекпоинта
    if not os.path.exists(args.checkpoint):
        print(f"Ошибка: Чекпоинт не найден: {args.checkpoint}")
        return
    
    # Инициализация устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка чекпоинта для получения параметров
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Использование параметров из чекпоинта, если они есть
    img_size = checkpoint.get('img_size', args.img_size)
    nz = checkpoint.get('nz', args.nz)
    
    print(f"\nЗагрузка модели из {args.checkpoint}...")
    print(f"Параметры модели:")
    print(f"  - Размер изображений: {img_size}")
    print(f"  - Размер вектора шума: {nz}")
    print(f"  - Расширенная архитектура: {args.advanced}")
    
    # Инициализация модели с правильными параметрами
    gan = FingerprintGAN(
        img_size=img_size,
        nz=nz,
        device=device,
        advanced=args.advanced
    )
    
    # Загрузка весов
    gan.load_checkpoint(args.checkpoint)
    
    # Генерация изображений
    if args.interpolate:
        interpolate_between_images(gan, output_dir=args.output_dir)
    else:
        generate_images(
            gan,
            num_images=args.num_images,
            output_dir=args.output_dir,
            grid_size=tuple(args.grid) if args.grid else None
        )
    
    print("\n" + "=" * 50)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 50)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Генерация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при генерации: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Для подробной диагностики используйте: py generate_with_error_handling.py")
        input("\nНажмите Enter для выхода...")

