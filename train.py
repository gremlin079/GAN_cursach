"""
Скрипт для обучения GAN модели генерации отпечатков пальцев
"""

import argparse
import os
import matplotlib.pyplot as plt
from gan import FingerprintGAN
from utils.data_loader import get_data_loader, denormalize_image
import torch


def save_training_samples(gan, fixed_noise, epoch, save_dir='samples'):
    """Сохранение примеров сгенерированных изображений"""
    os.makedirs(save_dir, exist_ok=True)
    
    gan.generator.eval()
    with torch.no_grad():
        fake_images = gan.generate(64, fixed_noise)
    
    gan.generator.train()
    
    # Денормализация изображений
    images = denormalize_image(fake_images)
    
    # Сохранение в виде сетки
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle(f'Эпоха {epoch}', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}.png'), dpi=100, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """Визуализация истории обучения"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('История обучения')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Обучение GAN для генерации отпечатков пальцев')
    parser.add_argument('--data_dir', type=str, default='data/fingerprints',
                       help='Директория с изображениями отпечатков пальцев')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Размер батча')
    parser.add_argument('--img_size', type=int, default=64,
                       help='Размер изображений')
    parser.add_argument('--nz', type=int, default=100,
                       help='Размер вектора шума')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Количество эпох')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                       help='Learning rate для генератора')
    parser.add_argument('--lr_d', type=float, default=0.0002,
                       help='Learning rate для дискриминатора')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--samples_dir', type=str, default='samples',
                       help='Директория для сохранения образцов')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Интервал сохранения чекпоинтов (в эпохах)')
    parser.add_argument('--advanced', action='store_true',
                       help='Использовать улучшенную архитектуру')
    parser.add_argument('--augment', action='store_true',
                       help='Использовать аугментацию данных')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ОБУЧЕНИЕ GAN ДЛЯ ГЕНЕРАЦИИ ОТПЕЧАТКОВ ПАЛЬЦЕВ")
    print("=" * 50)
    print(f"Директория данных: {args.data_dir}")
    print(f"Размер батча: {args.batch_size}")
    print(f"Размер изображений: {args.img_size}")
    print(f"Количество эпох: {args.epochs}")
    print(f"Архитектура: {'Расширенная' if args.advanced else 'Базовая'}")
    print("=" * 50)
    
    # Создание директорий
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    dataloader = get_data_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        shuffle=True,
        augment=args.augment
    )
    print(f"Данные загружены. Количество батчей: {len(dataloader)}")
    
    # Инициализация GAN
    print("\nИнициализация GAN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = FingerprintGAN(
        img_size=args.img_size,
        nz=args.nz,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        device=device,
        advanced=args.advanced
    )
    
    # Фиксированный шум для отслеживания прогресса
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
    
    # Обучение
    print("\nНачало обучения...")
    gan.train(
        dataloader=dataloader,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        samples_dir=args.samples_dir,
        sample_interval=100  # Сохранение образцов каждые 100 итераций
    )
    
    # Сохранение примеров после каждой эпохи (упрощенная версия)
    print("\nСохранение финальных образцов...")
    save_training_samples(gan, fixed_noise, args.epochs, args.samples_dir)
    
    # Визуализация истории обучения
    print("\nСохранение графика истории обучения...")
    plot_training_history(gan.history)
    
    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print(f"Чекпоинты сохранены в: {args.save_dir}")
    print(f"Образцы сохранены в: {args.samples_dir}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
        print("Чекпоинты сохранены до момента прерывания.")
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Для подробной диагностики используйте: py train_with_error_handling.py")
        input("\nНажмите Enter для выхода...")

