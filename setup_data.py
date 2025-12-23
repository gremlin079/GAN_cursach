"""
Скрипт для создания структуры директорий и демонстрационных данных
"""

import os
import numpy as np
from PIL import Image
import cv2


def create_directories():
    """Создание необходимых директорий"""
    directories = [
        'data/fingerprints',
        'checkpoints',
        'samples',
        'generated',
        'interpolations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Создана директория: {directory}")


def generate_synthetic_sample(output_path, img_size=64):
    """
    Генерация простого синтетического отпечатка для тестирования
    
    Args:
        output_path: путь для сохранения
        img_size: размер изображения
    """
    # Создание базового изображения
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Центр изображения
    center = img_size // 2
    
    # Рисование концентрических окружностей (папиллярные линии)
    for radius in range(10, img_size // 2, 3):
        cv2.circle(img, (center, center), radius, 255, 1)
    
    # Добавление линий
    for i in range(0, img_size, 2):
        cv2.line(img, (0, i), (img_size, i), 255, 1)
    
    # Добавление случайных точек (детали)
    for _ in range(20):
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        cv2.circle(img, (x, y), 1, 255, -1)
    
    # Добавление шума
    noise = np.random.normal(0, 10, (img_size, img_size))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Применение размытия для более реалистичного вида
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Сохранение
    Image.fromarray(img).save(output_path)
    return img


def create_demo_data(num_samples=50):
    """Создание демонстрационных синтетических данных"""
    data_dir = 'data/fingerprints'
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\nСоздание {num_samples} демонстрационных изображений...")
    
    for i in range(num_samples):
        output_path = os.path.join(data_dir, f'fingerprint_{i+1:04d}.png')
        generate_synthetic_sample(output_path)
        
        if (i + 1) % 10 == 0:
            print(f"Создано {i + 1}/{num_samples} изображений...")
    
    print(f"\nВсе демонстрационные изображения сохранены в: {data_dir}")
    print("Примечание: Это простые синтетические данные для тестирования.")
    print("Для лучших результатов используйте реальные данные отпечатков пальцев.")


def main():
    print("=" * 50)
    print("НАСТРОЙКА ПРОЕКТА")
    print("=" * 50)
    
    print("\n1. Создание директорий...")
    create_directories()
    
    print("\n2. Создание демонстрационных данных...")
    # Проверяем, есть ли уже данные
    data_dir = 'data/fingerprints'
    if os.path.exists(data_dir):
        existing_files = [f for f in os.listdir(data_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if existing_files:
            print(f"✓ Найдено {len(existing_files)} существующих изображений")
            print("Демонстрационные данные не создаются.")
            return
    
    # Автоматически создаем небольшое количество для теста
    print("Создание 20 демонстрационных изображений для теста...")
    create_demo_data(20)
    
    print("\n" + "=" * 50)
    print("НАСТРОЙКА ЗАВЕРШЕНА!")
    print("=" * 50)
    print("\nСледующие шаги:")
    print("1. Если нужно, поместите реальные данные в 'data/fingerprints/'")
    print("2. Запустите обучение: python train.py")
    print("3. После обучения генерируйте изображения: python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pth")


if __name__ == '__main__':
    main()

