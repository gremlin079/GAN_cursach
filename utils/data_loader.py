"""
Утилиты для загрузки и обработки данных отпечатков пальцев
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


class FingerprintDataset(Dataset):
    """
    Датасет для загрузки изображений отпечатков пальцев
    """
    
    def __init__(self, data_dir, img_size=64, transform=None, augment=False):
        """
        Args:
            data_dir: директория с изображениями
            img_size: размер изображения после ресайза
            transform: дополнительные трансформации
            augment: использовать ли аугментацию данных
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        
        # Получаем список всех изображений
        self.image_files = []
        if os.path.exists(data_dir):
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                self.image_files.extend([f for f in os.listdir(data_dir) 
                                        if f.lower().endswith(ext.replace('*', ''))])
        
        # Базовые трансформации
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
        ]
        
        if transform is None:
            # Стандартные трансформации для GAN
            transform_list = base_transforms + [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Нормализация к [-1, 1]
            ]
            
            if augment:
                # Агрессивная аугментация данных для малого датасета
                transform_list = base_transforms + [
                    transforms.RandomRotation(15),  # Увеличено с 5 до 15
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),  # Добавлено
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),  # Улучшено
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Добавлено
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),  # Добавлено
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ]
            
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform
    
    def __len__(self):
        if self.image_files:
            # Для малого датасета увеличиваем виртуальный размер через повторения
            # Это позволяет использовать больше аугментаций
            num_files = len(self.image_files)
            if num_files < 100:
                # Для малого датасета создаем виртуально больше образцов
                return max(1000, num_files * 50)  # Минимум 1000 образцов
        return len(self.image_files) if self.image_files else 1000  # Для синтетических данных
    
    def __getitem__(self, idx):
        if self.image_files:
            # Загрузка реального изображения с циклическим доступом
            # Это позволяет использовать аугментацию для создания вариаций
            img_path = os.path.join(self.data_dir, self.image_files[idx % len(self.image_files)])
            try:
                image = Image.open(img_path)
                if self.transform:
                    image = self.transform(image)
                return image
            except Exception as e:
                print(f"Ошибка загрузки изображения {img_path}: {e}")
                return self._generate_synthetic_sample()
        else:
            # Генерация синтетического отпечатка для демонстрации
            return self._generate_synthetic_sample()
    
    def _generate_synthetic_sample(self):
        """Генерация простого синтетического отпечатка для тестирования"""
        # Создание простого паттерна отпечатка
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Добавление линий (имитация папиллярных линий)
        for i in range(0, self.img_size, 3):
            cv2.line(img, (0, i), (self.img_size, i), 255, 1)
        
        # Добавление кривых линий
        center = self.img_size // 2
        for angle in range(0, 360, 10):
            x = int(center + 20 * np.cos(np.radians(angle)))
            y = int(center + 20 * np.sin(np.radians(angle)))
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                cv2.circle(img, (x, y), 5, 255, -1)
        
        # Добавление шума
        noise = np.random.normal(0, 10, (self.img_size, self.img_size))
        img = img + noise
        img = np.clip(img, 0, 255)
        
        # Нормализация
        img = img / 255.0
        img = (img - 0.5) / 0.5  # Нормализация к [-1, 1]
        
        # Преобразование в тензор
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        return img_tensor


def get_data_loader(data_dir, batch_size=64, img_size=64, shuffle=True, augment=False):
    """
    Создает DataLoader для обучения GAN
    
    Args:
        data_dir: путь к директории с изображениями
        batch_size: размер батча
        img_size: размер изображений
        shuffle: перемешивать ли данные
        augment: использовать ли аугментацию
    
    Returns:
        DataLoader объект
    """
    dataset = FingerprintDataset(data_dir, img_size=img_size, augment=augment)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Для Windows используем 0
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader


def denormalize_image(tensor):
    """
    Денормализация тензора изображения для визуализации
    
    Args:
        tensor: нормализованный тензор [-1, 1]
    
    Returns:
        numpy массив [0, 255]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    
    # Денормализация из [-1, 1] в [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Преобразование в numpy
    if len(tensor.shape) == 4:  # Batch
        tensor = tensor.permute(0, 2, 3, 1)
    elif len(tensor.shape) == 3:  # Single image
        tensor = tensor.permute(1, 2, 0)
    
    array = tensor.numpy()
    
    # Масштабирование к [0, 255]
    if array.shape[-1] == 1:  # Grayscale
        array = array.squeeze(-1)
    array = (array * 255).astype(np.uint8)
    
    return array

