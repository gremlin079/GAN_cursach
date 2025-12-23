"""
Генератор для создания изображений отпечатков пальцев
Использует глубокую нейронную сеть для генерации реалистичных отпечатков
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Генератор отпечатков пальцев на основе архитектуры DCGAN
    Принимает случайный шум и создает изображение отпечатка пальца
    """
    
    def __init__(self, nz=100, ngf=64, nc=1, img_size=64):
        """
        Args:
            nz: размер вектора входного шума
            ngf: количество фильтров в генераторе
            nc: количество каналов (1 для grayscale, 3 для RGB)
            img_size: размер генерируемого изображения
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.img_size = img_size
        
        # Основная архитектура: последовательность транс-сверточных слоев
        self.main = nn.Sequential(
            # Входной слой: шум -> проекция
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # Увеличение разрешения
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # Дальнейшее увеличение
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # Предпоследний слой
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # Выходной слой: генерация изображения
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Нормализация выхода к [-1, 1]
        )
        self._initialize_weights()
    
    def forward(self, input):
        """Генерация изображения из случайного шума"""
        return self.main(input)
    
    def _initialize_weights(self):
        """Инициализация весов для стабильного обучения"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, 1.0, 0.02)
                init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    """Residual блок для генератора"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        
        # Проекция для изменения размерности каналов
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class AdvancedGenerator(nn.Module):
    """
    Улучшенный генератор с residual connections для более реалистичных отпечатков
    Использует более глубокую архитектуру с residual блоками для лучшего обучения
    Поддерживает размеры 64x64 и 128x128
    """
    
    def __init__(self, nz=100, ngf=64, nc=1, img_size=64):
        super(AdvancedGenerator, self).__init__()
        self.nz = nz
        self.img_size = img_size
        
        # Определяем количество слоев апсемплинга в зависимости от размера изображения
        if img_size == 64:
            # Для 64x64: 4 -> 8 -> 16 -> 32 -> 64
            num_upsamples = 4
            initial_channels = ngf * 8
        else:
            # Для 128x128: 4 -> 8 -> 16 -> 32 -> 64 -> 128
            num_upsamples = 5
            initial_channels = ngf * 16
        
        # Начальная проекция шума
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(nz, initial_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(True)
        )
        
        # Блоки с residual connections и апсемплингом
        current_channels = initial_channels
        
        if img_size == 64:
            # Для 64x64
            self.block1 = ResidualBlock(current_channels, current_channels)
            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
            )
            current_channels = ngf * 4
            
            self.block2 = ResidualBlock(current_channels, current_channels)
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )
            current_channels = ngf * 2
            
            self.block3 = ResidualBlock(current_channels, current_channels)
            self.upsample3 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            )
            current_channels = ngf
            
            self.block4 = ResidualBlock(current_channels, current_channels)
            self.upsample4 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            )
            current_channels = ngf
        else:
            # Для 128x128
            self.block1 = ResidualBlock(current_channels, current_channels)
            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True)
            )
            current_channels = ngf * 8
            
            self.block2 = ResidualBlock(current_channels, current_channels)
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
            )
            current_channels = ngf * 4
            
            self.block3 = ResidualBlock(current_channels, current_channels)
            self.upsample3 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )
            current_channels = ngf * 2
            
            self.block4 = ResidualBlock(current_channels, current_channels)
            self.upsample4 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            )
            current_channels = ngf
            
            self.block5 = ResidualBlock(current_channels, current_channels)
            self.upsample5 = nn.Sequential(
                nn.ConvTranspose2d(current_channels, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            )
            current_channels = ngf
        
        # Финальные слои для детализации текстур
        # Используем несколько слоев для лучшей генерации текстур отпечатков
        self.final = nn.Sequential(
            nn.ConvTranspose2d(current_channels, current_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(current_channels, current_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(current_channels // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(current_channels // 2, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def forward(self, input):
        x = self.initial(input)
        
        x = self.block1(x)
        x = self.upsample1(x)
        
        x = self.block2(x)
        x = self.upsample2(x)
        
        x = self.block3(x)
        x = self.upsample3(x)
        
        x = self.block4(x)
        x = self.upsample4(x)
        
        if self.img_size == 128:
            x = self.block5(x)
            x = self.upsample5(x)
        
        x = self.final(x)
        return x
    
    def _initialize_weights(self):
        """Инициализация весов для стабильного обучения"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, 1.0, 0.02)
                init.constant_(m.bias, 0)

