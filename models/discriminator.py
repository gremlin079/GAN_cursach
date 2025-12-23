"""
Дискриминатор для оценки подлинности изображений отпечатков пальцев
Определяет, является ли изображение реальным отпечатком или сгенерированным
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class Discriminator(nn.Module):
    """
    Дискриминатор отпечатков пальцев
    Классифицирует изображения как реальные (1) или фальшивые (0)
    """
    
    def __init__(self, nc=1, ndf=64):
        """
        Args:
            nc: количество каналов входного изображения (1 для grayscale)
            ndf: количество фильтров в дискриминаторе
        """
        super(Discriminator, self).__init__()
        
        # Разделяем на блоки для извлечения промежуточных признаков
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Выходной слой: вероятность подлинности
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Вероятность [0, 1]
        )
        
        self._initialize_weights()
    
    def forward(self, input, return_features=False):
        """
        Args:
            input: входное изображение
            return_features: если True, возвращает также промежуточные признаки для feature matching
        
        Returns:
            output: вероятность подлинности [batch]
            features: (опционально) список промежуточных признаков
        """
        feat1 = self.conv1(input)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        output = self.conv4(feat3)
        
        # Обработка выхода: output имеет форму [batch, 1, H, W]
        output = output.view(output.size(0), -1)  # [batch, H*W]
        output = output.mean(dim=1)  # [batch] - среднее по пространственным измерениям
        
        if return_features:
            # Возвращаем признаки из средних слоев для feature matching
            features = [feat2, feat3]
            return output, features
        return output
    
    def _initialize_weights(self):
        """Инициализация весов для стабильного обучения"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, 1.0, 0.02)
                init.constant_(m.bias, 0)


class AdvancedDiscriminator(nn.Module):
    """
    Улучшенный дискриминатор с дополнительными слоями и поддержкой feature matching
    """
    
    def __init__(self, nc=1, ndf=64):
        super(AdvancedDiscriminator, self).__init__()
        
        # Разделяем на блоки для извлечения промежуточных признаков
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Выходной слой
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def forward(self, input, return_features=False):
        """
        Args:
            input: входное изображение
            return_features: если True, возвращает также промежуточные признаки для feature matching
        
        Returns:
            output: вероятность подлинности [batch]
            features: (опционально) список промежуточных признаков
        """
        feat1 = self.conv1(input)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        output = self.conv5(feat4)
        
        # Обработка выхода: output имеет форму [batch, 1, H, W]
        output = output.view(output.size(0), -1)  # [batch, H*W]
        output = output.mean(dim=1)  # [batch] - среднее по пространственным измерениям
        
        if return_features:
            # Возвращаем признаки из средних слоев для feature matching
            features = [feat2, feat3, feat4]
            return output, features
        return output
    
    def _initialize_weights(self):
        """Инициализация весов для стабильного обучения"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, 1.0, 0.02)
                init.constant_(m.bias, 0)

