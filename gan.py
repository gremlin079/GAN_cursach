"""
Основной класс для обучения GAN на отпечатках пальцев
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os

from models.generator import Generator, AdvancedGenerator
from models.discriminator import Discriminator, AdvancedDiscriminator


class FingerprintGAN:
    """
    Класс для обучения и использования GAN для генерации отпечатков пальцев
    """
    
    def __init__(
        self,
        img_size=64,
        nz=100,
        ngf=64,
        ndf=64,
        nc=1,
        lr_g=0.0002,
        lr_d=0.0002,
        beta1=0.5,
        device=None,
        advanced=False
    ):
        """
        Args:
            img_size: размер изображений
            nz: размер вектора шума
            ngf: количество фильтров генератора
            ndf: количество фильтров дискриминатора
            nc: количество каналов (1 для grayscale)
            lr_g: learning rate для генератора
            lr_d: learning rate для дискриминатора
            beta1: параметр для Adam optimizer
            device: устройство для вычислений (cuda/cpu)
            advanced: использовать улучшенную архитектуру
        """
        self.img_size = img_size
        self.nz = nz
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Используется устройство: {self.device}")
        
        # Инициализация генератора
        if advanced:
            self.generator = AdvancedGenerator(nz=nz, ngf=ngf, nc=nc, img_size=img_size).to(self.device)
        else:
            self.generator = Generator(nz=nz, ngf=ngf, nc=nc, img_size=img_size).to(self.device)
        
        # Инициализация дискриминатора
        if advanced:
            self.discriminator = AdvancedDiscriminator(nc=nc, ndf=ndf).to(self.device)
        else:
            self.discriminator = Discriminator(nc=nc, ndf=ndf).to(self.device)
        
        # Функция потерь
        self.criterion = nn.BCELoss()
        
        # Оптимизаторы
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(beta1, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, 0.999)
        )
        
        # Метки для обучения
        self.real_label = 1.0
        self.fake_label = 0.0
        
        # Label smoothing для стабильности - уменьшено для лучшей генерации
        self.label_smoothing = 0.05  # Уменьшено с 0.1 для более четкого обучения
        
        # Коэффициент для балансировки обучения
        self.g_train_ratio = 1.0  # Сколько раз обучать генератор относительно дискриминатора
        
        # Коэффициент для feature matching loss
        self.feature_matching_weight = 10.0
        
        # История обучения
        self.history = {
            'd_loss': [],
            'g_loss': []
        }
    
    def train_discriminator(self, real_images):
        """
        Обучение дискриминатора на реальных и сгенерированных изображениях
        
        Args:
            real_images: батч реальных изображений
        
        Returns:
            loss_d: потери дискриминатора
        """
        self.discriminator.zero_grad()
        
        # Обработка реальных изображений
        batch_size = real_images.size(0)
        # Label smoothing для реальных изображений: 1.0 -> 0.9
        label = torch.full((batch_size,), self.real_label - self.label_smoothing, dtype=torch.float, device=self.device)
        
        output = self.discriminator(real_images)
        err_d_real = self.criterion(output, label)
        err_d_real.backward()
        d_x = output.mean().item()
        
        # Генерация фальшивых изображений
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        # Label smoothing для фальшивых изображений: 0.0 -> 0.1
        label.fill_(self.fake_label + self.label_smoothing)
        
        output = self.discriminator(fake_images.detach())
        err_d_fake = self.criterion(output, label)
        err_d_fake.backward()
        d_g_z1 = output.mean().item()
        
        # Обновление весов дискриминатора
        err_d = err_d_real + err_d_fake
        self.optimizer_d.step()
        
        return err_d.item(), d_x, d_g_z1
    
    def train_generator(self, batch_size, real_images=None, use_feature_matching=True):
        """
        Обучение генератора с улучшенной стратегией и feature matching
        
        Args:
            batch_size: размер батча
            real_images: реальные изображения для feature matching (опционально)
            use_feature_matching: использовать ли feature matching loss
        
        Returns:
            loss_g: потери генератора
        """
        self.generator.zero_grad()
        
        label = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        
        # Основная adversarial loss
        output = self.discriminator(fake_images)
        err_g_adv = self.criterion(output, label)
        
        # Feature matching loss для лучшего качества
        err_g = err_g_adv
        if use_feature_matching and real_images is not None:
            # Получаем промежуточные признаки из дискриминатора для реальных и фальшивых изображений
            _, real_features = self.discriminator(real_images, return_features=True)
            _, fake_features = self.discriminator(fake_images, return_features=True)
            
            # Вычисляем L2 расстояние между признаками
            feature_matching_loss = 0.0
            for real_feat, fake_feat in zip(real_features, fake_features):
                # Усредняем по пространственным измерениям для каждого признака
                real_feat = real_feat.view(real_feat.size(0), -1).mean(dim=1)
                fake_feat = fake_feat.view(fake_feat.size(0), -1).mean(dim=1)
                feature_matching_loss += torch.nn.functional.mse_loss(fake_feat, real_feat)
            
            # Комбинируем adversarial loss и feature matching loss
            # Feature matching помогает генератору создавать более реалистичные текстуры
            err_g = err_g_adv + self.feature_matching_weight * feature_matching_loss
        
        err_g.backward()
        d_g_z2 = output.mean().item()
        
        self.optimizer_g.step()
        
        return err_g.item(), d_g_z2
    
    def train(self, dataloader, num_epochs=50, save_dir='checkpoints', save_interval=5, samples_dir='samples', sample_interval=100):
        """
        Основной цикл обучения
        
        Args:
            dataloader: DataLoader с данными
            num_epochs: количество эпох
            save_dir: директория для сохранения чекпоинтов
            save_interval: интервал сохранения (в эпохах)
            samples_dir: директория для сохранения образцов
            sample_interval: интервал сохранения образцов (в итерациях)
        """
        # Создание директорий
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        
        # Фиксированный шум для отслеживания прогресса
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
        
        print("Начало обучения...")
        print(f"Используется label smoothing: {self.label_smoothing}")
        
        iteration = 0
        
        for epoch in range(num_epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{num_epochs}")
            
            for i, real_images in enumerate(progress_bar):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # Обучение дискриминатора
                d_loss, d_x, d_g_z1 = self.train_discriminator(real_images)
                
                # Улучшенная балансировка обучения генератора
                # Обучаем генератор в зависимости от баланса дискриминатора
                num_g_steps = 1
                
                # Если дискриминатор слишком уверен в фальшивых (d_g_z1 < 0.1), генератор слабый
                if d_g_z1 < 0.1:
                    num_g_steps = 3  # Обучаем генератор 3 раза
                elif d_g_z1 < 0.3:
                    num_g_steps = 2  # Обучаем генератор 2 раза
                elif d_x > 0.95:
                    # Дискриминатор слишком уверен в реальных - тоже обучаем генератор больше
                    num_g_steps = 2
                
                g_loss, d_g_z2 = self.train_generator(batch_size, real_images, use_feature_matching=True)
                for _ in range(num_g_steps - 1):
                    g_loss, d_g_z2 = self.train_generator(batch_size, real_images, use_feature_matching=True)
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                num_batches += 1
                iteration += 1
                
                # Обновление прогресс-бара
                progress_bar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}',
                    'D(x)': f'{d_x:.4f}',
                    'D(G(z))': f'{d_g_z2:.4f}'
                })
                
                # Сохранение образцов для мониторинга прогресса
                if iteration % sample_interval == 0:
                    self._save_samples(fixed_noise, samples_dir, epoch, iteration)
                
            # Сохранение истории
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            self.history['d_loss'].append(avg_d_loss)
            self.history['g_loss'].append(avg_g_loss)
            
            print(f"\nЭпоха {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
            
            # Сохранение чекпоинта
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        torch.save(self.history, "training_history.pt")
        print("Обучение завершено!")
    
    def _save_samples(self, fixed_noise, samples_dir, epoch, iteration):
        """Сохранение образцов для мониторинга прогресса"""
        try:
            from utils.data_loader import denormalize_image
            import matplotlib.pyplot as plt
            
            self.generator.eval()
            with torch.no_grad():
                fake_images = self.generator(fixed_noise[:16])  # Первые 16 для сетки 4x4
            
            self.generator.train()
            
            # Денормализация
            images = denormalize_image(fake_images)
            
            # Создание сетки 4x4
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            fig.suptitle(f'Эпоха {epoch+1}, Итерация {iteration}', fontsize=12)
            
            for idx, ax in enumerate(axes.flat):
                if idx < len(images):
                    ax.imshow(images[idx], cmap='gray')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'epoch_{epoch+1}_iter_{iteration}.png'), dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Ошибка при сохранении образцов: {e}")
    
    def generate(self, num_images=1, noise=None):
        """
        Генерация изображений отпечатков пальцев
        
        Args:
            num_images: количество изображений для генерации
            noise: опциональный шум (если None, генерируется случайный)
        
        Returns:
            generated_images: тензор с сгенерированными изображениями
        """
        self.generator.eval()
        
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(num_images, self.nz, 1, 1, device=self.device)
            else:
                noise = noise.to(self.device)
            
            fake_images = self.generator(noise)
        
        self.generator.train()
        return fake_images
    
    def save_checkpoint(self, filepath):
        """Сохранение модели"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history,
            'img_size': self.img_size,
            'nz': self.nz,
            'advanced': isinstance(self.generator, AdvancedGenerator)
        }
        torch.save(checkpoint, filepath)
        print(f"Модель сохранена: {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        # Восстанавливаем параметры модели
        self.img_size = checkpoint.get('img_size', self.img_size)
        self.nz = checkpoint.get('nz', self.nz)

        # Загружаем веса
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        self.history = checkpoint.get('history', {'d_loss': [], 'g_loss': []})

        print(f"Модель загружена: {filepath}")

    

