"""
Скрипт для поиска лучшего чекпоинта для генерации
Анализирует историю обучения и рекомендует оптимальный чекпоинт
"""

import os
import glob
import torch
import numpy as np

def analyze_checkpoint(checkpoint_path):
    """Анализ одного чекпоинта"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Извлечение номера эпохи из имени файла
        filename = os.path.basename(checkpoint_path)
        epoch_num = int(filename.split('_')[-1].replace('.pth', ''))
        
        history = checkpoint.get('history', {'d_loss': [], 'g_loss': []})
        d_losses = history.get('d_loss', [])
        g_losses = history.get('g_loss', [])
        
        if len(g_losses) == 0:
            return None
        
        # Последние потери
        last_g_loss = g_losses[-1] if g_losses else float('inf')
        last_d_loss = d_losses[-1] if d_losses else float('inf')
        
        # Средние потери за последние 10 эпох (если доступно)
        if len(g_losses) >= 10:
            avg_g_loss = np.mean(g_losses[-10:])
            avg_d_loss = np.mean(d_losses[-10:]) if d_losses else last_d_loss
        else:
            avg_g_loss = np.mean(g_losses) if g_losses else last_g_loss
            avg_d_loss = np.mean(d_losses) if d_losses else last_d_loss
        
        # Критерий качества: баланс потерь и низкая потеря генератора
        balance_score = abs(avg_g_loss - avg_d_loss)  # Чем меньше разница, тем лучше
        
        # Комбинированная оценка: низкая потеря генератора + баланс
        total_score = avg_g_loss + balance_score * 0.3
        
        return {
            'epoch': epoch_num,
            'path': checkpoint_path,
            'last_g_loss': last_g_loss,
            'last_d_loss': last_d_loss,
            'avg_g_loss': avg_g_loss,
            'avg_d_loss': avg_d_loss,
            'balance_score': balance_score,
            'total_score': total_score
        }
    except Exception as e:
        print(f"Ошибка анализа {checkpoint_path}: {e}")
        return None

def find_best_checkpoints(checkpoints_dir='checkpoints', top_n=5):
    """Поиск лучших чекпоинтов"""
    
    print("=" * 70)
    print("ПОИСК ЛУЧШИХ ЧЕКПОИНТОВ ДЛЯ ГЕНЕРАЦИИ")
    print("=" * 70)
    print()
    
    # Поиск всех чекпоинтов
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, '*.pth'))
    
    if not checkpoint_files:
        print("❌ Чекпоинты не найдены!")
        return []
    
    print(f"Найдено чекпоинтов: {len(checkpoint_files)}")
    print("Анализ чекпоинтов...")
    print()
    
    # Анализ всех чекпоинтов
    results = []
    for cp_file in sorted(checkpoint_files):
        result = analyze_checkpoint(cp_file)
        if result:
            results.append(result)
    
    if not results:
        print("❌ Не удалось проанализировать чекпоинты")
        return []
    
    # Сортировка по качеству (лучшие первыми - минимальный total_score)
    results.sort(key=lambda x: x['total_score'])
    
    print("=" * 70)
    print(f"ТОП-{top_n} ЛУЧШИХ ЧЕКПОИНТОВ:")
    print("=" * 70)
    print()
    
    for i, result in enumerate(results[:top_n], 1):
        print(f"{i}. Эпоха {result['epoch']}")
        print(f"   Файл: {os.path.basename(result['path'])}")
        print(f"   Потеря генератора: {result['avg_g_loss']:.4f} (средняя)")
        print(f"   Потеря дискриминатора: {result['avg_d_loss']:.4f} (средняя)")
        print(f"   Баланс: {result['balance_score']:.4f} (меньше = лучше)")
        print()
    
    best = results[0]
    print("=" * 70)
    print(f"⭐ РЕКОМЕНДУЕМЫЙ ЧЕКПОИНТ: Эпоха {best['epoch']}")
    print("=" * 70)
    print(f"Файл: {os.path.basename(best['path'])}")
    print()
    print("Команда для генерации:")
    print(f"  py generate.py --checkpoint {best['path']} --num_images 16 --grid 4 4")
    print()
    
    return results

if __name__ == '__main__':
    find_best_checkpoints()
