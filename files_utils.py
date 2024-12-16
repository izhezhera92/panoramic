import os
import shutil
from typing import List, Union
from fnmatch import fnmatch


def split_list(data = None, chunk_size = 5, overlap_size = 2):
    result = None
    if data is not None:
        # Разбивает список на подсписки заданного размера
        modern_list = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        result = []
        for i in range(len(modern_list) - 1):
            combined = modern_list[i] + modern_list[i + 1][:overlap_size]
            result.append(combined)

    return result


def create_image_list(lst, image_path, resize_coef):
        img_list = []
        for i in range(len(lst)):
            new_image = cv2.imread(image_path + lst[i])
            new_image = cv2.resize(new_image, (0, 0), fx = resize_coef, fy = resize_coef)
            img_list.append(new_image)
        return img_list

def modificat_list(data = [], voyager_type = "center"):
    if voyager_type == "center":
        mid_index = len(data) // 2
        result = data[:mid_index][::-1] + data[mid_index:]
        return result
    return []


def move_files_from_a_to_b(path_a: str = "", path_b: str = ""):
    os.makedirs(path_b, exist_ok=True)

    for file_name in os.listdir(path_a):
        source_path = os.path.join(path_a, file_name)
        destination_path = os.path.join(path_b, file_name)

        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)
            return 0
    return 1


def get_files_list(
    path: str = "", 
    mode: str = "fast_mapping", 
    step: int = 1, 
    reverse: bool = False, 
    part_or_full: bool = True, 
    start: int = 0, 
    last: int = 0, 
    mask: Union[str, List[str]] = ["*.tif", "*.bmp", "*.JPG", "*.jpeg"]
) -> List[str]:
    """
    Получить список файлов из папки с фильтрацией по нескольким маскам.

    Args:
        path (str): Путь к папке.
        mode (str): Режим работы. Варианты: "fast_mapping", "video".
        step (int): Шаг выборки файлов.
        reverse (bool): Обратная сортировка.
        part_or_full (bool): True для возврата части списка, False для полного списка.
        start (int): Начало выборки (индекс).
        last (int): Конец выборки (индекс).
        mask (Union[str, List[str]]): Маска или список масок для фильтрации файлов (например, "*.jpg" или ["*.jpg", "*.png"]).

    Returns:
        List[str]: Список файлов, соответствующих условиям.
    """
    if isinstance(mask, str):
        mask = [mask]  # Преобразуем строку в список из одного элемента

    # Получаем список файлов в папке
    file_list = os.listdir(path)

    # Фильтруем файлы по всем маскам
    filtered_files = [f for f in file_list if any(fnmatch(f, m) for m in mask)]

    # Если выбран режим "video", выбираем файлы с шагом
    if mode == "video":
        filtered_files = filtered_files[::step]

    # Сортируем файлы
    result = sorted(filtered_files, reverse=reverse)

    # Если требуется часть списка
    if part_or_full is True:
        if last == 0:
            last = len(result)
        sublist = result[start:last + 1]
        return sublist

    return result



def buffer_folder_preparation(temp_path_a, temp_path_b):
    if not os.path.exists(temp_path_a):
        os.makedirs(temp_path_a)

    if not os.path.exists(temp_path_b):
        os.makedirs(temp_path_b)

    if os.path.exists(temp_path_a):
        shutil.rmtree(temp_path_a)  
        os.makedirs(temp_path_a)
        #logging.info(f"First cleaning folder a")

    if os.path.exists(temp_path_b):
        shutil.rmtree(temp_path_b)  
        os.makedirs(temp_path_b)
        #logging.info(f"First cleaning folder b")

    return 0