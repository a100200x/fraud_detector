import os
import sys
import pandas as pd
import time
import logging
from datetime import datetime

# Импортируем PollingObserver для стабильной работы в Docker на Windows
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# Корректно настраиваем пути для Docker (добавляем текущую директорию приложения)
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

from preprocessing import load_train_data, run_preproc
from scorer import make_pred

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            
            # Читаем исходный файл и сохраняем оригинальный индекс, 
            # чтобы предсказания не перемешались после merge/concat
            raw_df = pd.read_csv(file_path)
            file_index = raw_df.index
            
            input_df = raw_df.drop(columns=['name_1', 'name_2', 'street', 'post_code'], errors='ignore')
            
            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df)
            
            # Передаем оригинальный индекс в scorer для точного маппинга
            logger.info('Making prediction')
            submission = make_pred(processed_df, file_index)
            
            logger.info('Preparing submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)
            
        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return

class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service
        
    def on_created(self, event):
        # Проверяем, что это файл и он имеет расширение .csv
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    
    # Инициализируем PollingObserver для Windows/Docker окружения
    observer = PollingObserver()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
        
    observer.join()
