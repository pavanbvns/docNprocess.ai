import os
import logging
import threading
import hashlib
import time
from datetime import datetime
from fastapi import HTTPException, UploadFile
from uuid import uuid4


class Utils:
    @staticmethod
    def authenticate_user(username, password, config):
        """Authenticates users based on the provided username and password."""
        valid_username = config["auth"]["username"]
        valid_password = config["auth"]["password"]
        if username != valid_username or password != valid_password:
            raise HTTPException(status_code=401, detail="Invalid username or password")

    @staticmethod
    def ensure_directory_exists(directory_path):
        """Ensures that the specified directory exists, creating it if necessary."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def calculate_file_hash(file_path):
        """Generates a SHA-256 hash for the specified file for deduplication."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()
        logging.info(f"Generated hash for {file_path}: {file_hash}")
        return file_hash

    @staticmethod
    def execute_in_threads(target_function, args_list, max_threads=4):
        """Executes a function in multiple threads with a limit on the maximum threads."""
        threads = []
        for args in args_list:
            thread = threading.Thread(target=target_function, args=args)
            threads.append(thread)
            thread.start()
            if len(threads) >= max_threads:
                for t in threads:
                    t.join()
                threads = []
        for t in threads:
            t.join()

    @staticmethod
    def run_with_retries(func, retries=3, delay=2, *args, **kwargs):
        """Runs a function with retry attempts and delay between retries."""
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                time.sleep(delay)
        logging.error(f"Failed to execute {func.__name__} after {retries} attempts.")
        raise

    @staticmethod
    def file_size_within_limit(file: UploadFile, max_size_mb: int) -> bool:
        """Checks if a file’s size is within a specified limit in megabytes without saving it."""
        try:
            file.file.seek(0, 2)  # Move the pointer to the end of the file
            file_size = file.file.tell() / (1024 * 1024)  # File size in MB
            file.file.seek(0)  # Reset the pointer to the beginning
            if file_size > max_size_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds the limit of {max_size_mb} MB.",
                )
                return HTTPException(
                    status_code=400,
                    detail=f"File size exceeds the limit of {max_size_mb} MB.",
                )
            return True
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error checking file size: {str(e)}",
            )

    @staticmethod
    def validate_file_type(file: UploadFile, allowed_extensions: list) -> bool:
        """Validates file extension based on allowed types without saving it."""
        ext = file.filename.split(".")[-1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type '.{ext}' is not allowed. Allowed types: {allowed_extensions}.",
            )
            return HTTPException(
                status_code=400,
                detail=f"File type '.{ext}' is not allowed. Allowed types: {allowed_extensions}.",
            )
        return True

    @staticmethod
    def get_model_configuration(config):
        """Determines which model setup to use based on configuration."""
        try:
            use_full_pipeline = config["settings"].get("use_full_pipeline", False)
            logging.info(
                f"Using {'full pipeline' if use_full_pipeline else 'single model'} configuration."
            )
            return use_full_pipeline
        except KeyError as e:
            logging.error(f"Config missing 'use_full_pipeline' setting: {e}")
            raise

    @staticmethod
    def load_models_for_pipeline(config):
        """Loads models based on configuration settings for the pipeline."""
        try:
            from utils.model import (
                ModelLoader,
            )  # Imported here to avoid circular dependency

            use_full_pipeline = Utils.get_model_configuration(config)
            model_loader = ModelLoader(config_file_path="config.yml")

            primary_model = model_loader.primary_model_loader
            text_embedding_model = None
            image_embedding_model = None

            if use_full_pipeline:
                text_embedding_model = model_loader.text_embedding_model_loader
                image_embedding_model = model_loader.image_embedding_model_loader
                logging.info(
                    "Full pipeline setup with primary, text, and image embedding models."
                )
            else:
                logging.info("Using primary model only for pipeline.")

            return primary_model, text_embedding_model, image_embedding_model
        except Exception as e:
            logging.error(f"Error loading models for pipeline: {e}")
            raise


class JobTracker:
    """Tracks jobs and their status, ensuring thread-safe updates."""

    _job_counter = 0  # Class-level counter for generating unique job IDs
    _counter_lock = threading.Lock()

    def __init__(self):
        self.jobs = {}
        self.job_lock = threading.Lock()

    def _generate_job_id(self):
        """Generates a job ID with a timestamp and incrementing counter."""
        with self._counter_lock:
            JobTracker._job_counter += 1
            counter = JobTracker._job_counter

        # Use the current timestamp formatted to the second level
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_{counter}"

    def create_job(self):
        """Creates a new job entry and returns the job ID."""
        with self.job_lock:
            job_id = self._generate_job_id()
            self.jobs[job_id] = {
                "status": "Started",
                "start_time": datetime.now(),
                "end_time": None,
            }
            logging.info(f"Job {job_id} started.")
        return job_id

    def update_job_status(self, job_id, status):
        """Updates the status of a job by its job ID."""
        with self.job_lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = status
                if status in ["Completed", "Aborted"]:
                    self.jobs[job_id]["end_time"] = datetime.now()
                    logging.info(f"Job {job_id} {status}.")
            else:
                raise ValueError(f"Job {job_id} does not exist.")

    def get_job_history(self):
        """Retrieves a copy of the job history."""
        with self.job_lock:
            return self.jobs.copy()
