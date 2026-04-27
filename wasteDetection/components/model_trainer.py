import os
import sys
import yaml
import subprocess
from wasteDetection.utils.main_utils import read_yaml_file
from wasteDetection.logger import logging
from wasteDetection.exception import AppException
from wasteDetection.entity.config_entity import ModelTrainerConfig
from wasteDetection.entity.artifacts_entity import ModelTrainerArtifact



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method")

        try:
            # ---------------------------
            # Unzip data
            # ---------------------------
            logging.info("Unzipping data.zip")
            subprocess.run(["unzip", "data.zip"], check=True)
            os.remove("data.zip")

            # ---------------------------
            # Read number of classes
            # ---------------------------
            with open("data.yaml", 'r') as stream:
                data_yaml = yaml.safe_load(stream)
                num_classes = int(data_yaml['nc'])

            # ---------------------------
            # Prepare model config
            # ---------------------------
            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            logging.info(f"Model config base: {model_config_file_name}")

            config_path = f"yolov5/models/{model_config_file_name}.yaml"
            config = read_yaml_file(config_path)

            config['nc'] = num_classes

            custom_config_path = f"yolov5/models/custom_{model_config_file_name}.yaml"

            with open(custom_config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)

            # ---------------------------
            # Train model
            # ---------------------------
            train_command = [
                "python", "train.py",
                "--img", "416",
                "--batch", str(self.model_trainer_config.batch_size),
                "--epochs", str(self.model_trainer_config.no_epochs),
                "--data", "../data.yaml",
                "--cfg", f"./models/custom_{model_config_file_name}.yaml",
                "--weights", self.model_trainer_config.weight_name,
                "--name", "yolov5_results",
                "--cache"
            ]

            subprocess.run(train_command, cwd="yolov5", check=True)

            # ---------------------------
            # Copy best model
            # ---------------------------
            best_model_src = "yolov5/runs/train/yolov5_results/weights/best.pt"
            best_model_dest = "yolov5/best.pt"

            subprocess.run(["cp", best_model_src, best_model_dest], check=True)

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            subprocess.run([
                "cp",
                best_model_src,
                os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            ], check=True)

            # ---------------------------
            # Cleanup
            # ---------------------------
            subprocess.run(["rm", "-rf", "yolov5/runs"], check=True)

            for folder in ["train", "valid"]:
                if os.path.exists(folder):
                    subprocess.run(["rm", "-rf", folder], check=True)

            if os.path.exists("data.yaml"):
                os.remove("data.yaml")

            # ---------------------------
            # Return artifact
            # ---------------------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Training completed successfully")
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise AppException(e, sys)