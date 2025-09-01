"""
Model comparison and analytics for YOLO Trainer
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

# Optional seaborn import
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from . import logger, format_timestamp, ensure_directory

class ModelComparator:
    """Compare multiple trained YOLO models"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.comparisons_dir = self.models_dir / "comparisons"
        ensure_directory(str(self.comparisons_dir))

    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models and generate analysis"""
        try:
            model_data = []

            for model_name in model_names:
                model_info = self._load_model_info(model_name)
                if model_info:
                    model_data.append(model_info)

            if not model_data:
                return {"error": "No valid model data found"}

            # Generate comparison report
            comparison = {
                "timestamp": format_timestamp(),
                "models_compared": model_names,
                "summary": self._generate_summary(model_data),
                "performance_comparison": self._compare_performance(model_data),
                "training_comparison": self._compare_training(model_data),
                "recommendations": self._generate_recommendations(model_data)
            }

            # Save comparison
            comparison_file = self.comparisons_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {"error": str(e)}

    def _load_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load information for a specific model"""
        try:
            model_dir = self.models_dir / model_name
            if not model_dir.exists():
                return None

            # Load training config
            config_file = model_dir / "training_config.json"
            if not config_file.exists():
                return None

            with open(config_file, 'r') as f:
                config = json.load(f)

            # Load results if available
            results_file = model_dir / "results.csv"
            results = None
            if results_file.exists():
                results = pd.read_csv(results_file)

            return {
                "name": model_name,
                "config": config,
                "results": results,
                "model_dir": str(model_dir),
                "created": config.get("created", ""),
                "dataset": config.get("dataset", "")
            }

        except Exception as e:
            logger.error(f"Failed to load model info for {model_name}: {e}")
            return None

    def _generate_summary(self, model_data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            "total_models": len(model_data),
            "datasets_used": list(set(m["dataset"] for m in model_data if m["dataset"])),
            "best_performing": None,
            "fastest_training": None,
            "most_recent": None
        }

        # Find best performing model
        best_fitness = -1
        fastest_time = float('inf')
        most_recent = None

        for model in model_data:
            if model["results"] is not None:
                fitness = model["results"]["mAP50-95(B)"].max() if "mAP50-95(B)" in model["results"].columns else 0
                if fitness > best_fitness:
                    best_fitness = fitness
                    summary["best_performing"] = model["name"]

            # Training time comparison (if available)
            if "training_time" in model["config"]:
                train_time = model["config"]["training_time"]
                if train_time < fastest_time:
                    fastest_time = train_time
                    summary["fastest_training"] = model["name"]

            # Most recent
            if not most_recent or model["created"] > most_recent:
                most_recent = model["created"]
                summary["most_recent"] = model["name"]

        return summary

    def _compare_performance(self, model_data: List[Dict]) -> Dict[str, Any]:
        """Compare model performance metrics"""
        performance = {}

        for model in model_data:
            if model["results"] is not None:
                metrics = {}
                for col in model["results"].columns:
                    if col.startswith("mAP") or col in ["precision", "recall", "f1-score"]:
                        metrics[col] = model["results"][col].max()

                performance[model["name"]] = metrics

        return performance

    def _compare_training(self, model_data: List[Dict]) -> Dict[str, Any]:
        """Compare training configurations and results"""
        training_comparison = {}

        for model in model_data:
            training_comparison[model["name"]] = {
                "epochs": model["config"].get("epochs", 0),
                "batch_size": model["config"].get("batch_size", 0),
                "img_size": model["config"].get("img_size", 0),
                "base_model": model["config"].get("model_path", ""),
                "dataset": model["dataset"],
                "final_metrics": {}
            }

            # Add final metrics if available
            if model["results"] is not None and not model["results"].empty:
                final_row = model["results"].iloc[-1]
                for col in model["results"].columns:
                    if any(keyword in col.lower() for keyword in ["map", "precision", "recall", "f1"]):
                        training_comparison[model["name"]]["final_metrics"][col] = final_row[col]

        return training_comparison

    def _generate_recommendations(self, model_data: List[Dict]) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []

        # Find best model
        best_model = None
        best_score = -1

        for model in model_data:
            if model["results"] is not None:
                score = model["results"]["mAP50-95(B)"].max() if "mAP50-95(B)" in model["results"].columns else 0
                if score > best_score:
                    best_score = score
                    best_model = model

        if best_model:
            recommendations.append(f"Best performing model: {best_model['name']} (mAP: {best_score:.3f})")

        # Check for overfitting patterns
        for model in model_data:
            if model["results"] is not None and len(model["results"]) > 1:
                train_loss = model["results"].get("train/box_loss", pd.Series()).iloc[-1] if "train/box_loss" in model["results"].columns else 0
                val_loss = model["results"].get("val/box_loss", pd.Series()).iloc[-1] if "val/box_loss" in model["results"].columns else 0

                if train_loss > 0 and val_loss > 0 and train_loss < val_loss * 0.8:
                    recommendations.append(f"Model {model['name']} may be overfitting - consider regularization")

        # Suggest improvements
        recommendations.append("Consider using data augmentation for better generalization")
        recommendations.append("Experiment with different learning rates and batch sizes")

        return recommendations

    def generate_comparison_report(self, comparison_data: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a detailed comparison report"""
        if not output_file:
            output_file = str(self.comparisons_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        try:
            with open(output_file, 'w') as f:
                f.write("YOLO Model Comparison Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {format_timestamp()}\n\n")

                # Summary
                f.write("SUMMARY\n")
                f.write("-" * 20 + "\n")
                summary = comparison_data.get("summary", {})
                f.write(f"Models compared: {summary.get('total_models', 0)}\n")
                f.write(f"Datasets used: {', '.join(summary.get('datasets_used', []))}\n")
                f.write(f"Best performing: {summary.get('best_performing', 'N/A')}\n")
                f.write(f"Fastest training: {summary.get('fastest_training', 'N/A')}\n\n")

                # Performance comparison
                f.write("PERFORMANCE COMPARISON\n")
                f.write("-" * 30 + "\n")
                performance = comparison_data.get("performance_comparison", {})
                for model, metrics in performance.items():
                    f.write(f"\n{model}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")

                # Recommendations
                f.write("\nRECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                for rec in comparison_data.get("recommendations", []):
                    f.write(f"• {rec}\n")

            return str(output_file)

        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            return ""

# Global model comparator instance
model_comparator = ModelComparator()
