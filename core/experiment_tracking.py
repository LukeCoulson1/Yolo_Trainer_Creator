"""
Experiment tracking and management for YOLO Trainer
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from . import logger, format_timestamp, ensure_directory

class ExperimentTracker:
    """Track and manage YOLO training experiments"""

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.current_experiment: Optional[Dict[str, Any]] = None
        ensure_directory(str(self.experiments_dir))

    def create_experiment(self, name: str, description: str = "",
                         config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new experiment"""
        try:
            # Sanitize name
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')

            if not safe_name:
                safe_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            experiment_dir = self.experiments_dir / safe_name
            experiment_dir.mkdir(parents=True, exist_ok=True)

            experiment_id = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            experiment_data = {
                "id": experiment_id,
                "name": name,
                "safe_name": safe_name,
                "description": description,
                "created": format_timestamp(),
                "status": "created",
                "config": config or {},
                "runs": [],
                "best_run": None,
                "tags": [],
                "notes": ""
            }

            # Save experiment metadata
            metadata_file = experiment_dir / "experiment.json"
            with open(metadata_file, 'w') as f:
                json.dump(experiment_data, f, indent=2, default=str)

            self.current_experiment = experiment_data
            logger.info(f"Created experiment: {experiment_id}")

            return experiment_id

        except Exception as e:
            logger.error(f"Failed to create experiment '{name}': {e}")
            return None

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load an existing experiment"""
        try:
            # Find experiment directory
            for exp_dir in self.experiments_dir.iterdir():
                if exp_dir.is_dir():
                    metadata_file = exp_dir / "experiment.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            experiment_data = json.load(f)

                        if experiment_data.get("id") == experiment_id:
                            self.current_experiment = experiment_data
                            return experiment_data

            logger.warning(f"Experiment not found: {experiment_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to load experiment '{experiment_id}': {e}")
            return None

    def add_run(self, model_name: str, config: Dict[str, Any],
               results: Optional[Dict[str, Any]] = None) -> bool:
        """Add a training run to the current experiment"""
        if not self.current_experiment:
            logger.error("No active experiment")
            return False

        try:
            run_data = {
                "id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "model_name": model_name,
                "config": config,
                "results": results or {},
                "timestamp": format_timestamp(),
                "status": "completed" if results else "running"
            }

            self.current_experiment["runs"].append(run_data)

            # Update best run if this is better
            if results and self._is_better_run(run_data, self.current_experiment.get("best_run")):
                self.current_experiment["best_run"] = run_data["id"]

            # Save updated experiment
            self._save_experiment()

            logger.info(f"Added run to experiment: {run_data['id']}")
            return True

        except Exception as e:
            logger.error(f"Failed to add run: {e}")
            return False

    def update_run_results(self, run_id: str, results: Dict[str, Any]) -> bool:
        """Update results for a specific run"""
        if not self.current_experiment:
            return False

        try:
            for run in self.current_experiment["runs"]:
                if run["id"] == run_id:
                    run["results"] = results
                    run["status"] = "completed"

                    # Update best run if necessary
                    if self._is_better_run(run, self.current_experiment.get("best_run")):
                        self.current_experiment["best_run"] = run_id

                    self._save_experiment()
                    return True

            logger.warning(f"Run not found: {run_id}")
            return False

        except Exception as e:
            logger.error(f"Failed to update run results: {e}")
            return False

    def _is_better_run(self, run: Dict[str, Any], current_best_id: Optional[str]) -> bool:
        """Determine if a run is better than the current best"""
        if not current_best_id:
            return True

        # Find current best run
        current_best = None
        if self.current_experiment:
            for r in self.current_experiment["runs"]:
                if r["id"] == current_best_id:
                    current_best = r
                    break

        if not current_best or not current_best.get("results"):
            return True

        # Compare mAP scores (higher is better)
        new_score = run.get("results", {}).get("mAP50-95(B)", 0)
        best_score = current_best.get("results", {}).get("mAP50-95(B)", 0)

        return new_score > best_score

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        experiments = []

        try:
            for exp_dir in self.experiments_dir.iterdir():
                if exp_dir.is_dir():
                    metadata_file = exp_dir / "experiment.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                experiment_data = json.load(f)
                            experiments.append(experiment_data)
                        except Exception as e:
                            logger.warning(f"Could not read experiment in {exp_dir}: {e}")

            # Sort by creation date (most recent first)
            experiments.sort(key=lambda x: x.get("created", ""), reverse=True)

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")

        return experiments

    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for an experiment"""
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            return None

        try:
            runs = experiment.get("runs", [])
            completed_runs = [r for r in runs if r.get("status") == "completed"]

            summary = {
                "experiment_id": experiment_id,
                "name": experiment.get("name", ""),
                "total_runs": len(runs),
                "completed_runs": len(completed_runs),
                "best_run": experiment.get("best_run"),
                "created": experiment.get("created"),
                "last_updated": max([r.get("timestamp", "") for r in runs]) if runs else experiment.get("created")
            }

            # Calculate performance statistics
            if completed_runs:
                mAP_scores = []
                training_times = []

                for run in completed_runs:
                    results = run.get("results", {})
                    if "mAP50-95(B)" in results:
                        mAP_scores.append(results["mAP50-95(B)"])
                    if "training_time" in run.get("config", {}):
                        training_times.append(run["config"]["training_time"])

                if mAP_scores:
                    summary["best_mAP"] = max(mAP_scores)
                    summary["avg_mAP"] = sum(mAP_scores) / len(mAP_scores)
                    summary["mAP_std"] = (sum((x - summary["avg_mAP"])**2 for x in mAP_scores) / len(mAP_scores))**0.5

                if training_times:
                    summary["avg_training_time"] = sum(training_times) / len(training_times)

            return summary

        except Exception as e:
            logger.error(f"Failed to generate experiment summary: {e}")
            return None

    def export_experiment(self, experiment_id: str, export_path: str) -> bool:
        """Export experiment data"""
        try:
            experiment = self.load_experiment(experiment_id)
            if not experiment:
                return False

            export_path_obj = Path(export_path)
            export_path_obj.mkdir(parents=True, exist_ok=True)

            # Export experiment metadata
            with open(export_path_obj / "experiment.json", 'w') as f:
                json.dump(experiment, f, indent=2, default=str)

            # Export summary
            summary = self.get_experiment_summary(experiment_id)
            if summary:
                with open(export_path_obj / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2, default=str)

            logger.info(f"Experiment exported to: {export_path_obj}")
            return True

        except Exception as e:
            logger.error(f"Failed to export experiment: {e}")
            return False

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment"""
        try:
            experiment = self.load_experiment(experiment_id)
            if not experiment:
                return False

            experiment_dir = self.experiments_dir / experiment["safe_name"]

            if experiment_dir.exists():
                import shutil
                shutil.rmtree(experiment_dir)
                logger.info(f"Deleted experiment: {experiment_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete experiment '{experiment_id}': {e}")
            return False

    def add_note(self, note: str) -> bool:
        """Add a note to the current experiment"""
        if not self.current_experiment:
            return False

        try:
            if "notes" not in self.current_experiment:
                self.current_experiment["notes"] = ""

            timestamp = format_timestamp()
            self.current_experiment["notes"] += f"\n[{timestamp}] {note}"

            self._save_experiment()
            return True

        except Exception as e:
            logger.error(f"Failed to add note: {e}")
            return False

    def add_tags(self, tags: List[str]) -> bool:
        """Add tags to the current experiment"""
        if not self.current_experiment:
            return False

        try:
            if "tags" not in self.current_experiment:
                self.current_experiment["tags"] = []

            for tag in tags:
                if tag not in self.current_experiment["tags"]:
                    self.current_experiment["tags"].append(tag)

            self._save_experiment()
            return True

        except Exception as e:
            logger.error(f"Failed to add tags: {e}")
            return False

    def _save_experiment(self):
        """Save current experiment to disk"""
        if not self.current_experiment:
            return

        try:
            experiment_dir = self.experiments_dir / self.current_experiment["safe_name"]
            metadata_file = experiment_dir / "experiment.json"

            with open(metadata_file, 'w') as f:
                json.dump(self.current_experiment, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")

# Global experiment tracker instance
experiment_tracker = ExperimentTracker()
