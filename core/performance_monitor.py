"""
Performance monitoring and system resource tracking for YOLO Trainer
"""

import os
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import json

# Optional GPUtil import
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from . import logger, format_timestamp, ensure_directory

class PerformanceMonitor:
    """Monitor system performance during training"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.performance_log = self.log_dir / "performance.jsonl"
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []
        self.monitoring_interval = 5  # seconds
        ensure_directory(str(self.log_dir))

    def start_monitoring(self, session_id: Optional[str] = None) -> bool:
        """Start performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return False

        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session_id = session_id
        self.is_monitoring = True

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()

        logger.info(f"Started performance monitoring: {session_id}")
        return True

    def stop_monitoring(self) -> bool:
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return False

        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)

        logger.info("Stopped performance monitoring")
        return True

    def add_callback(self, callback: Callable):
        """Add callback for performance updates"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove performance callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._log_metrics(metrics)
                self._notify_callbacks(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(1)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            "timestamp": format_timestamp(),
            "session_id": self.session_id,
            "cpu": self._get_cpu_metrics(),
            "memory": self._get_memory_metrics(),
            "disk": self._get_disk_metrics(),
            "gpu": self._get_gpu_metrics(),
            "network": self._get_network_metrics()
        }

        return metrics

    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU performance metrics"""
        try:
            return {
                "usage_percent": psutil.cpu_percent(interval=1),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            logger.warning(f"Failed to get CPU metrics: {e}")
            return {"error": str(e)}

    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory performance metrics"""
        try:
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "usage_percent": mem.percent,
                "swap_total_gb": round(psutil.swap_memory().total / (1024**3), 2),
                "swap_used_gb": round(psutil.swap_memory().used / (1024**3), 2),
                "swap_percent": psutil.swap_memory().percent
            }
        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
            return {"error": str(e)}

    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk performance metrics"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()

            metrics = {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "usage_percent": disk_usage.percent
            }

            if disk_io:
                metrics.update({
                    "read_count": disk_io.read_count,
                    "write_count": disk_io.write_count,
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes
                })

            return metrics
        except Exception as e:
            logger.warning(f"Failed to get disk metrics: {e}")
            return {"error": str(e)}

    def _get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get GPU performance metrics"""
        if not GPUTIL_AVAILABLE:
            return [{"error": "GPUtil not available"}]

        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = []

            for gpu in gpus:
                gpu_metrics.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_utilization": gpu.memoryUtil * 100,
                    "gpu_utilization": gpu.load * 100,
                    "temperature": gpu.temperature
                })

            return gpu_metrics
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return [{"error": str(e)}]

    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network performance metrics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout
            }
        except Exception as e:
            logger.warning(f"Failed to get network metrics: {e}")
            return {"error": str(e)}

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to file"""
        try:
            with open(self.performance_log, 'a', encoding='utf-8') as f:
                json.dump(metrics, f, default=str)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")

    def _notify_callbacks(self, metrics: Dict[str, Any]):
        """Notify registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Performance callback error: {e}")

    def get_performance_history(self, session_id: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history for a session"""
        try:
            history = []
            cutoff_time = datetime.now().timestamp() - (hours * 3600)

            if not self.performance_log.exists():
                return history

            with open(self.performance_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if session_id and entry.get("session_id") != session_id:
                            continue

                        # Convert timestamp to check if within time range
                        entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
                        if entry_time >= cutoff_time:
                            history.append(entry)
                    except json.JSONDecodeError:
                        continue

            return history

        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []

    def generate_performance_report(self, session_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            history = self.get_performance_history(session_id, hours)

            if not history:
                return {"error": "No performance data available"}

            # Calculate averages and peaks
            report = {
                "session_id": session_id,
                "time_range_hours": hours,
                "total_entries": len(history),
                "averages": {},
                "peaks": {},
                "trends": {}
            }

            # CPU metrics
            cpu_usage = [entry["cpu"]["usage_percent"] for entry in history if "cpu" in entry and "usage_percent" in entry["cpu"]]
            if cpu_usage:
                report["averages"]["cpu_usage_percent"] = sum(cpu_usage) / len(cpu_usage)
                report["peaks"]["cpu_usage_percent"] = max(cpu_usage)

            # Memory metrics
            mem_usage = [entry["memory"]["usage_percent"] for entry in history if "memory" in entry and "usage_percent" in entry["memory"]]
            if mem_usage:
                report["averages"]["memory_usage_percent"] = sum(mem_usage) / len(mem_usage)
                report["peaks"]["memory_usage_percent"] = max(mem_usage)

            # GPU metrics
            gpu_usage = []
            for entry in history:
                if "gpu" in entry and entry["gpu"]:
                    for gpu in entry["gpu"]:
                        if "gpu_utilization" in gpu:
                            gpu_usage.append(gpu["gpu_utilization"])

            if gpu_usage:
                report["averages"]["gpu_usage_percent"] = sum(gpu_usage) / len(gpu_usage)
                report["peaks"]["gpu_usage_percent"] = max(gpu_usage)

            # Generate recommendations
            report["recommendations"] = self._generate_performance_recommendations(report)

            return report

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}

    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []

        averages = report.get("averages", {})
        peaks = report.get("peaks", {})

        # CPU recommendations
        cpu_avg = averages.get("cpu_usage_percent", 0)
        if cpu_avg > 90:
            recommendations.append("High CPU usage detected - consider using more CPU cores or optimizing data loading")
        elif cpu_avg < 30:
            recommendations.append("CPU underutilized - consider increasing batch size or using more workers")

        # Memory recommendations
        mem_avg = averages.get("memory_usage_percent", 0)
        if mem_avg > 90:
            recommendations.append("High memory usage - consider reducing batch size or using gradient accumulation")
        elif mem_avg < 50:
            recommendations.append("Memory underutilized - can potentially increase batch size")

        # GPU recommendations
        gpu_avg = averages.get("gpu_usage_percent", 0)
        if gpu_avg > 95:
            recommendations.append("GPU fully utilized - good performance")
        elif gpu_avg < 70:
            recommendations.append("GPU underutilized - consider increasing batch size or model complexity")

        return recommendations

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
