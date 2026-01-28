"""Health check system for validating dependencies."""

import sys
import importlib
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests

from logger_config import get_logger

logger = get_logger(__name__)


class HealthChecker:
    """System health checker for validating dependencies."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, bool] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def check_python_version(
        self,
        min_version: tuple = (3, 8),
        max_version: Optional[tuple] = None,
    ) -> bool:
        """Check Python version.

        Args:
            min_version: Minimum required version tuple
            max_version: Optional maximum version tuple

        Returns:
            True if version is compatible
        """
        current = sys.version_info[:2]

        if current < min_version:
            self.errors.append(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"found {current[0]}.{current[1]}"
            )
            return False

        if max_version and current > max_version:
            self.warnings.append(
                f"Python {max_version[0]}.{max_version[1]} or older recommended, "
                f"found {current[0]}.{current[1]}"
            )

        self.checks["python_version"] = True
        return True

    def check_required_packages(
        self,
        packages: List[str],
    ) -> bool:
        """Check if required packages are installed.

        Args:
            packages: List of package names to check

        Returns:
            True if all packages available
        """
        missing = []

        for package in packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(package)

        if missing:
            self.errors.append(f"Missing required packages: {', '.join(missing)}")
            self.checks["required_packages"] = False
            return False

        self.checks["required_packages"] = True
        return True

    def check_fastf1_api(self, timeout: int = 10) -> bool:
        """Check FastF1 API connectivity.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if API is accessible
        """
        try:
            # Try to import fastf1
            import fastf1

            # Simple connectivity check
            response = requests.get(
                "https://ergast.com/api/f1/current.json",
                timeout=timeout,
            )

            if response.status_code == 200:
                self.checks["fastf1_api"] = True
                return True
            else:
                self.warnings.append(
                    f"FastF1 API returned status {response.status_code}"
                )
                self.checks["fastf1_api"] = False
                return False

        except requests.RequestException as e:
            self.warnings.append(f"FastF1 API check failed: {e}")
            self.checks["fastf1_api"] = False
            return False
        except ImportError:
            self.errors.append("FastF1 package not installed")
            self.checks["fastf1_api"] = False
            return False

    def check_cache_directory(self, cache_path: str) -> bool:
        """Check cache directory accessibility.

        Args:
            cache_path: Path to cache directory

        Returns:
            True if accessible and writable
        """
        try:
            cache_dir = Path(cache_path)

            # Create if doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = cache_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            self.checks["cache_directory"] = True
            return True

        except Exception as e:
            self.errors.append(f"Cache directory not accessible: {e}")
            self.checks["cache_directory"] = False
            return False

    def check_output_directory(self, output_path: str) -> bool:
        """Check output directory accessibility.

        Args:
            output_path: Path to output directory

        Returns:
            True if accessible and writable
        """
        try:
            output_dir = Path(output_path)

            # Create if doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = output_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            self.checks["output_directory"] = True
            return True

        except Exception as e:
            self.errors.append(f"Output directory not accessible: {e}")
            self.checks["output_directory"] = False
            return False

    def check_memory_available(self, min_mb: int = 500) -> bool:
        """Check available system memory.

        Args:
            min_mb: Minimum required memory in MB

        Returns:
            True if sufficient memory available
        """
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_mb = mem.available / 1024 / 1024

            if available_mb < min_mb:
                self.warnings.append(
                    f"Low memory: {available_mb:.0f}MB available "
                    f"(recommended: {min_mb}MB+)"
                )
                self.checks["memory_available"] = False
                return False

            self.checks["memory_available"] = True
            return True

        except ImportError:
            self.warnings.append("psutil not available, cannot check memory")
            return True
        except Exception as e:
            self.warnings.append(f"Memory check failed: {e}")
            return True

    def check_disk_space(
        self,
        path: str,
        min_gb: float = 1.0,
    ) -> bool:
        """Check available disk space.

        Args:
            path: Path to check
            min_gb: Minimum required space in GB

        Returns:
            True if sufficient space available
        """
        try:
            import shutil

            stat = shutil.disk_usage(path)
            available_gb = stat.free / 1024 / 1024 / 1024

            if available_gb < min_gb:
                self.warnings.append(
                    f"Low disk space: {available_gb:.1f}GB available "
                    f"(recommended: {min_gb}GB+)"
                )
                self.checks["disk_space"] = False
                return False

            self.checks["disk_space"] = True
            return True

        except Exception as e:
            self.warnings.append(f"Disk space check failed: {e}")
            return True

    def run_all_checks(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Run all health checks.

        Args:
            config_dict: Optional configuration dictionary

        Returns:
            True if all critical checks pass
        """
        logger.info("Running health checks...")

        # Python version
        self.check_python_version()

        # Required packages
        required = [
            "fastf1",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "yaml",
            "customtkinter",
        ]
        self.check_required_packages(required)

        # FastF1 API
        self.check_fastf1_api()

        # Configuration-based checks
        if config_dict:
            # Cache directory
            if "data" in config_dict and "cache_path" in config_dict["data"]:
                self.check_cache_directory(config_dict["data"]["cache_path"])

            # Output directory
            if "data" in config_dict and "output_folder" in config_dict["data"]:
                self.check_output_directory(config_dict["data"]["output_folder"])

        # System resources
        self.check_memory_available()
        self.check_disk_space(".")

        # Report results
        passed = sum(1 for v in self.checks.values() if v)
        total = len(self.checks)

        logger.info(f"Health checks: {passed}/{total} passed")

        if self.errors:
            logger.error(f"Critical errors found ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  ❌ {error}")

        if self.warnings:
            logger.warning(f"Warnings found ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")

        # Return True only if no critical errors
        return len(self.errors) == 0

    def get_report(self) -> Dict[str, Any]:
        """Get health check report.

        Returns:
            Dictionary with check results
        """
        return {
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": sum(1 for v in self.checks.values() if v),
            "total": len(self.checks),
        }


def run_health_check(config_dict: Optional[Dict[str, Any]] = None) -> bool:
    """Run system health check.

    Args:
        config_dict: Optional configuration dictionary

    Returns:
        True if all checks pass
    """
    checker = HealthChecker()
    return checker.run_all_checks(config_dict)
