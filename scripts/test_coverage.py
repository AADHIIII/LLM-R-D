#!/usr/bin/env python3
"""
Test coverage and quality metrics script.
"""
import os
import sys
import subprocess
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
import argparse


class TestCoverageAnalyzer:
    """Analyze test coverage and generate quality metrics."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.coverage_file = self.project_root / "coverage.xml"
        self.html_coverage_dir = self.project_root / "htmlcov"
        
    def run_tests_with_coverage(self) -> Dict[str, Any]:
        """Run tests and generate coverage report."""
        print("Running tests with coverage...")
        
        # Run pytest with coverage
        cmd = [
            "python", "-m", "pytest",
            "--cov=.",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Tests timed out after 5 minutes",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": -1
            }
    
    def parse_coverage_xml(self) -> Dict[str, Any]:
        """Parse coverage XML report."""
        if not self.coverage_file.exists():
            return {"error": "Coverage XML file not found"}
        
        try:
            tree = ET.parse(self.coverage_file)
            root = tree.getroot()
            
            # Extract overall coverage
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get("line-rate", 0))
                branch_rate = float(coverage_elem.get("branch-rate", 0))
            else:
                line_rate = 0
                branch_rate = 0
            
            # Extract package/module coverage
            packages = []
            for package in root.findall(".//package"):
                package_name = package.get("name", "unknown")
                package_line_rate = float(package.get("line-rate", 0))
                package_branch_rate = float(package.get("branch-rate", 0))
                
                classes = []
                for class_elem in package.findall(".//class"):
                    class_name = class_elem.get("name", "unknown")
                    class_line_rate = float(class_elem.get("line-rate", 0))
                    class_branch_rate = float(class_elem.get("branch-rate", 0))
                    
                    classes.append({
                        "name": class_name,
                        "line_coverage": class_line_rate * 100,
                        "branch_coverage": class_branch_rate * 100
                    })
                
                packages.append({
                    "name": package_name,
                    "line_coverage": package_line_rate * 100,
                    "branch_coverage": package_branch_rate * 100,
                    "classes": classes
                })
            
            return {
                "overall": {
                    "line_coverage": line_rate * 100,
                    "branch_coverage": branch_rate * 100
                },
                "packages": packages
            }
            
        except Exception as e:
            return {"error": f"Failed to parse coverage XML: {e}"}
    
    def analyze_test_files(self) -> Dict[str, Any]:
        """Analyze test files and structure."""
        test_dirs = ["tests", "test"]
        test_files = []
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                for test_file in test_path.rglob("test_*.py"):
                    test_files.append({
                        "path": str(test_file.relative_to(self.project_root)),
                        "size": test_file.stat().st_size,
                        "lines": self._count_lines(test_file)
                    })
        
        return {
            "total_test_files": len(test_files),
            "test_files": test_files,
            "total_test_lines": sum(f["lines"] for f in test_files)
        }
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def analyze_source_files(self) -> Dict[str, Any]:
        """Analyze source code files."""
        source_files = []
        
        # Common source directories
        source_dirs = ["api", "database", "evaluator", "fine_tuning", "utils", "web_interface"]
        
        for source_dir in source_dirs:
            source_path = self.project_root / source_dir
            if source_path.exists():
                for py_file in source_path.rglob("*.py"):
                    if not py_file.name.startswith("__"):
                        source_files.append({
                            "path": str(py_file.relative_to(self.project_root)),
                            "size": py_file.stat().st_size,
                            "lines": self._count_lines(py_file)
                        })
        
        return {
            "total_source_files": len(source_files),
            "source_files": source_files,
            "total_source_lines": sum(f["lines"] for f in source_files)
        }
    
    def calculate_quality_metrics(self, coverage_data: Dict[str, Any], 
                                test_data: Dict[str, Any], 
                                source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        
        # Coverage metrics
        line_coverage = coverage_data.get("overall", {}).get("line_coverage", 0)
        branch_coverage = coverage_data.get("overall", {}).get("branch_coverage", 0)
        
        # Test metrics
        total_test_files = test_data.get("total_test_files", 0)
        total_test_lines = test_data.get("total_test_lines", 0)
        
        # Source metrics
        total_source_files = source_data.get("total_source_files", 0)
        total_source_lines = source_data.get("total_source_lines", 0)
        
        # Calculate ratios
        test_to_source_ratio = (total_test_lines / total_source_lines * 100) if total_source_lines > 0 else 0
        test_file_ratio = (total_test_files / total_source_files * 100) if total_source_files > 0 else 0
        
        # Quality score (weighted average)
        quality_score = (
            line_coverage * 0.4 +
            branch_coverage * 0.3 +
            min(test_to_source_ratio, 100) * 0.2 +
            min(test_file_ratio, 100) * 0.1
        )
        
        return {
            "coverage": {
                "line_coverage": line_coverage,
                "branch_coverage": branch_coverage,
                "average_coverage": (line_coverage + branch_coverage) / 2
            },
            "testing": {
                "test_files": total_test_files,
                "test_lines": total_test_lines,
                "test_to_source_ratio": test_to_source_ratio,
                "test_file_ratio": test_file_ratio
            },
            "source": {
                "source_files": total_source_files,
                "source_lines": total_source_lines
            },
            "quality_score": quality_score,
            "grade": self._calculate_grade(quality_score)
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade based on quality score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test coverage and quality report."""
        print("Generating test coverage and quality report...")
        
        # Run tests
        test_results = self.run_tests_with_coverage()
        
        if not test_results["success"]:
            return {
                "success": False,
                "error": "Tests failed",
                "test_output": test_results
            }
        
        # Parse coverage
        coverage_data = self.parse_coverage_xml()
        
        # Analyze files
        test_data = self.analyze_test_files()
        source_data = self.analyze_source_files()
        
        # Calculate metrics
        quality_metrics = self.calculate_quality_metrics(coverage_data, test_data, source_data)
        
        return {
            "success": True,
            "timestamp": subprocess.check_output(["date"], text=True).strip(),
            "test_results": test_results,
            "coverage": coverage_data,
            "test_analysis": test_data,
            "source_analysis": source_data,
            "quality_metrics": quality_metrics
        }
    
    def save_report(self, report: Dict[str, Any], output_file: str = "quality_report.json"):
        """Save report to JSON file."""
        output_path = self.project_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print summary of quality metrics."""
        if not report["success"]:
            print("❌ Report generation failed")
            return
        
        quality = report["quality_metrics"]
        
        print("\n" + "="*60)
        print("📊 TEST COVERAGE AND QUALITY REPORT")
        print("="*60)
        
        print(f"\n🎯 Overall Quality Score: {quality['quality_score']:.1f}/100 (Grade: {quality['grade']})")
        
        print(f"\n📈 Coverage Metrics:")
        print(f"  • Line Coverage: {quality['coverage']['line_coverage']:.1f}%")
        print(f"  • Branch Coverage: {quality['coverage']['branch_coverage']:.1f}%")
        print(f"  • Average Coverage: {quality['coverage']['average_coverage']:.1f}%")
        
        print(f"\n🧪 Testing Metrics:")
        print(f"  • Test Files: {quality['testing']['test_files']}")
        print(f"  • Test Lines: {quality['testing']['test_lines']:,}")
        print(f"  • Test-to-Source Ratio: {quality['testing']['test_to_source_ratio']:.1f}%")
        print(f"  • Test File Ratio: {quality['testing']['test_file_ratio']:.1f}%")
        
        print(f"\n📁 Source Metrics:")
        print(f"  • Source Files: {quality['source']['source_files']}")
        print(f"  • Source Lines: {quality['source']['source_lines']:,}")
        
        # Coverage recommendations
        print(f"\n💡 Recommendations:")
        if quality['coverage']['line_coverage'] < 80:
            print("  • Increase line coverage to at least 80%")
        if quality['coverage']['branch_coverage'] < 70:
            print("  • Improve branch coverage to at least 70%")
        if quality['testing']['test_to_source_ratio'] < 50:
            print("  • Add more comprehensive tests")
        if quality['quality_score'] < 80:
            print("  • Focus on improving overall code quality")
        
        print("\n" + "="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test coverage and quality analysis")
    parser.add_argument("--output", "-o", default="quality_report.json",
                       help="Output file for the report")
    parser.add_argument("--project-root", "-p", default=".",
                       help="Project root directory")
    parser.add_argument("--no-tests", action="store_true",
                       help="Skip running tests, only analyze existing coverage")
    
    args = parser.parse_args()
    
    analyzer = TestCoverageAnalyzer(args.project_root)
    
    if args.no_tests:
        # Only analyze existing coverage
        coverage_data = analyzer.parse_coverage_xml()
        test_data = analyzer.analyze_test_files()
        source_data = analyzer.analyze_source_files()
        quality_metrics = analyzer.calculate_quality_metrics(coverage_data, test_data, source_data)
        
        report = {
            "success": True,
            "coverage": coverage_data,
            "test_analysis": test_data,
            "source_analysis": source_data,
            "quality_metrics": quality_metrics
        }
    else:
        # Run full analysis
        report = analyzer.generate_report()
    
    # Save and display report
    analyzer.save_report(report, args.output)
    analyzer.print_summary(report)
    
    # Exit with appropriate code
    if report["success"]:
        quality_score = report["quality_metrics"]["quality_score"]
        if quality_score < 60:
            print("\n⚠️  Quality score below acceptable threshold (60)")
            sys.exit(1)
        else:
            print("\n✅ Quality metrics meet standards")
            sys.exit(0)
    else:
        print("\n❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()