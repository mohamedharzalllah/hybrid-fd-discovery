"""
Task Set 4 Part 1
FD Pipeline Preparation: Prepare FD lists for pipeline input
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


class FDPipelinePrep:
    """
    Prepares parsed FD data for use in the hybrid discovery pipeline
    """
    
    def __init__(self, parsed_fd_file):
        """
        Initialize with parsed FD JSON file
        
        Args:
            parsed_fd_file: Path to parsed FD JSON from fd_parser.py
        """
        self.parsed_fd_file = parsed_fd_file
        self.data = self._load_data()
        self.fds = self.data.get("fds", [])
        self.statistics = self.data.get("statistics", {})
        self.suspicious = self.data.get("suspicious", {})
    
    def _load_data(self):
        """Load parsed FD data"""
        with open(self.parsed_fd_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def filter_fds(self, min_lhs_size=None, max_lhs_size=None, 
                   exclude_id_based=False, exclude_large_lhs=False):
        """
        Filter FDs based on criteria
        
        Args:
            min_lhs_size: Minimum LHS size
            max_lhs_size: Maximum LHS size
            exclude_id_based: Remove ID-based FDs
            exclude_large_lhs: Remove FDs with large LHS
            
        Returns:
            list: Filtered FDs
        """
        filtered = self.fds.copy()
        
        # Filter by LHS size
        if min_lhs_size is not None:
            filtered = [fd for fd in filtered if fd['lhs_size'] >= min_lhs_size]
        
        if max_lhs_size is not None:
            filtered = [fd for fd in filtered if fd['lhs_size'] <= max_lhs_size]
        
        # Exclude ID-based
        if exclude_id_based:
            id_based_set = {
                tuple(fd['lhs']) + (fd['rhs'],) 
                for fd in self.suspicious.get('id_based', [])
            }
            filtered = [
                fd for fd in filtered 
                if tuple(fd['lhs']) + (fd['rhs'],) not in id_based_set
            ]
        
        # Exclude large LHS
        if exclude_large_lhs:
            large_lhs_set = {
                tuple(fd['lhs']) + (fd['rhs'],) 
                for fd in self.suspicious.get('large_lhs', [])
            }
            filtered = [
                fd for fd in filtered 
                if tuple(fd['lhs']) + (fd['rhs'],) not in large_lhs_set
            ]
        
        return filtered
    
    def categorize_fds(self):
        """
        Categorize FDs for pipeline processing
        
        Categories:
        - plausible: Normal-looking FDs (1-3 attributes in LHS)
        - suspicious: ID-based or large LHS
        - simple: Single attribute LHS
        - complex: Multi-attribute LHS
        """
        categories = {
            "plausible": [],
            "suspicious": [],
            "simple": [],
            "complex": []
        }
        
        id_based_set = {
            tuple(fd['lhs']) + (fd['rhs'],) 
            for fd in self.suspicious.get('id_based', [])
        }
        
        large_lhs_set = {
            tuple(fd['lhs']) + (fd['rhs'],) 
            for fd in self.suspicious.get('large_lhs', [])
        }
        
        for fd in self.fds:
            fd_tuple = tuple(fd['lhs']) + (fd['rhs'],)
            
            # Simple vs complex
            if fd['lhs_size'] == 1:
                categories["simple"].append(fd)
            else:
                categories["complex"].append(fd)
            
            # Suspicious vs plausible
            if fd_tuple in id_based_set or fd_tuple in large_lhs_set:
                categories["suspicious"].append(fd)
            elif 1 <= fd['lhs_size'] <= 3:
                categories["plausible"].append(fd)
        
        return categories
    
    def select_sample_fds(self, num_plausible=3, num_suspicious=3):
        """
        Select sample FDs for Task Set 2 (LLM evaluation)
        
        Args:
            num_plausible: Number of plausible FDs to select
            num_suspicious: Number of suspicious FDs to select
            
        Returns:
            dict: Selected FDs
        """
        categories = self.categorize_fds()
        
        def spread(fds, k):
            # Prefer short LHS and different RHS attributes, instead of the
            # first k FDs (which usually share the same LHS/RHS)
            ranked = sorted(fds, key=lambda fd: (fd['lhs_size'], fd['rhs'], fd['lhs']))
            picked, seen_rhs = [], set()
            for fd in ranked:
                if fd['rhs'] not in seen_rhs:
                    picked.append(fd)
                    seen_rhs.add(fd['rhs'])
                if len(picked) == k:
                    return picked
            for fd in ranked:
                if len(picked) == k:
                    break
                if fd not in picked:
                    picked.append(fd)
            return picked

        selected = {
            "plausible": spread(categories["plausible"], num_plausible),
            "suspicious": spread(categories["suspicious"], num_suspicious)
        }
        
        return selected
    
    def format_for_llm_evaluation(self, fds):
        """
        Format FDs for LLM evaluation (Task Set 2)
        
        Args:
            fds: List of FD objects
            
        Returns:
            str: Formatted text for LLM
        """
        lines = []
        lines.append("Please evaluate the following functional dependencies:")
        lines.append("")
        
        for i, fd in enumerate(fds, 1):
            lhs_str = ', '.join(fd['lhs'])
            lines.append(f"{i}. {lhs_str} -> {fd['rhs']}")
        
        lines.append("")
        lines.append("For each FD, classify it as:")
        lines.append("- meaningful (real-world constraint)")
        lines.append("- accidental (coincidence in this dataset)")
        lines.append("- encoding-based (information embedded in codes)")
        lines.append("- degenerate (ID determines everything)")
        lines.append("- unlikely (doesn't make sense)")
        
        return '\n'.join(lines)
    
    def export_pipeline_format(self, output_dir):
        """
        Export FDs in various formats for pipeline use
        
        Creates:
        - all_fds.json: All FDs
        - plausible_fds.json: Plausible FDs only
        - suspicious_fds.json: Suspicious FDs only
        - sample_for_llm.txt: Sample FDs formatted for LLM
        - categories.json: FDs by category
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # All FDs
        with open(output_dir / "all_fds.json", 'w', encoding='utf-8') as f:
            json.dump({"fds": self.fds, "statistics": self.statistics}, f, indent=2)
        
        # Categorized FDs
        categories = self.categorize_fds()
        with open(output_dir / "categories.json", 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2)
        
        # Plausible and suspicious
        with open(output_dir / "plausible_fds.json", 'w', encoding='utf-8') as f:
            json.dump({"fds": categories["plausible"]}, f, indent=2)
        
        with open(output_dir / "suspicious_fds.json", 'w', encoding='utf-8') as f:
            json.dump({"fds": categories["suspicious"]}, f, indent=2)
        
        # Sample for LLM
        sample_fds = self.select_sample_fds()
        all_sample = sample_fds["plausible"] + sample_fds["suspicious"]
        llm_text = self.format_for_llm_evaluation(all_sample)
        
        with open(output_dir / "sample_for_llm.txt", 'w', encoding='utf-8') as f:
            f.write(llm_text)
        
        with open(output_dir / "sample_fds.json", 'w', encoding='utf-8') as f:
            json.dump(sample_fds, f, indent=2)
        
        # Create summary
        summary = {
            "total_fds": len(self.fds),
            "plausible_fds": len(categories["plausible"]),
            "suspicious_fds": len(categories["suspicious"]),
            "simple_fds": len(categories["simple"]),
            "complex_fds": len(categories["complex"]),
            "statistics": self.statistics,
            "output_files": [
                "all_fds.json",
                "categories.json",
                "plausible_fds.json",
                "suspicious_fds.json",
                "sample_for_llm.txt",
                "sample_fds.json"
            ]
        }
        
        with open(output_dir / "pipeline_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_pipeline_summary(self, summary):
        """Print pipeline preparation summary"""
        print("\n" + "="*60)
        print("PIPELINE PREPARATION SUMMARY")
        print("="*60)
        
        print(f"\nTotal FDs: {summary['total_fds']}")
        print(f"  Plausible: {summary['plausible_fds']}")
        print(f"  Suspicious: {summary['suspicious_fds']}")
        print(f"  Simple (1 attr LHS): {summary['simple_fds']}")
        print(f"  Complex (multi attr LHS): {summary['complex_fds']}")
        
        print("\nOutput files created:")
        for filename in summary['output_files']:
            print(f"  [OK] {filename}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Prepare FD data for pipeline processing"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to parsed FD JSON file"
    )
    parser.add_argument(
        "--output",
        default="output/pipeline_ready",
        help="Output directory for pipeline-ready files"
    )
    
    args = parser.parse_args()
    
    print(f"Loading parsed FDs from: {args.input}")
    
    # Initialize pipeline prep
    prep = FDPipelinePrep(args.input)
    
    print(f"✓ Loaded {len(prep.fds)} FDs")
    
    # Export in pipeline format
    summary = prep.export_pipeline_format(args.output)
    
    # Print summary
    prep.print_pipeline_summary(summary)
    
    print(f"\n✓ Pipeline-ready files exported to: {args.output}")


if __name__ == "__main__":
    main()
