"""
Task Set 1 / Task Set 4 Part 1
FD Parser: parse FD lists (our discovery output or the provided Metanome
reference files), compute statistics and flag suspicious FDs.
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict


class FDParser:
    """
    Parses functional dependencies from various formats
    (Metanome output, existing FD files, etc.)
    """
    
    # An LHS with more than this many attributes is a "large determinant"
    LARGE_LHS = 3
    # A column whose distinct-value ratio is at least this is treated as an ID
    ID_UNIQUENESS = 0.90

    def __init__(self, column_profile=None):
        """
        column_profile (optional): {column: distinct_ratio} computed on the
        data. When given, ID-based FDs are detected from the data itself
        (near-unique columns) instead of only from LHS frequency.
        """
        self.fds = []
        self.statistics = {}
        self.column_profile = column_profile or {}
    
    def parse_metanome_output(self, file_path):
        """
        Parse FDs written by fd_discovery_stage / Metanome-like text output:
        - [attribute1, attribute2] -> attribute3
        - attribute1 -> attribute2
        - [] -> attribute   (empty LHS: constant column)
        Attribute names may contain dots, dashes or digits.
        """
        fds = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or '->' not in line:
                continue
            lhs_part, rhs_part = (p.strip() for p in line.rsplit('->', 1))
            lhs_part = lhs_part.strip('[]').strip()
            lhs = [a.strip() for a in lhs_part.split(',') if a.strip()]
            fds.append({"lhs": lhs, "rhs": rhs_part, "lhs_size": len(lhs)})

        self.fds = fds
        return fds

    def parse_existing_fd_file(self, file_path):
        """
        Parse the reference FD files provided with the assignment (Valid_FDS).

        Two Metanome formats are supported:
        - text: '# COLUMN' section mapping names to indices, then
          '# RESULTS' lines such as '3,2,1->5' (indices are mapped back to names)
        - JSON lines: {"type":"FunctionalDependency","determinant":..,"dependant":..}
        Plain 'a, b -> c' lines are also accepted.
        """
        fds = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f]

        index_to_name = {}
        section = None
        for line in lines:
            if not line:
                continue
            if line.startswith('#'):
                section = line.lstrip('#').strip().upper()
                continue
            if line.startswith('{'):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lhs = [c["columnIdentifier"] for c in obj["determinant"]["columnIdentifiers"]]
                rhs = obj["dependant"]["columnIdentifier"]
                fds.append({"lhs": lhs, "rhs": rhs, "lhs_size": len(lhs)})
                continue
            if section == "COLUMN" and '\t' in line:
                name, idx = line.rsplit('\t', 1)
                # Metanome prefixes names with '<table number>.'
                index_to_name[idx.strip()] = name.split('.', 1)[1] if '.' in name else name
                continue
            if '->' not in line:
                continue
            lhs_part, rhs_part = (p.strip() for p in line.rsplit('->', 1))
            lhs_part = lhs_part.replace('[', '').replace(']', '')
            rhs_part = rhs_part.replace('[', '').replace(']', '')
            lhs = [a.strip() for a in lhs_part.split(',') if a.strip()]
            lhs = [index_to_name.get(a, a) for a in lhs]
            rhs = index_to_name.get(rhs_part, rhs_part)
            if rhs:
                fds.append({"lhs": lhs, "rhs": rhs, "lhs_size": len(lhs)})

        self.fds = fds
        return fds

    def calculate_statistics(self):
        """
        Calculate statistics about the FDs
        """
        if not self.fds:
            return {}
        
        # Count FDs
        num_fds = len(self.fds)
        
        # Average LHS size
        lhs_sizes = [fd['lhs_size'] for fd in self.fds]
        avg_lhs_size = sum(lhs_sizes) / len(lhs_sizes) if lhs_sizes else 0
        
        # Attribute frequency
        lhs_freq = defaultdict(int)
        rhs_freq = defaultdict(int)
        
        for fd in self.fds:
            for attr in fd['lhs']:
                lhs_freq[attr] += 1
            rhs_freq[fd['rhs']] += 1
        
        # Find large determinants
        large_determinants = [
            fd for fd in self.fds if fd['lhs_size'] > self.LARGE_LHS
        ]
        
        # Identify potential ID-based FDs
        # (single attribute LHS that appears frequently)
        if self.column_profile:
            # Data-driven: near-unique columns are IDs / keys
            id_candidates = {
                attr: round(ratio, 3) for attr, ratio in self.column_profile.items()
                if ratio >= self.ID_UNIQUENESS
            }
        else:
            # Fallback heuristic: attribute appears in >10% of FD LHSs
            id_candidates = {
                attr: count for attr, count in lhs_freq.items()
                if count > num_fds * 0.1
            }
        
        self.statistics = {
            "num_fds": num_fds,
            "avg_lhs_size": round(avg_lhs_size, 2),
            "min_lhs_size": min(lhs_sizes) if lhs_sizes else 0,
            "max_lhs_size": max(lhs_sizes) if lhs_sizes else 0,
            "lhs_frequency": dict(lhs_freq),
            "rhs_frequency": dict(rhs_freq),
            "num_large_determinants": len(large_determinants),
            "large_determinants": large_determinants[:10],  # Top 10
            "id_candidates": id_candidates
        }
        
        return self.statistics
    
    def identify_suspicious_fds(self):
        """
        Identify potentially suspicious FDs
        
        Categories:
        - ID-based (single attribute determines many)
        - Large LHS (overfitting)
        - Rare attributes (spurious)
        """
        suspicious = {
            "id_based": [],
            "large_lhs": [],
            "rare_attributes": []
        }
        
        if not self.fds or not self.statistics:
            return suspicious
        
        # ID-based FDs
        id_candidates = self.statistics.get("id_candidates", {})
        for fd in self.fds:
            if self.column_profile:
                # any LHS that contains an ID column is degenerate
                is_id = any(a in id_candidates for a in fd['lhs'])
            else:
                is_id = fd['lhs_size'] == 1 and fd['lhs'][0] in id_candidates
            if is_id:
                suspicious["id_based"].append(fd)
        
        # Large LHS (same threshold as "large determinants")
        for fd in self.fds:
            if fd['lhs_size'] > self.LARGE_LHS:
                suspicious["large_lhs"].append(fd)
        
        # Rare attributes (appear in <3 FDs)
        lhs_freq = self.statistics.get("lhs_frequency", {})
        rare_attrs = {attr for attr, count in lhs_freq.items() if count < 3}
        
        for fd in self.fds:
            if any(attr in rare_attrs for attr in fd['lhs']):
                suspicious["rare_attributes"].append(fd)
        
        return suspicious
    
    def export_to_json(self, output_path):
        """Export parsed FDs and statistics to JSON"""
        data = {
            "fds": self.fds,
            "statistics": self.statistics,
            "suspicious": self.identify_suspicious_fds()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def export_to_csv(self, output_path):
        """Export FDs to CSV format"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['LHS', 'RHS', 'LHS_Size'])
            
            for fd in self.fds:
                lhs_str = ', '.join(fd['lhs'])
                writer.writerow([lhs_str, fd['rhs'], fd['lhs_size']])
        
        return output_path
    
    def print_summary(self):
        """Print a human-readable summary"""
        print("\n" + "="*60)
        print("FD PARSING SUMMARY")
        print("="*60)
        
        if not self.fds:
            print("No FDs found!")
            return
        
        stats = self.statistics
        print(f"\nTotal FDs: {stats.get('num_fds', 0)}")
        print(f"Average LHS size: {stats.get('avg_lhs_size', 0)}")
        print(f"LHS size range: {stats.get('min_lhs_size', 0)} - {stats.get('max_lhs_size', 0)}")
        
        print("\nTop 5 LHS attributes:")
        lhs_freq = stats.get('lhs_frequency', {})
        for attr, count in sorted(lhs_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {attr}: {count}")
        
        print("\nTop 5 RHS attributes:")
        rhs_freq = stats.get('rhs_frequency', {})
        for attr, count in sorted(rhs_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {attr}: {count}")
        
        suspicious = self.identify_suspicious_fds()
        print(f"\nSuspicious FDs:")
        print(f"  ID-based: {len(suspicious['id_based'])}")
        print(f"  Large LHS: {len(suspicious['large_lhs'])}")
        print(f"  Rare attributes: {len(suspicious['rare_attributes'])}")
        
        print("\nSample FDs:")
        for i, fd in enumerate(self.fds[:5], 1):
            lhs_str = ', '.join(fd['lhs'])
            print(f"  {i}. {lhs_str} -> {fd['rhs']}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Parse FD results from Metanome or existing FD files"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input FD file"
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON file (default: parsed_fds.json)"
    )
    parser.add_argument(
        "--format",
        choices=["metanome", "existing"],
        default="existing",
        help="Input format type"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export to CSV"
    )
    
    args = parser.parse_args()
    
    # Parse FDs
    fd_parser = FDParser()
    
    print(f"Parsing FDs from: {args.input}")
    
    if args.format == "metanome":
        fds = fd_parser.parse_metanome_output(args.input)
    else:
        fds = fd_parser.parse_existing_fd_file(args.input)
    
    print(f"✓ Parsed {len(fds)} FDs")
    
    # Calculate statistics
    fd_parser.calculate_statistics()
    
    # Print summary
    fd_parser.print_summary()
    
    # Export to JSON
    if args.output:
        output_json = args.output
    else:
        input_path = Path(args.input)
        output_json = input_path.parent / f"{input_path.stem}_parsed.json"
    
    fd_parser.export_to_json(output_json)
    print(f"\n✓ JSON exported to: {output_json}")
    
    # Export to CSV if requested
    if args.csv:
        output_csv = Path(output_json).with_suffix('.csv')
        fd_parser.export_to_csv(output_csv)
        print(f"✓ CSV exported to: {output_csv}")


if __name__ == "__main__":
    main()
