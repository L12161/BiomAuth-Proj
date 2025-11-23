"""
FVC Filename Parser Addon

Parses filenames in format: fvcYYYY_DBID_originalName.ext
Examples:
  - fvc2000_1_101_1.tif  → Year: 2000, DB: 1, Subject: 101, Impression: 1
  - fvc2002_3_103_8.tif  → Year: 2002, DB: 3, Subject: 103, Impression: 8
  - fvc2004_2_105_4.png  → Year: 2004, DB: 2, Subject: 105, Impression: 4

Add this function to train_fvc_transfer.py or use as import.
"""

import os
import re


def parse_fvc_filename(filename):
    """
    Parse FVC filename format: fvcYYYY_DBID_SUBJECT_IMPRESSION.ext
    
    Args:
        filename: e.g., 'fvc2002_1_101_3.tif'
    
    Returns:
        dict with keys: year, db_id, subject_id, impression_id
        or None if parsing fails
    """
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]  # Remove extension
    
    # Pattern: fvcYYYY_DBID_SUBJECT_IMPRESSION
    pattern = r'^fvc(\d{4})_(\d)_(\d+)_(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        return {
            'year': int(match.group(1)),
            'db_id': int(match.group(2)),
            'subject_id': int(match.group(3)),
            'impression_id': int(match.group(4)),
            'unique_subject_key': f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
        }
    
    return None


def get_subject_id_from_fvc(filename):
    """
    Extract unique subject identifier from FVC filename.
    
    Returns a unique key combining year, db, and subject number
    to handle cases where same subject numbers exist across DBs.
    """
    parsed = parse_fvc_filename(filename)
    if parsed:
        # Return unique key: "YYYY_DB_SUBJECT" as integer hash
        # Or just subject_id if all files are from same DB
        return parsed['subject_id']
    return None


# ============================================================
# REPLACEMENT CODE FOR train_fvc_transfer.py
# ============================================================
# Replace the parse_subject_id function in FVCDataset class with:

def parse_subject_id_fvc_format(filename):
    """
    Parse subject ID from FVC format: fvcYYYY_DBID_SUBJECT_IMPRESSION.ext
    
    Args:
        filename: Full path or just filename
    
    Returns:
        subject_id (int) or None
    """
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    
    # Pattern: fvcYYYY_DBID_SUBJECT_IMPRESSION
    pattern = r'^fvc(\d{4})_(\d)_(\d+)_(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        return int(match.group(3))  # Return subject number
    
    # Fallback: try original format (SUBJECT_IMPRESSION)
    parts = name.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[0].lstrip('s'))
        except ValueError:
            pass
    
    return None


# ============================================================
# TEST
# ============================================================
if __name__ == '__main__':
    # Test cases
    test_files = [
        'fvc2000_1_101_1.tif',
        'fvc2002_3_103_8.tif', 
        'fvc2004_2_105_4.png',
        'fvc2000_4_110_8.bmp',
        '/path/to/fvc2002_1_101_3.tif',
    ]
    
    print("Testing FVC filename parser:")
    print("-" * 60)
    
    for f in test_files:
        parsed = parse_fvc_filename(f)
        subject = parse_subject_id_fvc_format(f)
        print(f"{f}")
        print(f"  → Parsed: {parsed}")
        print(f"  → Subject ID: {subject}")
        print()
