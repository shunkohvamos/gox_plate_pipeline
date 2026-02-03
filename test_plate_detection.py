#!/usr/bin/env python3
"""Test script to verify plate detection in multi-plate CSV files."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gox_plate_pipeline.loader import extract_tidy_from_synergy_export

def test_3platedata():
    """Test plate detection in 3platedata.csv"""
    csv_path = Path("data/raw/260203-2/3platedata (2).csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    
    config = {"layout": {"rows": list("ABCDEFGH"), "max_col": 12}}
    
    try:
        df = extract_tidy_from_synergy_export(csv_path, config)
        
        # Check plate IDs
        plate_ids = sorted(df["plate_id"].unique())
        print(f"Detected plate IDs: {plate_ids}")
        print(f"Number of unique plates: {len(plate_ids)}")
        
        # Check data counts per plate
        for plate_id in plate_ids:
            plate_df = df[df["plate_id"] == plate_id]
            wells = sorted(plate_df["well"].unique())
            print(f"\n{plate_id}:")
            print(f"  Number of rows: {len(plate_df)}")
            print(f"  Number of unique wells: {len(wells)}")
            print(f"  Sample wells: {wells[:5]}...{wells[-5:] if len(wells) > 10 else wells}")
        
        if len(plate_ids) != 3:
            print(f"\nWARNING: Expected 3 plates, but found {len(plate_ids)}")
            print("This indicates a problem with plate detection.")
        else:
            print("\nSUCCESS: All 3 plates detected correctly!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3platedata()
