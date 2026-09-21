"""Public API and CSV command-line interface."""
from pathlib import Path
import argparse
import pandas as pd
from hhsa_sohail_core import run_hhsa_sohail

def run_file(path,time_column="date",columns=None,output_dir="outputs",config=None,forecast_end=None):
    df=pd.read_csv(path); df[time_column]=pd.to_datetime(df[time_column])
    columns=columns or list(df.select_dtypes("number").columns)
    return {c:run_hhsa_sohail(df[time_column].to_numpy(),df[c].to_numpy(float),c,Path(output_dir)/c,config,forecast_end) for c in columns}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--time-column",default="date"); p.add_argument("--columns",nargs="+"); p.add_argument("--output",default="outputs"); p.add_argument("--forecast-end"); a=p.parse_args()
    run_file(a.input,a.time_column,a.columns,a.output,forecast_end=a.forecast_end)
if __name__=="__main__": main()
