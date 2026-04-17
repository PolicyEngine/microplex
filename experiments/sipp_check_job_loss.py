"""Check if synthetics capture job loss patterns."""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.data_loaders import load_sipp
from experiments.sipp_inspect_holdouts import (
    prepare_sipp_panel, CombinedModel, generate_synth
)


def main():
    print("Loading SIPP...")
    sipp_raw = load_sipp(sample_frac=0.5)
    sipp = prepare_sipp_panel(sipp_raw)

    feature_cols = ['age', 'total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    n_features = len(feature_cols)
    n_periods = 6

    persons = sipp['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(persons)
    n_train = int(len(persons) * 0.8)
    train_df = sipp[sipp['person_id'].isin(persons[:n_train])]

    def filter_complete(df, n_periods):
        periods_per_person = df.groupby('person_id')['period'].nunique()
        complete = periods_per_person[periods_per_person >= n_periods].index
        df = df[df['person_id'].isin(complete)]
        df = df.sort_values(['person_id', 'period']).groupby('person_id').head(n_periods)
        return df

    train_df = filter_complete(train_df, n_periods)

    print("Training model...")
    model = CombinedModel(n_features)
    model.fit(train_df, feature_cols, epochs=100)

    print("Generating 2000 synthetics...")
    synth_df = generate_synth(model, train_df, feature_cols, 2000, n_periods, seed=42)

    # Check for job losers in synthetics
    print("\n" + "="*70)
    print("JOB LOSS PATTERNS IN SYNTHETICS (n=2000)")
    print("="*70)

    job_losers = []
    for pid in synth_df['person_id'].unique():
        person = synth_df[synth_df['person_id'] == pid].sort_values('period')
        job1 = person['job1_income'].values
        for t in range(len(job1) - 1):
            if job1[t] > 1000 and job1[t+1] == 0:
                job_losers.append((pid, t, job1[t], job1[t+1]))
                break

    print(f"Synthetics with job1 loss (>$1K → $0): {len(job_losers)}/{synth_df['person_id'].nunique()}")

    if job_losers:
        print("\nExamples of synthetic job losers:")
        for pid, t, before, after in job_losers[:5]:
            person = synth_df[synth_df['person_id'] == pid].sort_values('period')
            print(f"  Person {pid}: job1 = {list(person['job1_income'].round(0).astype(int))}")

    # Compare to training data
    print("\n" + "="*70)
    print("JOB LOSS PATTERNS IN TRAINING DATA")
    print("="*70)

    train_job_losers = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        job1 = person['job1_income'].values
        for t in range(len(job1) - 1):
            if job1[t] > 1000 and job1[t+1] == 0:
                train_job_losers.append(pid)
                break

    n_train_persons = train_df['person_id'].nunique()
    print(f"Training with job1 loss: {len(train_job_losers)}/{n_train_persons} ({100*len(train_job_losers)/n_train_persons:.1f}%)")

    # Expected in synthetics if matching training rate
    expected = 2000 * len(train_job_losers) / n_train_persons
    print(f"Expected in synthetics: {expected:.0f}")
    print(f"Actual in synthetics: {len(job_losers)}")

    # Also check income jumps (>2x increase)
    print("\n" + "="*70)
    print("INCOME JUMP PATTERNS (>2x increase)")
    print("="*70)

    synth_jumpers = []
    for pid in synth_df['person_id'].unique():
        person = synth_df[synth_df['person_id'] == pid].sort_values('period')
        inc = person['total_income'].values
        for t in range(len(inc) - 1):
            if inc[t] > 1000 and inc[t+1] > inc[t] * 2:
                synth_jumpers.append((pid, inc[t], inc[t+1]))
                break

    train_jumpers = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        inc = person['total_income'].values
        for t in range(len(inc) - 1):
            if inc[t] > 1000 and inc[t+1] > inc[t] * 2:
                train_jumpers.append(pid)
                break

    print(f"Training with >2x income jump: {len(train_jumpers)}/{n_train_persons} ({100*len(train_jumpers)/n_train_persons:.1f}%)")
    print(f"Synthetics with >2x income jump: {len(synth_jumpers)}/2000 ({100*len(synth_jumpers)/2000:.1f}%)")


if __name__ == "__main__":
    main()
