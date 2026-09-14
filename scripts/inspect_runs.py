"""Print saved BC/DAgger scalar results without launching TensorBoard."""
import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logdir', nargs='?', type=Path, default=ROOT / 'hw1/data')
    args = parser.parse_args()
    folders = ([args.logdir] if list(args.logdir.glob('events.out.tfevents.*'))
               else sorted(p for p in args.logdir.iterdir() if p.is_dir()))
    for folder in folders:
        if not list(folder.glob('events.out.tfevents.*')):
            continue
        events = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
        tags = events.Tags()['scalars']
        if 'Eval_AverageReturn' not in tags:
            continue
        values = {tag: {e.step: e.value for e in events.Scalars(tag)} for tag in tags}
        print(f'\n{folder.name}')
        print('iter | eval mean | eval std | avg ep len | final batch MSE | % expert')
        for step, mean in values['Eval_AverageReturn'].items():
            std = values.get('Eval_StdReturn', {}).get(step, float('nan'))
            length = values.get('Eval_AverageEpLen', {}).get(step, float('nan'))
            loss = values.get('Training_Loss', {}).get(step, float('nan'))
            expert = values.get('Initial_DataCollection_AverageReturn', {}).get(step, float('nan'))
            ratio = 100 * mean / expert if expert else float('nan')
            print(f'{step:4d} | {mean:9.2f} | {std:8.2f} | {length:10.1f} | {loss:15.6f} | {ratio:8.2f}')
    print('\nCompare % expert only for runs using the full 1000-step episode limit; earlier smoke tests used 100.')


if __name__ == '__main__':
    main()
