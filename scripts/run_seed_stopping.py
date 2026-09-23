"""Sequential, logged reproduction of the recognizer and stopping seed check."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--existing-recognizer',action='store_true')
    args=parser.parse_args()
    root=Path('research/runs'); tag=f'20260906_seed{args.seed}'
    shared=root/f'20260906_shared_seed{args.seed}_v1'
    obs=root/f'{tag}_stopping_observations_v1';policy=root/f'{tag}_stopping_policy_v1'
    orders=root/f'{tag}_order_observations_v1';order_policy=root/f'{tag}_order_policy_v1'
    commands=[]
    if args.existing_recognizer:
        if not (shared/'summary.json').exists():
            raise ValueError('Existing recognizer must have completed')
        if json.loads((shared/'config.json').read_text())['seed']!=args.seed:
            raise ValueError('Seed mismatch')
    else:
        commands.append(('train_shared_emg',['--cache','data/public/grabmyo/cache_20260906_v1','--out',str(shared),'--seed',str(args.seed)]))
    commands += [
        ('verify_shared_development',['--run',str(shared),'--kind','neural']),
        ('pilot_calibration_stopping',['--prior',str(shared),'--out',str(obs)]),
        ('evaluate_calibration_stopping',['--features',str(obs),'--out',str(policy)]),
        ('verify_calibration_stopping',['--observations',str(obs),'--policies',str(policy)]),
        ('calibration_order_observations',['--prior',str(shared),'--original-observations',str(obs),'--out',str(orders)]),
        ('evaluate_calibration_stopping',['--features',str(orders),'--frozen-policies',str(policy),'--out',str(order_policy)]),
        ('verify_stopping_orders',['--observations',str(orders),'--policies',str(order_policy)])]
    ledger=[];ledger_path=root/f'{tag}_pipeline.json'
    if ledger_path.exists():
        raise ValueError('Pipeline ledger exists; preserve prior run and inspect before retry')
    for index,(module,arguments) in enumerate(commands):
        command=[sys.executable,'-B','-m','scripts.'+module,*arguments]
        log=root/f'{tag}_stage{index}_{module}.log'
        entry={'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),'log':str(log.resolve())}
        ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
        with log.open('x') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
        entry.update(completed_utc=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode)
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
