import csv
import json

from experiments.expD34_readout_race import plateau_campaign_analysis as analysis


def test_summary_keeps_step_and_grid_controls_separate(tmp_path):
    def write(name, records):
        with (tmp_path/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    write('endpoints.csv',[dict(stage='checks',failed=False,complete=True,motion_identity=0.)])
    write('contrasts.csv',[dict(stage='checks',bundle=f'{kind}_600000',target='moment9',
        fork=600000,arm='joint',weight=1.,gamma_change_difference=value,
        eval_relative_mse_difference=value,reference_effective_norm_difference=value)
        for kind,value in [('half',.01),('grid',.5)]])
    analysis.summarize(tmp_path,tmp_path/'summary')
    records=json.loads((tmp_path/'summary/summary.json').read_text())['contrasts']
    assert len(records)==2
    assert {r['control']:r['gamma_change_difference_median'] for r in records}=={'half':.01,'grid':.5}
