from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveCycle import WaveCycle
from models.WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
from models.helpers import plot_pattern
from models.helpers import plot_cycle
import pandas as pd
import numpy as np

global df
df = pd.read_csv(r'data/BTC-USD (3).csv')
# df = pd.read_csv(r'GOOGL.csv')
idx_start = np.argmin(np.array(list(df['Low'])))

wa = WaveAnalyzer(df=df, verbose=False)
wave_options_impulse = WaveOptionsGenerator5(up_to=15)  # generates WaveOptions up to [15, 15, 15, 15, 15]
wave_options_correction = WaveOptionsGenerator3(up_to=9)

impulse = Impulse('impulse')
leading_diagonal = LeadingDiagonal('leading diagonal')
correction = Correction('correction')
rules_to_check = [impulse, correction]

print(f'Start at idx: {idx_start}')
print(f"will run up to {wave_options_impulse.number / 1e6}M combinations.")

# set up a set to store already found wave counts
# it can be the case, that 2 WaveOptions lead to the same WavePattern.
# This can be seen in a chart, where for example we try to skip more maxima as there are. In such a case
# e.g. [1,2,3,4,5] and [1,2,3,4,10] will lead to the same WavePattern (has same sub-wave structure, same begin / end,
# same high / low etc.
# If we find the same WavePattern, we skip and do not plot it

wavepatterns_up = list()
wavepatterns_down = list()
completeList = list()

# loop over all combinations of wave options [i,j,k,l,m] for impulsive waves sorted from small, e.g.  [0,1,...] to
# large e.g. [3,2, ...]

# Plotting Impulsive Wave
for new_option_impulse in wave_options_impulse.options_sorted:

    waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)

    if waves_up:
        wavepattern_up = WavePattern(waves_up, verbose=True)

        for rule in rules_to_check:

            if wavepattern_up.check_rule(rule):
                if wavepattern_up in wavepatterns_up:
                    continue
                else:
                    wavepatterns_up.append(wavepattern_up)
                    print(f'{rule.name} found: {new_option_impulse.values}')
                    cor_end = waves_up[4].idx_end
                    cor_date = waves_up[4].date_end
                    cor_high = waves_up[4].high
                    cor_low = waves_up[4].low

print(f"\n\n\nCor_end {cor_end}\n\n\n",)
print(f"Cor_date {cor_date}\n\n\n", type(cor_date), "\n\n\n\n")
end_date = str(cor_date)
print(f"end_date {end_date}\t", type(end_date), "\n\n\n\n")
end_date_idx = int(np.where(df['Date'] == end_date)[0])
print(f"End Date Df idx {end_date_idx} \n\n\n")
print(f"Cor_high {cor_high}\n\n\n")
print(f"Cor_low {cor_low}\n\n\n")

# Plotting Corrective Wave
wave_cycles = set()
for new_option_correction in wave_options_correction.options_sorted:
    waves_cor = wa.find_corrective_wave(idx_start=cor_end, wave_config=new_option_correction.values)

    if waves_cor:
        wavepattern_cor = WavePattern(waves_cor, verbose=True)

        for rule in rules_to_check:

            if wavepattern_cor.check_rule(rule):
                if wavepattern_cor in wavepatterns_down:
                    continue
                else:
                    wavepatterns_down.append(wavepattern_cor)
                    print(f'{rule.name} found: {new_option_correction.values}')

completeWave=WaveCycle(wavepatterns_up[-1],wavepatterns_down[-1])
# Plotting wave cycle
# plot_pattern(df=df, wave_pattern=wavepatterns_up[-1], title="Impulsive Wave")
# plot_pattern(df=df, wave_pattern=wavepatterns_down[-1], title="Corrective Wave")
# plot_cycle(df=df, wave_cycle=completeWave, title="Elliot Wave")
