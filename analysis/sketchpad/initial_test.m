addpath ~/rds_rhythmic-sampling/gb/fieldtrip-20181118/
ft_defaults

cd ~/Documents/projects/saccades-attention/data/raw/191101
dataset = 'test.fif';

hdr = ft_read_header(dataset);
event = ft_read_event(dataset);

%%
cfg                         = [];
cfg.dataset                 = 'test.fif';
cfg.trialfun                = 'ft_trialfun_general'; % this is the default
cfg.trialdef.eventtype      = '?';
cfg = ft_definetrial(cfg);