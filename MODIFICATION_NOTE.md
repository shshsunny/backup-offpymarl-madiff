## Modification Note

**Migrated [MADiff](https://github.com/zbzhu99/madiff) (CTCE) algorithm into the [OffPyMARL](https://github.com/zzq-bot/offline-marl-framework-offpymarl) training framework and performed comparative training & evaluation in the StarCraft 2 environment.**

### Training

Train behavior policy & collect offline dataset:

```bash
python src/main.py --collect --config=qmix --env-config=sc2_collect --map_name=2m_vs_4m_split --offline_data_quality=expert --save_replay_buffer=true --num_episodes_collected=4000 --stop_winrate=0.9 --seed=0
```



Train different behavior policies (via reward shaping) & collect offline dataset:

```bash
sed -i '741s/.*/        pref = 1/' src/envs/smac/smac/env/starcraft2/starcraft2.py ; python src/main.py --collect --config=qmix --env-config=sc2_collect --map_name=2m_vs_4m_split --offline_data_quality=expert --save_replay_buffer=true --num_episodes_collected=2000 --stop_winrate=0.9 --seed=0 --save_model=true --save_model_interval=1000000 ; sed -i '741s/.*/        pref = 2/' src/envs/smac/smac/env/starcraft2/starcraft2.py ; python src/main.py --collect --config=qmix --env-config=sc2_collect --map_name=2m_vs_4m_split --offline_data_quality=expert --save_replay_buffer=true --num_episodes_collected=2000 --stop_winrate=0.9 --seed=0 --save_model=true --save_model_interval=1000000 ; 
```

Remember to **merge the dataset directories from the different runs** to use them together for offline training.



Train different behavior policies (via reward shaping) & collect offline dataset, **with IDs for different data sources** attached to observations:

```bash
sed -i '741s/.*/        pref = 1/' src/envs/smac/smac/env/starcraft2/starcraft2.py ; python src/main.py --collect --config=qmix --env-config=sc2_collect --map_name=2m_vs_4m_split --offline_data_quality=expert --save_replay_buffer=true --num_episodes_collected=2000 --stop_winrate=0.9 --seed=0 --save_model=true --save_model_interval=1000000 --obs_id_len=2 --obs_id_num=0 ; sed -i '741s/.*/        pref = 2/' src/envs/smac/smac/env/starcraft2/starcraft2.py ; python src/main.py --collect --config=qmix --env-config=sc2_collect --map_name=2m_vs_4m_split --offline_data_quality=expert --save_replay_buffer=true --num_episodes_collected=2000 --stop_winrate=0.9 --seed=0 --save_model=true --save_model_interval=1000000 --obs_id_len=2 --obs_id_num=1 ; 
```



Train offline, behavior cloning:

```bash
python src/main.py --offline --config=bc --env-config=sc2_offline --map_name=2m_vs_4m_split  --offline_data_quality=expert --seed=0 --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=100001
```



Train offline, MADiff diffusion policy:

```bash
sed -i '741s/.*/        pref = 0/' src/envs/smac/smac/env/starcraft2/starcraft2.py; python src/main.py --offline --config=madiff_ctce --env-config=sc2_offline --map_name=2m_vs_4m_split  --offline_data_quality=expert --seed=100 --log_interval=500 --runner_log_interval=500 --learner_log_interval=500 --save_model_interval=100001 --save_model=true \
	--test_nepisode=8 --test_interval=50000 --t_max=1000000 --offline_max_buffer_size=50000
```



To train offline on a dataset from `2` sources, attach `--obs_id_len=2` to the end of the command.



Evaluate a multi-agent policy online:

```bash
python src/main.py --online --config=[bc/madiff_ctce] --env-config=sc2 --map_name=2m_vs_4m_split --test_nepisode=16 --checkpoint_path=<model_directory_path> --evaluate=true --seed=0 --save_replay=true --runner=episode
```



**Cross-play** between behavior policy & offline-trained policy to testify their coordination ability:

```bash
python src/main.py --online --config=madiff_ctce --env-config=sc2_offline --map_name=2m_vs_4m_split --test_nepisode=16 --checkpoint_path=<offline_trained_model_directory_path> --checkpoint_path2=<behavior_policy_model_directory_path> --n_ego=1 --evaluate=true --seed=0 --save_replay=true --runner=episode_xp --ego_learner=madiff_learner --tm_learner=q_learner --agent2=rnn --obs_last_action2=true
```

Add `--obs_id_len` if necessary. `sed -i '741s/.*/        pref = <0/1>/' src/envs/smac/smac/env/starcraft2/starcraft2.py` and add  `--obs_id_num=<0/1>` if the behavior policy is trained with certain reward shaping preference and ID.



### Results

Result charts available in `exp-records/smac1.docx` and `exp-records/smac2.docx`.
