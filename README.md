## Modification Note

**Migrated [MADiff](https://github.com/zbzhu99/madiff) (CTCE) algorithm into the [OffPyMARL](https://github.com/zzq-bot/offline-marl-framework-offpymarl) training framework and conducted comparative training & evaluation in the StarCraft environment.**

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

### Evaluating

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



# Offline MARL framework - OffPyMARL

> 🚧 This repo is not ready for release, benchmarking is ongoing. 🚧

OffPyMARL provides unofficial and benchmarked PyTorch implementations for selected Offline MARL algorithms, including:

- [BC](https://arxiv.org/abs/1805.01954)
- [VDN/QMIX+CQL](https://arxiv.org/abs/2006.04779) (We provide three types of calculating coservatism term, "individual" and "global_simplified" are recommended) 
- [ITD3/MATD3+BC](https://arxiv.org/abs/2106.06860)
- [MAICQ](https://arxiv.org/abs/2106.03400)
- [OMAR](https://arxiv.org/abs/2111.11188) (Centralized Critic)
- [CFCQL](https://arxiv.org/abs/2309.12696)
- [OMIGA](https://arxiv.org/abs/2307.11620)

we also implement selected [Multi-Task versions](https://github.com/zzq-bot/mt_offpymarl) to tackle with the population-invariante issue for BC, QMIX+CQL and MATD3+BC.

## Installation

```bash
conda create -n offpymarl python=3.10 -y
conda activate offpymarl
pip install -r requirements.txt
bash install_sc2.sh # if you have not installed StarCraftII on your machine
bash install_smac.sh
```

## Collect Data
```bash
python src/main.py --collect --config=<alg> --env-config=sc2_collect --map_name=<map_name> --offline_data_quality=<quality> --save_replay_buffer=<whether_to_save_replay>
--num_episodes_collected=<num_episodes_per_collection> --stop_winrate=<stop_winrate> --seed=<seed>
```
quality is optinal in ['random', 'medium', 'expert', 'full'].

if save_replay_buffer is set, it will generate 'medium_replay', 'expert_replay' offline data with adequate offline_data_quality param.

## Offline Dataset

We provide the small-scale datasets (less than 4k episodes) in [Google Drive](https://drive.google.com/drive/folders/1FzSetZJ89Vq99o8LQHXiIxU9_tS70laE?usp=sharing) for a quick start.
After placing the full dataset in the dataset folder, you can run experiments using our predefined task sets.

Additionally, we now support the use of [OG-MARL](https://github.com/instadeepai/og-marl) datasets. To integrate this with the (off)pymarl pipeline, we have transformed it into an H5 file as demonstrated in `src/transform_data.ipynb` (please refer to this file for details).

Benchmarking on OG-MARL is currently in progress...




## Offline Training
```bash
python src/main.py --offline --config=<alg_name> --env-config=sc2_offline --map_name=<sc2_map>  --offline_data_quality=<data_quality> --seed=<seed> --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=100001 
```
see test/offline_train for more information.


## Citing OffPyMARL

If you use OffPyMARL your work, please use the following bibtex

```tex
@misc{OffPyMARL,
  author = {Ziqian Zhang},
  title = {OffPyMARL: Benchmarked Implementations of Offline Reinforcement Learning Algorithms},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zzq-bot/offline-marl-framwork-offpymarl}},
}
```

## Acknowledgements
We thank [ODIS](https://github.com/LAMDA-RL/ODIS) for providing data collection related code and [EPyMARL](https://github.com/uoe-agents/epymarl) for providing MADDPG related code. 

